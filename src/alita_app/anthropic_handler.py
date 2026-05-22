"""Anthropic-based conversation handler for Reachy Mini.

Audio pipeline:
  Mic → silero-vad → ElevenLabs Scribe v2 STT → Anthropic Claude
      → ElevenLabs TTS (pcm_16000) → Speaker

Public interface mirrors the old OpenaiRealtimeHandler so the rest of the
app (main.py, console.py, headless_personality_ui.py) needs minimal changes.

Default state is CONVERSATION (app is launched explicitly when the user
wants to talk). HIBERNATE is an opt-in pause state, reachable via
`POST /api/sleep`, the SIGUSR2 signal, or the "alita schlaf" voice command.
Re-wake via `POST /api/wake` or SIGUSR1.

Local Piper binary + de_DE-eva_k-x_low.onnx voice file remain on disk at
~/Labor/scripts/piper/ as an offline fallback (re-enable manually if needed).
"""

import io
import re
import sys
import json
import base64
import asyncio
import logging
from typing import Any, Dict, Final, Tuple, Literal, Optional
from datetime import datetime

import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from anthropic import AsyncAnthropic
from numpy.typing import NDArray
from scipy.signal import resample
from elevenlabs.client import AsyncElevenLabs

from alita_app import store
from alita_app.config import config, set_custom_profile
from alita_app.state_machine import (
    AlitaState,
    StateMachine,
    detect_wake_command,
    random_wake_confirmation,
)
from alita_app.tools.core_tools import ToolDependencies, get_tool_specs
from alita_app.tools.background_tool_manager import (
    ToolCallRoutine,
    ToolNotification,
    BackgroundToolManager,
)


logger = logging.getLogger(__name__)

# ── Audio constants ────────────────────────────────────────────────────────────

STT_SAMPLE_RATE: Final[int] = 16_000   # mic → VAD → Scribe v2 (WAV 16 kHz mono)
TTS_SAMPLE_RATE: Final[int] = 16_000   # ElevenLabs pcm_16000 output

# ── VAD constants ──────────────────────────────────────────────────────────────

VAD_THRESHOLD: Final[float] = 0.5
MIN_SILENCE_MS: Final[int] = 600          # ms of silence that ends a turn
VAD_CHUNK_SAMPLES: Final[int] = 512       # silero-vad window at 16 kHz

# ── Conversation constants ─────────────────────────────────────────────────────

MAX_HISTORY_TURNS: Final[int] = 30        # turns loaded from / kept in store
MAX_CLAUDE_TOKENS: Final[int] = 1024
IDLE_SECONDS: Final[float] = 15.0        # idle threshold before self-initiated action
TOOL_TIMEOUT_S: Final[float] = 30.0      # max time to await a single tool result

# ── Sentence splitter ──────────────────────────────────────────────────────────

_SENT_END = re.compile(r"(?<=[.!?;:])\s+")


def _split_sentences(buf: str) -> Tuple[list[str], str]:
    """Split buf at sentence boundaries; return (complete_sentences, remainder)."""
    parts = _SENT_END.split(buf)
    if len(parts) <= 1:
        return [], buf
    return parts[:-1], parts[-1]


# ── Handler ────────────────────────────────────────────────────────────────────


class AnthropicHandler(AsyncStreamHandler):
    """Anthropic conversation handler — drop-in replacement for OpenaiRealtimeHandler."""

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=TTS_SAMPLE_RATE,
            input_sample_rate=STT_SAMPLE_RATE,
        )

        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        # Anthropic client (set in start_up)
        self.client: Optional[AsyncAnthropic] = None

        # ElevenLabs client (set in start_up)
        self.eleven: Optional[AsyncElevenLabs] = None

        # Output queue shared between _tts_and_queue and emit()
        self.output_queue: asyncio.Queue[
            Tuple[int, NDArray[np.int16]] | AdditionalOutputs
        ] = asyncio.Queue()

        # State machine
        self.state_machine = StateMachine()

        # VAD state
        self._vad_model: Any = None
        self._vad_iterator: Any = None
        self._speaking: bool = False
        self._speech_buf: list[NDArray] = []
        self._vad_buf: NDArray = np.array([], dtype=np.float32)
        self._call_lock: asyncio.Lock = asyncio.Lock()

        # In-memory conversation history (Anthropic Messages format)
        self._history: list[dict] = []

        # Pending tool futures: call_id → Future that resolves when tool completes
        self._pending_tool_futures: Dict[str, "asyncio.Future[dict]"] = {}

        # Background tool manager
        self.tool_manager = BackgroundToolManager()

        # Idle tracking
        self.last_activity_time: float = 0.0
        self._is_idle_call: bool = False

        # API key provenance (Gradio textbox flow)
        self._key_source: Literal["env", "textbox"] = "env"
        self._provided_api_key: Optional[str] = None

        # System prompt (built once in start_up)
        self._system_prompt: str = ""

        # Lifecycle
        self._shutdown_requested: bool = False
        self._connected_event: asyncio.Event = asyncio.Event()
        self.start_time: float = 0.0

    # ── fastrtc interface ──────────────────────────────────────────────────────

    def copy(self) -> "AnthropicHandler":
        return AnthropicHandler(self.deps, self.gradio_mode, self.instance_path)

    async def start_up(self) -> None:
        """Initialise client, VAD, Whisper, store, and history, then run idle loop."""
        # Idempotent: start_up may be called both proactively at app launch
        # (main.py) and reactively by fastrtc on websocket-connect.
        if self._connected_event.is_set():
            return
        self.start_time = asyncio.get_event_loop().time()
        self.last_activity_time = self.start_time

        # ── Resolve API key ────────────────────────────────────────────────────
        api_key = config.ANTHROPIC_API_KEY
        if self.gradio_mode and not api_key:
            await self.wait_for_args()  # type: ignore[no-untyped-call]
            args = list(self.latest_args)
            textbox_key = args[3] if len(args) > 3 and len(args[3]) > 0 else None
            if textbox_key:
                api_key = textbox_key
                self._key_source = "textbox"
                self._provided_api_key = textbox_key
        if not api_key or not api_key.strip():
            logger.warning("ANTHROPIC_API_KEY missing. Proceeding with placeholder (tests/offline).")
            api_key = "DUMMY"

        self.client = AsyncAnthropic(api_key=api_key)
        logger.info("Anthropic client initialised (model=%s)", config.MODEL_NAME)

        # ── ElevenLabs client (STT + TTS) ─────────────────────────────────────
        el_key = config.ELEVENLABS_API_KEY
        if not el_key or not el_key.strip():
            logger.warning("ELEVENLABS_API_KEY missing — audio I/O will be disabled.")
            self.eleven = None
        else:
            self.eleven = AsyncElevenLabs(api_key=el_key)
            logger.info(
                "ElevenLabs client initialised (stt=%s, tts=%s, voice_id=%s)",
                config.ELEVENLABS_STT_MODEL,
                config.ELEVENLABS_TTS_MODEL,
                config.ELEVENLABS_TTS_VOICE_ID or "<unset>",
            )

        # ── Build system prompt once ──────────────────────────────────────────
        try:
            from alita_app.build_prompt import build_system_prompt
            self._system_prompt = build_system_prompt()
            logger.info("System prompt loaded: %d chars", len(self._system_prompt))
        except FileNotFoundError as e:
            logger.error("Cannot start without system prompt: %s", e)
            sys.exit(1)

        # ── Load silero-vad ────────────────────────────────────────────────────
        try:
            from silero_vad import VADIterator, load_silero_vad  # type: ignore[import]
            self._vad_model = load_silero_vad()
            self._vad_iterator = VADIterator(
                self._vad_model,
                sampling_rate=STT_SAMPLE_RATE,
                threshold=VAD_THRESHOLD,
                min_silence_duration_ms=MIN_SILENCE_MS,
            )
            logger.info("silero-vad loaded")
        except Exception as e:
            logger.error("Failed to load silero-vad: %s — mic input disabled", e)

        # ── Init store and load history ────────────────────────────────────────
        try:
            store.init_db()
            self._history = store.as_chat_history(n=MAX_HISTORY_TURNS)
            logger.info("Store initialised, loaded %d history turns", len(self._history))
        except Exception as e:
            logger.error("Store init failed: %s — starting with empty history", e)
            self._history = []

        # ── Start background tool manager ──────────────────────────────────────
        self.tool_manager.start_up(tool_callbacks=[self._handle_tool_result])

        # Start active: head tracking on, movement enabled. HIBERNATE is now
        # opt-in via /api/sleep (see _enter_hibernate for the inverse setup).
        if self.deps.camera_worker is not None:
            self.deps.camera_worker.set_head_tracking_enabled(True)
        if self.deps.movement_manager is not None:
            self.deps.movement_manager.set_hibernating(False)

        self._connected_event.set()
        logger.info("AnthropicHandler ready (state=CONVERSATION, active)")

        # Keep running until shutdown (emit() drives idle; this task stays alive)
        try:
            while not self._shutdown_requested:
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass

    # ── Audio receive ──────────────────────────────────────────────────────────

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive mic frame, run VAD, and trigger STT+Claude on end-of-speech."""
        # HIBERNATE: discard all mic input — wake happens externally.
        if self.state_machine.state == AlitaState.HIBERNATE:
            return

        if self._vad_iterator is None or self.eleven is None:
            return

        input_rate, audio = frame

        # Flatten to 1-D (handles (N,), (N, 1), (N, C) — picks channel 0)
        if audio.ndim == 2:
            if audio.shape[1] > audio.shape[0]:
                audio = audio.T
            audio = audio[:, 0]
        else:
            audio = audio.flatten()

        # Resample to 16 kHz if needed
        if input_rate != STT_SAMPLE_RATE:
            audio = resample(audio, int(len(audio) * STT_SAMPLE_RATE / input_rate))

        # Convert to float32 in [-1, 1] for VAD
        if audio.dtype in (np.float32, np.float64):
            audio_f32 = audio.astype(np.float32)
        else:
            audio_f32 = audio.astype(np.float32) / 32768.0

        # Accumulate samples and feed 512-sample chunks to VAD
        self._vad_buf = np.concatenate([self._vad_buf, audio_f32])
        while len(self._vad_buf) >= VAD_CHUNK_SAMPLES:
            chunk = self._vad_buf[:VAD_CHUNK_SAMPLES]
            self._vad_buf = self._vad_buf[VAD_CHUNK_SAMPLES:]
            try:
                import torch  # silero-vad requires torch tensors
                tensor = torch.from_numpy(chunk)
                result = self._vad_iterator(tensor)
            except Exception:
                result = None

            if result:
                if "start" in result and not self._speaking:
                    self._speaking = True
                    self._speech_buf = []
                    if self.state_machine.is_active and self.deps.movement_manager:
                        self.deps.movement_manager.set_listening(True)
                    logger.debug("VAD: speech start (state=%s)", self.state_machine.state.value)

                if "end" in result and self._speaking:
                    self._speaking = False
                    if self.state_machine.is_active and self.deps.movement_manager:
                        self.deps.movement_manager.set_listening(False)
                    logger.debug("VAD: speech end (state=%s)", self.state_machine.state.value)
                    captured = np.concatenate(self._speech_buf) if self._speech_buf else np.array([], dtype=np.float32)
                    self._speech_buf = []
                    asyncio.create_task(self._process_speech(captured), name="stt-task")

            if self._speaking:
                self._speech_buf.append(chunk)

    # ── STT ───────────────────────────────────────────────────────────────────

    async def _process_speech(self, audio: NDArray) -> None:
        """Transcribe captured speech via Scribe v2 and route to Claude."""
        if len(audio) < STT_SAMPLE_RATE * 0.3:  # ignore < 0.3 s
            logger.debug("STT: audio too short (%d samples), skipping", len(audio))
            return

        if self._call_lock.locked():
            logger.debug("STT: Claude call in progress, dropping speech frame")
            return

        self.state_machine.touch()

        try:
            text = await self._scribe_transcribe(audio)
        except Exception as e:
            logger.error("STT error: %s", e)
            return

        if not text:
            logger.debug("STT: empty transcript, skipping")
            return

        # In-conversation voice commands (sleep / stay-awake)
        cmd = detect_wake_command(text)
        if cmd == "sleep":
            logger.info("Sleep command detected: %r", text)
            await self._tts_and_queue("Schlafe ein.")
            self._enter_hibernate()
            return
        if cmd == "conversation" and self.state_machine.state == AlitaState.LISTENING:
            logger.info("Stay-awake command: %r -> CONVERSATION", text)
            self.state_machine.transition(AlitaState.CONVERSATION)
            await self._tts_and_queue("Bin wach.")
            return

        logger.info("STT transcript: %r", text)
        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": text}))
        await self._call_claude(text)

        # LISTENING mode: return to HIBERNATE after one response
        if self.state_machine.state == AlitaState.LISTENING:
            self._enter_hibernate()

    async def _scribe_transcribe(self, audio: NDArray) -> str:
        """Send float32 mono audio @ 16 kHz to ElevenLabs Scribe v2 and return text."""
        if self.eleven is None:
            return ""

        # Build a WAV blob in-memory (Scribe wants a file-like upload).
        wav_buf = io.BytesIO()
        import soundfile as sf  # late import keeps top-of-file lean
        audio_f32 = audio.astype(np.float32, copy=False)
        sf.write(wav_buf, audio_f32, STT_SAMPLE_RATE, format="WAV", subtype="PCM_16")
        wav_buf.seek(0)
        wav_buf.name = "speech.wav"  # SDK uses .name for the multipart filename

        t0 = asyncio.get_event_loop().time()
        resp = await self.eleven.speech_to_text.convert(
            model_id=config.ELEVENLABS_STT_MODEL,
            file=wav_buf,
            language_code="de",
            tag_audio_events=False,
            diarize=False,
        )
        dt_ms = (asyncio.get_event_loop().time() - t0) * 1000
        text = (getattr(resp, "text", None) or "").strip()
        logger.info("STT latency=%.0fms, %d chars", dt_ms, len(text))
        return text

    # ── State transition helpers ──────────────────────────────────────────

    async def _on_wake(self) -> None:
        """Called when transitioning out of HIBERNATE — confirm + enable movement."""
        if self.deps.camera_worker is not None:
            self.deps.camera_worker.set_head_tracking_enabled(True)
        if self.deps.movement_manager is not None:
            self.deps.movement_manager.set_hibernating(False)
        confirmation = random_wake_confirmation()
        logger.info("Wake confirmation: %r", confirmation)
        await self._tts_and_queue(confirmation)

    def _enter_hibernate(self) -> None:
        """Transition to HIBERNATE — disable tracking, suppress movement."""
        self.state_machine.transition(AlitaState.HIBERNATE)
        if self.deps.camera_worker is not None:
            self.deps.camera_worker.set_head_tracking_enabled(False)
        if self.deps.movement_manager is not None:
            self.deps.movement_manager.set_hibernating(True)
        # Drop any speech buffered before sleep to avoid replay on next wake.
        self._speaking = False
        self._speech_buf = []
        self._vad_buf = np.array([], dtype=np.float32)

    # ── Public wake/sleep API (HTTP endpoints + signal handlers call into here) ──

    async def wake(self) -> None:
        """External trigger: transition HIBERNATE → CONVERSATION (open session)."""
        if self.state_machine.state != AlitaState.HIBERNATE:
            logger.debug("wake: not hibernating (state=%s)", self.state_machine.state.value)
            return
        logger.info("External wake → CONVERSATION")
        self.state_machine.transition(AlitaState.CONVERSATION)
        await self._on_wake()

    async def force_sleep(self) -> None:
        """External trigger: transition to HIBERNATE."""
        if self.state_machine.state == AlitaState.HIBERNATE:
            return
        logger.info("External sleep → HIBERNATE")
        await self._tts_and_queue("Schlafe ein.")
        self._enter_hibernate()

    # ── Claude call ────────────────────────────────────────────────────────────

    async def _call_claude(
        self,
        user_text: str,
        tool_choice: Optional[dict] = None,
        save_to_store: bool = True,
    ) -> None:
        """Send user_text to Claude, stream response through Piper, handle tools."""
        async with self._call_lock:
            if save_to_store:
                try:
                    store.save_turn("user", user_text)
                except Exception as e:
                    logger.warning("store.save_turn failed: %s", e)

            self._history.append({"role": "user", "content": user_text})
            full_assistant_text = ""

            try:
                _tool_rounds = 0
                _MAX_TOOL_ROUNDS = 3  # after this many tool-only rounds, force text
                _is_idle = self._is_idle_call  # capture before loop resets it

                while True:  # tool-call loop
                    sentence_buf = ""
                    collected_text = ""
                    final_msg = None

                    # Force text-only after too many consecutive tool rounds
                    # (but not for idle calls — those are intentionally tool-only)
                    force_text = _tool_rounds >= _MAX_TOOL_ROUNDS and not _is_idle

                    stream_kwargs: dict[str, Any] = dict(
                        model=config.MODEL_NAME,
                        max_tokens=MAX_CLAUDE_TOKENS,
                        system=self._system_prompt,
                        messages=self._history,
                    )
                    if force_text:
                        # Omit tools entirely so Claude MUST produce text.
                        # History likely ends with tool_result — add a nudge so Claude
                        # has a proper user turn to respond to.
                        logger.info("Forcing text-only response (after %d tool rounds)", _tool_rounds)
                        if self._history and self._history[-1].get("role") == "user":
                            last_content = self._history[-1].get("content")
                            if isinstance(last_content, list):
                                # tool_result list — append a text nudge
                                self._history.append({
                                    "role": "user",
                                    "content": "Bitte antworte jetzt mit Text.",
                                })
                    else:
                        tools = get_tool_specs()
                        tools.append({"type": "web_search_20250305", "name": "web_search"})
                        stream_kwargs["tools"] = tools
                        if tool_choice is not None:
                            stream_kwargs["tool_choice"] = tool_choice

                    try:
                        async with self.client.messages.stream(**stream_kwargs) as stream:  # type: ignore[union-attr]
                            async for text_chunk in stream.text_stream:
                                sentence_buf += text_chunk
                                collected_text += text_chunk
                                sentences, sentence_buf = _split_sentences(sentence_buf)
                                for sentence in sentences:
                                    logger.info("TTS sentence: %r", sentence)
                                    await self._tts_and_queue(sentence)

                            # flush trailing text
                            if sentence_buf.strip():
                                await self._tts_and_queue(sentence_buf)

                            final_msg = await stream.get_final_message()
                            logger.info(
                                "Claude stop_reason=%s, content_types=%s, collected_text=%r",
                                final_msg.stop_reason if final_msg else None,
                                [b.type for b in final_msg.content] if final_msg else [],
                                collected_text[:200] if collected_text else "",
                            )

                    except Exception as e:
                        logger.error("Claude streaming error: %s", e)
                        await self.output_queue.put(
                            AdditionalOutputs({"role": "assistant", "content": f"[Fehler: {e}]"})
                        )
                        break

                    full_assistant_text += collected_text

                    if final_msg is None or final_msg.stop_reason != "tool_use" or force_text:
                        break

                    # ── Handle tool calls ──────────────────────────────────────
                    tool_blocks = [b for b in final_msg.content if b.type == "tool_use"]

                    # Append assistant turn with tool_use content blocks
                    self._history.append({"role": "assistant", "content": final_msg.content})

                    tool_results = []
                    for block in tool_blocks:
                        await self.output_queue.put(
                            AdditionalOutputs({
                                "role": "assistant",
                                "content": f"Tool: {block.name}({json.dumps(block.input)})",
                                "metadata": {"title": f"Using {block.name}", "status": "pending"},
                            })
                        )
                        logger.info("Tool call: %s  args=%s", block.name, block.input)

                        # Register future BEFORE starting tool (avoids race if tool completes instantly)
                        loop = asyncio.get_running_loop()
                        fut: asyncio.Future[dict] = loop.create_future()
                        self._pending_tool_futures[block.id] = fut

                        await self.tool_manager.start_tool(
                            call_id=block.id,
                            tool_call_routine=ToolCallRoutine(
                                tool_name=block.name,
                                args_json_str=json.dumps(block.input),
                                deps=self.deps,
                            ),
                            is_idle_tool_call=self._is_idle_call,
                        )

                        result = await self._await_tool_result(block.id)

                        # If the tool returned a b64 image, send it as an
                        # image content block so Claude can actually see it.
                        if isinstance(result, dict) and "b64_im" in result:
                            tool_content = [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": result["b64_im"],
                                    },
                                },
                            ]
                        else:
                            tool_content = json.dumps(result)

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": tool_content,
                        })

                    self._history.append({"role": "user", "content": tool_results})
                    self._is_idle_call = False
                    _tool_rounds += 1

                    # Idle calls: one tool round is enough, don't chain
                    if _is_idle:
                        logger.debug("Idle call: breaking after 1 tool round")
                        break

            finally:
                # Remove the forced-text nudge if we added one
                if (
                    self._history
                    and self._history[-1].get("role") == "user"
                    and self._history[-1].get("content") == "Bitte antworte jetzt mit Text."
                ):
                    self._history.pop()

                if full_assistant_text:
                    if save_to_store:
                        try:
                            store.save_turn(
                                "assistant",
                                full_assistant_text,
                                model=config.MODEL_NAME,
                                route="claude",
                            )
                        except Exception as e:
                            logger.warning("store.save_turn (assistant) failed: %s", e)

                    # Append clean text turn to history (replaces streaming blocks)
                    self._history.append({"role": "assistant", "content": full_assistant_text})
                    await self.output_queue.put(
                        AdditionalOutputs({"role": "assistant", "content": full_assistant_text})
                    )

                self.last_activity_time = asyncio.get_event_loop().time()

    # ── Tool result handling ───────────────────────────────────────────────────

    async def _handle_tool_result(self, bg_tool: ToolNotification) -> None:
        """Called by BackgroundToolManager when a tool finishes."""
        result: dict
        if bg_tool.error is not None:
            logger.error("Tool '%s' (id=%s) failed: %s", bg_tool.tool_name, bg_tool.id, bg_tool.error)
            result = {"error": bg_tool.error}
        elif bg_tool.result is not None:
            result = bg_tool.result
            logger.info("Tool '%s' (id=%s) completed", bg_tool.tool_name, bg_tool.id)
        else:
            result = {"error": "No result returned"}

        # Notify UI
        await self.output_queue.put(
            AdditionalOutputs({
                "role": "assistant",
                "content": json.dumps(result),
                "metadata": {"title": f"Tool {bg_tool.tool_name} done", "status": "done"},
            })
        )

        # Resolve pending future so _call_claude can continue
        fut = self._pending_tool_futures.pop(bg_tool.id, None)
        if fut is not None and not fut.done():
            fut.set_result(result)

        # Re-sync head wobble after tool (motion may have ended)
        if self.deps.head_wobbler is not None:
            self.deps.head_wobbler.reset()

    async def _await_tool_result(self, call_id: str) -> dict:
        """Wait for the tool identified by call_id to complete."""
        fut = self._pending_tool_futures.get(call_id)
        if fut is None:
            return {"error": f"No future registered for call_id={call_id}"}
        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout=TOOL_TIMEOUT_S)
        except asyncio.TimeoutError:
            self._pending_tool_futures.pop(call_id, None)
            logger.warning("Tool %s timed out after %ss", call_id, TOOL_TIMEOUT_S)
            return {"error": "Tool timed out"}

    # ── TTS ───────────────────────────────────────────────────────────────────

    async def _tts_and_queue(self, text: str) -> None:
        """Synthesise text via ElevenLabs streaming and push audio frames to output_queue."""
        text = text.strip()
        if not text:
            return

        if self.eleven is None:
            logger.warning("TTS skipped: ElevenLabs client not configured")
            return
        voice_id = config.ELEVENLABS_TTS_VOICE_ID
        if not voice_id:
            logger.warning("TTS skipped: ELEVENLABS_TTS_VOICE_ID not set")
            return

        t0 = asyncio.get_event_loop().time()
        first_chunk_at: Optional[float] = None
        total_bytes = 0
        pending = bytearray()
        try:
            stream = self.eleven.text_to_speech.stream(
                voice_id=voice_id,
                text=text,
                model_id=config.ELEVENLABS_TTS_MODEL,
                output_format="pcm_16000",
                language_code="de",
            )
            async for chunk in stream:
                if not chunk:
                    continue
                if first_chunk_at is None:
                    first_chunk_at = asyncio.get_event_loop().time()
                pending.extend(chunk)
                total_bytes += len(chunk)
                # Emit on 2-byte (int16) boundary, otherwise frombuffer fails.
                emit_len = len(pending) - (len(pending) % 2)
                if emit_len == 0:
                    continue
                frame = bytes(pending[:emit_len])
                del pending[:emit_len]

                audio = np.frombuffer(frame, dtype=np.int16).reshape(1, -1)
                if self.deps.head_wobbler is not None:
                    self.deps.head_wobbler.feed(base64.b64encode(frame).decode())
                await self.output_queue.put((TTS_SAMPLE_RATE, audio))
        except Exception as e:
            logger.error("ElevenLabs TTS error: %s", e)
            return

        if total_bytes == 0:
            logger.warning("TTS: empty stream for %r", text[:80])
            return
        first_ms = ((first_chunk_at or t0) - t0) * 1000
        total_ms = (asyncio.get_event_loop().time() - t0) * 1000
        logger.info(
            "TTS first_chunk=%.0fms total=%.0fms bytes=%d text=%r",
            first_ms, total_ms, total_bytes, text[:80],
        )

    # ── Emit (fastrtc periodic pull) ───────────────────────────────────────────

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Return next audio frame or metadata; triggers idle signal when quiet."""
        # Check conversation timeout
        if self.state_machine.state == AlitaState.CONVERSATION:
            if self.state_machine.check_conversation_timeout():
                logger.info("Conversation timeout — entering HIBERNATE")
                await self._tts_and_queue("Ich schlafe jetzt ein.")
                self._enter_hibernate()

        now = asyncio.get_event_loop().time()
        idle_duration = now - self.last_activity_time

        # Only send idle signals in CONVERSATION mode
        if (
            self.state_machine.state == AlitaState.CONVERSATION
            and idle_duration > IDLE_SECONDS
            and not self._call_lock.locked()
            and self.deps.movement_manager.is_idle()
        ):
            try:
                await self._send_idle_signal(idle_duration)
            except Exception as e:
                logger.warning("Idle signal skipped: %s", e)
            self.last_activity_time = asyncio.get_event_loop().time()

        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    async def _send_idle_signal(self, idle_duration: float) -> None:
        """Inject an idle nudge into the conversation (tool-only response)."""
        ts = datetime.now().strftime("%H:%M:%S")
        msg = (
            f"[System {ts} — {idle_duration:.0f}s Stille] "
            "Du warst eine Weile idle. Mach was — eine Emotion, schau dich um, oder sei einfach still."
        )
        logger.debug("Sending idle signal")
        self._is_idle_call = True
        asyncio.create_task(
            self._call_claude(
                msg,
                tool_choice={"type": "any"},
                save_to_store=False,
            ),
            name="idle-call",
        )

    # ── Shutdown ──────────────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        self._shutdown_requested = True
        await self.tool_manager.shutdown()

        # Resolve any pending futures so nothing hangs
        for fut in self._pending_tool_futures.values():
            if not fut.done():
                fut.set_result({"error": "shutdown"})
        self._pending_tool_futures.clear()

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # ── Personality / voice stubs (Gradio compat) ──────────────────────────────

    async def apply_personality(self, profile: Optional[str]) -> str:
        """Update active profile. Instructions are read fresh on next Claude call."""
        try:
            set_custom_profile(profile)
            logger.info("Personality applied: %r", profile)
            return "Persönlichkeit aktualisiert. Gilt ab dem nächsten Turn."
        except Exception as e:
            logger.error("apply_personality error: %s", e)
            return f"Fehler: {e}"

    async def get_available_voices(self) -> list[str]:
        """Return available TTS voices. Stub for Gradio voice dropdown."""
        return ["de_DE-eva_k-x_low"]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def format_timestamp(self) -> str:
        elapsed = asyncio.get_event_loop().time() - self.start_time
        return f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed:.1f}s]"
