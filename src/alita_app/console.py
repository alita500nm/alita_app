"""Bidirectional local audio stream with optional settings UI.

In headless mode, there is no Gradio UI. If the OpenAI API key is not
available via environment/.env, we expose a minimal settings page via the
Reachy Mini Apps settings server to let non-technical users enter it.

The settings UI is served from this package's ``static/`` folder and offers a
single password field to set ``ANTHROPIC_API_KEY``. Once set, we persist it to the
app instance's ``.env`` file (if available) and proceed to start streaming.
"""

import os
import sys
import time
import asyncio
import logging
from typing import List, Optional
from pathlib import Path

import cv2
from fastrtc import AdditionalOutputs, audio_to_float32
from scipy.signal import resample

from reachy_mini import ReachyMini
from reachy_mini.media.media_manager import MediaBackend
from alita_app.config import LOCKED_PROFILE, config
from alita_app.anthropic_handler import AnthropicHandler
from alita_app.headless_personality_ui import mount_personality_routes


try:
    # FastAPI is provided by the Reachy Mini Apps runtime
    from fastapi import FastAPI, Response
    from pydantic import BaseModel
    from fastapi.responses import FileResponse, JSONResponse
    from starlette.staticfiles import StaticFiles
except Exception:  # pragma: no cover - only loaded when settings_app is used
    FastAPI = object  # type: ignore
    FileResponse = object  # type: ignore
    JSONResponse = object  # type: ignore
    StaticFiles = object  # type: ignore
    BaseModel = object  # type: ignore


logger = logging.getLogger(__name__)


class LocalStream:
    """LocalStream using Reachy Mini's recorder/player."""

    def __init__(
        self,
        handler: AnthropicHandler,
        robot: ReachyMini,
        *,
        settings_app: Optional[FastAPI] = None,
        instance_path: Optional[str] = None,
    ):
        """Initialize the stream with an OpenAI realtime handler and pipelines.

        - ``settings_app``: the Reachy Mini Apps FastAPI to attach settings endpoints.
        - ``instance_path``: directory where per-instance ``.env`` should be stored.
        """
        self.handler = handler
        self._robot = robot
        self._stop_event = asyncio.Event()
        self._tasks: List[asyncio.Task[None]] = []
        # Allow the handler to flush the player queue when appropriate.
        self.handler._clear_queue = self.clear_audio_queue
        self._settings_app: Optional[FastAPI] = settings_app
        self._instance_path: Optional[str] = instance_path
        self._settings_initialized = False
        self._asyncio_loop = None

        # Register /stream MJPEG endpoint
        if self._settings_app is not None:
            self._register_mjpeg_stream(self._settings_app)
        else:
            self._start_standalone_mjpeg()

    def _mjpeg_gen(self):
        """Yield MJPEG frames from camera_worker."""
        cam = self.handler.deps.camera_worker
        while True:
            frame = cam.get_latest_frame() if cam else None
            if frame is not None:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(0.066)  # ~15 fps

    def _register_mjpeg_stream(self, app) -> None:
        """Register GET /stream MJPEG endpoint on a FastAPI app."""
        from fastapi.responses import StreamingResponse
        streamer = self

        @app.get("/stream")
        def _mjpeg_stream():
            return StreamingResponse(streamer._mjpeg_gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    def _start_standalone_mjpeg(self, port: int = 8001) -> None:
        """Spin up a lightweight MJPEG server on its own thread when no settings_app exists."""
        import threading
        import uvicorn

        app = FastAPI()
        self._register_mjpeg_stream(app)

        thread = threading.Thread(
            target=uvicorn.run, args=(app,),
            kwargs={"host": "0.0.0.0", "port": port, "log_level": "warning"},
            daemon=True,
        )
        thread.start()
        logger.info("Standalone MJPEG server on http://0.0.0.0:%d/stream", port)

    # ---- Settings UI (only when API key is missing) ----
    def _read_env_lines(self, env_path: Path) -> list[str]:
        """Load env file contents or a template as a list of lines."""
        inst = env_path.parent
        try:
            if env_path.exists():
                try:
                    return env_path.read_text(encoding="utf-8").splitlines()
                except Exception:
                    return []
            template_text = None
            ex = inst / ".env.example"
            if ex.exists():
                try:
                    template_text = ex.read_text(encoding="utf-8")
                except Exception:
                    template_text = None
            if template_text is None:
                try:
                    cwd_example = Path.cwd() / ".env.example"
                    if cwd_example.exists():
                        template_text = cwd_example.read_text(encoding="utf-8")
                except Exception:
                    template_text = None
            if template_text is None:
                packaged = Path(__file__).parent / ".env.example"
                if packaged.exists():
                    try:
                        template_text = packaged.read_text(encoding="utf-8")
                    except Exception:
                        template_text = None
            return template_text.splitlines() if template_text else []
        except Exception:
            return []

    def _persist_api_key(self, key: str) -> None:
        """Persist API key to environment and instance ``.env`` if possible.

        Behavior:
        - Always sets ``ANTHROPIC_API_KEY`` in process env and in-memory config.
        - Writes/updates ``<instance_path>/.env``:
          * If ``.env`` exists, replaces/append ANTHROPIC_API_KEY line.
          * Else, copies template from ``<instance_path>/.env.example`` when present,
            otherwise falls back to the packaged template
            ``alita_app/.env.example``.
          * Ensures the resulting file contains the full template plus the key.
        - Loads the written ``.env`` into the current process environment.
        """
        k = (key or "").strip()
        if not k:
            return
        # Update live process env and config so consumers see it immediately
        try:
            os.environ["ANTHROPIC_API_KEY"] = k
        except Exception:  # best-effort
            pass
        try:
            config.ANTHROPIC_API_KEY = k
        except Exception:
            pass

        if not self._instance_path:
            return
        try:
            inst = Path(self._instance_path)
            env_path = inst / ".env"
            lines = self._read_env_lines(env_path)
            replaced = False
            for i, ln in enumerate(lines):
                if ln.strip().startswith("ANTHROPIC_API_KEY="):
                    lines[i] = f"ANTHROPIC_API_KEY={k}"
                    replaced = True
                    break
            if not replaced:
                lines.append(f"ANTHROPIC_API_KEY={k}")
            final_text = "\n".join(lines) + "\n"
            env_path.write_text(final_text, encoding="utf-8")
            logger.info("Persisted ANTHROPIC_API_KEY to %s", env_path)

            # Load the newly written .env into this process to ensure downstream imports see it
            try:
                from dotenv import load_dotenv

                load_dotenv(dotenv_path=str(env_path), override=True)
            except Exception:
                pass
        except Exception as e:
            logger.warning("Failed to persist ANTHROPIC_API_KEY: %s", e)

    def _persist_personality(self, profile: Optional[str]) -> None:
        """Persist the startup personality to the instance .env and config."""
        if LOCKED_PROFILE is not None:
            return
        selection = (profile or "").strip() or None
        try:
            from alita_app.config import set_custom_profile

            set_custom_profile(selection)
        except Exception:
            pass

        if not self._instance_path:
            return
        try:
            env_path = Path(self._instance_path) / ".env"
            lines = self._read_env_lines(env_path)
            replaced = False
            for i, ln in enumerate(list(lines)):
                if ln.strip().startswith("REACHY_MINI_CUSTOM_PROFILE="):
                    if selection:
                        lines[i] = f"REACHY_MINI_CUSTOM_PROFILE={selection}"
                    else:
                        lines.pop(i)
                    replaced = True
                    break
            if selection and not replaced:
                lines.append(f"REACHY_MINI_CUSTOM_PROFILE={selection}")
            if selection is None and not env_path.exists():
                return
            final_text = "\n".join(lines) + "\n"
            env_path.write_text(final_text, encoding="utf-8")
            logger.info("Persisted startup personality to %s", env_path)
            try:
                from dotenv import load_dotenv

                load_dotenv(dotenv_path=str(env_path), override=True)
            except Exception:
                pass
        except Exception as e:
            logger.warning("Failed to persist REACHY_MINI_CUSTOM_PROFILE: %s", e)

    def _read_persisted_personality(self) -> Optional[str]:
        """Read persisted startup personality from instance .env (if any)."""
        if not self._instance_path:
            return None
        env_path = Path(self._instance_path) / ".env"
        try:
            if env_path.exists():
                for ln in env_path.read_text(encoding="utf-8").splitlines():
                    if ln.strip().startswith("REACHY_MINI_CUSTOM_PROFILE="):
                        _, _, val = ln.partition("=")
                        v = val.strip()
                        return v or None
        except Exception:
            pass
        return None

    def _init_settings_ui_if_needed(self) -> None:
        """Attach minimal settings UI to the settings app.

        Always mounts the UI when a settings_app is provided so that users
        see a confirmation message even if the API key is already configured.
        """
        if self._settings_initialized:
            return
        if self._settings_app is None:
            return

        static_dir = Path(__file__).parent / "static"
        index_file = static_dir / "index.html"

        if hasattr(self._settings_app, "mount"):
            try:
                # Serve /static/* assets
                self._settings_app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
            except Exception:
                pass

        class ApiKeyPayload(BaseModel):
            openai_api_key: str  # field name kept for UI/API compat; holds ANTHROPIC_API_KEY value

        # GET / -> index.html
        @self._settings_app.get("/")
        def _root() -> FileResponse:
            return FileResponse(str(index_file))

        # GET /favicon.ico -> optional, avoid noisy 404s on some browsers
        @self._settings_app.get("/favicon.ico")
        def _favicon() -> Response:
            return Response(status_code=204)

        # GET /status -> whether key is set
        @self._settings_app.get("/status")
        def _status() -> JSONResponse:
            has_key = bool(config.ANTHROPIC_API_KEY and str(config.ANTHROPIC_API_KEY).strip())
            return JSONResponse({"has_key": has_key})

        # GET /ready -> whether backend finished loading tools
        @self._settings_app.get("/ready")
        def _ready() -> JSONResponse:
            try:
                mod = sys.modules.get("alita_app.tools.core_tools")
                ready = bool(getattr(mod, "_TOOLS_INITIALIZED", False)) if mod else False
            except Exception:
                ready = False
            return JSONResponse({"ready": ready})

        # POST /anthropic_api_key -> set/persist key
        @self._settings_app.post("/anthropic_api_key")
        def _set_key(payload: ApiKeyPayload) -> JSONResponse:
            key = (payload.openai_api_key or "").strip()
            if not key:
                return JSONResponse({"ok": False, "error": "empty_key"}, status_code=400)
            self._persist_api_key(key)
            return JSONResponse({"ok": True})

        # POST /validate_api_key -> validate key without persisting it
        @self._settings_app.post("/validate_api_key")
        async def _validate_key(payload: ApiKeyPayload) -> JSONResponse:
            key = (payload.openai_api_key or "").strip()
            if not key:
                return JSONResponse({"valid": False, "error": "empty_key"}, status_code=400)

            # Try to validate by listing Anthropic models
            try:
                import httpx

                headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get("https://api.anthropic.com/v1/models", headers=headers)
                    if response.status_code == 200:
                        return JSONResponse({"valid": True})
                    elif response.status_code == 401:
                        return JSONResponse({"valid": False, "error": "invalid_api_key"}, status_code=401)
                    else:
                        return JSONResponse(
                            {"valid": False, "error": "validation_failed"}, status_code=response.status_code
                        )
            except Exception as e:
                logger.warning(f"API key validation failed: {e}")
                return JSONResponse({"valid": False, "error": "validation_error"}, status_code=500)

        self._settings_initialized = True

    def launch(self) -> None:
        """Start the recorder/player and run the async processing loops.

        If the OpenAI key is missing, expose a tiny settings UI via the
        Reachy Mini settings server to collect it before starting streams.
        """
        self._stop_event.clear()

        # Try to load an existing instance .env first (covers subsequent runs)
        if self._instance_path:
            try:
                from dotenv import load_dotenv

                from alita_app.config import set_custom_profile

                env_path = Path(self._instance_path) / ".env"
                if env_path.exists():
                    load_dotenv(dotenv_path=str(env_path), override=True)
                    # Update config with newly loaded values
                    new_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
                    if new_key:
                        try:
                            config.ANTHROPIC_API_KEY = new_key
                        except Exception:
                            pass
                    if LOCKED_PROFILE is None:
                        new_profile = os.getenv("REACHY_MINI_CUSTOM_PROFILE")
                        if new_profile is not None:
                            try:
                                set_custom_profile(new_profile.strip() or None)
                            except Exception:
                                pass  # Best-effort profile update
            except Exception:
                pass  # Instance .env loading is optional; continue with defaults

        # If key is still missing, try to download one from HuggingFace
        if not (config.ANTHROPIC_API_KEY and str(config.ANTHROPIC_API_KEY).strip()):
            logger.info("ANTHROPIC_API_KEY not set, attempting to download from HuggingFace...")
            try:
                from gradio_client import Client
                client = Client("HuggingFaceM4/gradium_setup", verbose=False)
                key, status = client.predict(api_name="/claim_b_key")
                if key and key.strip():
                    logger.info("Successfully downloaded API key from HuggingFace")
                    # Persist it immediately
                    self._persist_api_key(key)
            except Exception as e:
                logger.warning(f"Failed to download API key from HuggingFace: {e}")

        # Always expose settings UI if a settings app is available
        # (do this AFTER loading/downloading the key so status endpoint sees the right value)
        self._init_settings_ui_if_needed()

        # If key is still missing -> wait until provided via the settings UI
        if not (config.ANTHROPIC_API_KEY and str(config.ANTHROPIC_API_KEY).strip()):
            logger.warning("ANTHROPIC_API_KEY not found. Open the app settings page to enter it.")
            # Poll until the key becomes available (set via the settings UI)
            try:
                while not (config.ANTHROPIC_API_KEY and str(config.ANTHROPIC_API_KEY).strip()):
                    time.sleep(0.2)
            except KeyboardInterrupt:
                logger.info("Interrupted while waiting for API key.")
                return

        # Start media after key is set/available
        logger.info("DIAG launch: starting media pipelines (backend=%s)", self._robot.media.backend)
        self._robot.media.start_recording()
        self._robot.media.start_playing()
        time.sleep(1)  # give some time to the pipelines to start
        # Check pipeline states after startup
        try:
            audio = self._robot.media.audio
            if hasattr(audio, '_pipeline_record'):
                rec_state = audio._pipeline_record.get_state(0)
                play_state = audio._pipeline_playback.get_state(0)
                logger.info("DIAG launch: record_pipeline=%s playback_pipeline=%s", rec_state, play_state)
        except Exception as e:
            logger.warning("DIAG launch: pipeline state check failed: %s", e)
        logger.info("DIAG launch: media pipelines started, entering asyncio.run")

        async def runner() -> None:
            logger.info("DIAG runner: entered")
            # Capture loop for cross-thread personality actions
            loop = asyncio.get_running_loop()
            self._asyncio_loop = loop  # type: ignore[assignment]
            # Mount personality routes now that loop and handler are available
            try:
                if self._settings_app is not None:
                    mount_personality_routes(
                        self._settings_app,
                        self.handler,
                        lambda: self._asyncio_loop,
                        persist_personality=self._persist_personality,
                        get_persisted_personality=self._read_persisted_personality,
                    )
            except Exception:
                pass
            logger.info("DIAG runner: creating tasks")
            self._tasks = [
                asyncio.create_task(self.handler.start_up(), name="openai-handler"),
                asyncio.create_task(self.record_loop(), name="stream-record-loop"),
                asyncio.create_task(self.play_loop(), name="stream-play-loop"),
            ]
            logger.info("DIAG runner: awaiting gather")
            try:
                await asyncio.gather(*self._tasks)
            except asyncio.CancelledError:
                logger.info("Tasks cancelled during shutdown")
            finally:
                # Ensure handler connection is closed
                await self.handler.shutdown()

        asyncio.run(runner())

    def close(self) -> None:
        """Stop the stream and underlying media pipelines.

        This method:
        - Stops audio recording and playback first
        - Sets the stop event to signal async loops to terminate
        - Cancels all pending async tasks (openai-handler, record-loop, play-loop)
        """
        logger.info("Stopping LocalStream...")

        # Stop media pipelines FIRST before cancelling async tasks
        # This ensures clean shutdown before PortAudio cleanup
        try:
            self._robot.media.stop_recording()
        except Exception as e:
            logger.debug(f"Error stopping recording (may already be stopped): {e}")

        try:
            self._robot.media.stop_playing()
        except Exception as e:
            logger.debug(f"Error stopping playback (may already be stopped): {e}")

        # Now signal async loops to stop
        self._stop_event.set()

        # Cancel all running tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

    def clear_audio_queue(self) -> None:
        """Flush the player's appsrc to drop any queued audio immediately."""
        logger.info("User intervention: flushing player queue")
        if self._robot.media.backend == MediaBackend.GSTREAMER:
            # Directly flush gstreamer audio pipe
            self._robot.media.audio.clear_player()
        elif self._robot.media.backend == MediaBackend.DEFAULT or self._robot.media.backend == MediaBackend.DEFAULT_NO_VIDEO:
            self._robot.media.audio.clear_output_buffer()
        self.handler.output_queue = asyncio.Queue()

    async def record_loop(self) -> None:
        """Read mic frames from the recorder and forward them to the handler."""
        logger.info("DIAG record_loop: entered")
        input_sample_rate = self._robot.media.get_input_audio_samplerate()
        logger.info("DIAG record_loop: sample_rate=%d, starting loop", input_sample_rate)
        # Check GStreamer pipeline state
        try:
            audio = self._robot.media.audio
            if hasattr(audio, '_pipeline_record'):
                state = audio._pipeline_record.get_state(0)
                logger.info("DIAG record_loop: recording pipeline state=%s", state)
            else:
                logger.info("DIAG record_loop: no _pipeline_record attribute")
        except Exception as e:
            logger.warning("DIAG record_loop: pipeline state check failed: %s", e)

        _rec_frames = 0
        _rec_nones = 0
        _consecutive_nones = 0
        _last_status_time = 0.0
        import time as _time
        loop = asyncio.get_event_loop()
        while not self._stop_event.is_set():
            audio_frame = await loop.run_in_executor(
                None, self._robot.media.get_audio_sample
            )
            now = _time.monotonic()
            if audio_frame is not None:
                _rec_frames += 1
                if _consecutive_nones > 10:
                    logger.info(
                        "DIAG record_loop: audio RESUMED after %d consecutive Nones (total frames=%d)",
                        _consecutive_nones, _rec_frames,
                    )
                _consecutive_nones = 0
                if _rec_frames == 1:
                    logger.info(
                        "DIAG record_loop: first frame shape=%s dtype=%s max=%.4f",
                        audio_frame.shape, audio_frame.dtype, float(abs(audio_frame).max()),
                    )
                await self.handler.receive((input_sample_rate, audio_frame))
            else:
                _rec_nones += 1
                _consecutive_nones += 1
                if _rec_nones == 1:
                    logger.warning("DIAG record_loop: first None from get_audio_sample()")
                if now - _last_status_time >= 5.0:
                    logger.warning(
                        "DIAG record_loop: status frames=%d nones=%d consecutive_nones=%d",
                        _rec_frames, _rec_nones, _consecutive_nones,
                    )
                    _last_status_time = now
                await asyncio.sleep(0.01)  # back off on empty frames

    async def play_loop(self) -> None:
        """Fetch outputs from the handler: log text and play audio frames."""
        logger.info("DIAG play_loop: entered")
        _play_audio_count = 0
        _play_meta_count = 0
        _play_none_count = 0
        while not self._stop_event.is_set():
            handler_output = await self.handler.emit()

            if isinstance(handler_output, AdditionalOutputs):
                _play_meta_count += 1
                for msg in handler_output.args:
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        logger.info(
                            "role=%s content=%s",
                            msg.get("role"),
                            content if len(content) < 500 else content[:500] + "…",
                        )

            elif isinstance(handler_output, tuple):
                _play_audio_count += 1
                input_sample_rate, audio_data = handler_output
                output_sample_rate = self._robot.media.get_output_audio_samplerate()

                if _play_audio_count == 1:
                    logger.info(
                        "DIAG play_loop: first audio chunk sr=%d→%d shape=%s dtype=%s max=%.4f",
                        input_sample_rate, output_sample_rate,
                        audio_data.shape, audio_data.dtype, float(abs(audio_data).max()),
                    )

                # Reshape if needed
                if audio_data.ndim == 2:
                    # Scipy channels last convention
                    if audio_data.shape[1] > audio_data.shape[0]:
                        audio_data = audio_data.T
                    # Multiple channels -> Mono channel
                    if audio_data.shape[1] > 1:
                        audio_data = audio_data[:, 0]

                # Cast if needed
                audio_frame = audio_to_float32(audio_data)

                # Resample if needed
                if input_sample_rate != output_sample_rate:
                    audio_frame = resample(
                        audio_frame,
                        int(len(audio_frame) * output_sample_rate / input_sample_rate),
                    )

                logger.info(
                    "DIAG play_loop: pushing chunk #%d to speaker shape=%s dtype=%s max=%.4f",
                    _play_audio_count, audio_frame.shape, audio_frame.dtype, float(abs(audio_frame).max()),
                )

                self._robot.media.push_audio_sample(audio_frame)

            elif handler_output is None:
                _play_none_count += 1
                if _play_none_count == 1:
                    logger.debug("DIAG play_loop: first None from emit()")
            else:
                logger.debug("Ignoring output type=%s", type(handler_output).__name__)

            await asyncio.sleep(0)  # yield to event loop
