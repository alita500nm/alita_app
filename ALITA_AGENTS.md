# ALITA_AGENTS.md — project-specific overrides

**Read `AGENTS.md` (Pollen Robotics' official guide) first.** It's in
this same directory. This file only documents what's *different* or
*specific to this project*.

---

## Goal

A Reachy Mini app that embodies **Alita** — a Cyborg-Poetin character
(Battle Angel Alita / 銃夢). Turquoise frequency, terse and warm,
not an assistant. German primary language.

## Stack (decisions made)

- **Base**: conversation template (already generated)
- **LLM**: Anthropic Claude (`claude-sonnet-4-5-20250929` or newer)
  via Anthropic API. The template's default OpenAI Realtime API
  must be **replaced** with Anthropic. Env var: `ANTHROPIC_API_KEY`.
- **Local fallback LLM**: Ollama with `qwen2.5:14b` on
  `localhost:11434` (already running on host). Use for short
  command-style turns to keep latency low; route emotional/longer
  turns to Claude.
- **STT**: **ElevenLabs Scribe v2** (batch upload after silero-VAD
  end-of-speech detection). Env: `ELEVENLABS_API_KEY`,
  `ELEVENLABS_STT_MODEL=scribe_v2`. Language pinned to `de`.
- **TTS**: **ElevenLabs streaming**, `pcm_16000` output. Env:
  `ELEVENLABS_TTS_MODEL` (default `eleven_turbo_v2_5` for latency;
  `eleven_v3` for higher German expressiveness at ~3× first-audio
  latency) and `ELEVENLABS_TTS_VOICE_ID`. Voice picked from the
  shared library — candidate German voices: Clara (warm, calm, mid-40s)
  and Sina (warm, approachable, 30s).
- **Wake mechanism (no audio listening in HIBERNATE)**: external
  trigger via `POST /api/wake`, `POST /api/wake/conversation`,
  `POST /api/sleep` on the FastAPI app, or `SIGUSR1`/`SIGUSR2`
  signals as a headless fallback. Vosk wake-word detection was
  removed — too many false triggers from background TV/noise.
- **Local fallback audio (on disk only, NOT in code paths)**:
  - `~/Labor/scripts/piper/piper` (binary)
  - `~/Labor/scripts/piper/voices/de_DE-eva_k-x_low.onnx`
  - `vosk-model-small-de-0.15` (HF cache)
  Kept available for manual reactivation if ElevenLabs is down or
  offline operation is needed; the code no longer references them.
- **Memory**: SQLite (working schema in `~/alita_app_rescue/store.py` —
  re-use or adapt, don't reinvent)

## Movement constraints (important)

Reachy's default motions are enthusiastic. We want **calmer**:
- Reduce amplitudes ~10% from defaults
- No motions that hit the casing
- Avoid continuous small oscillations (no "fidgeting")
  unless explicitly an idle/breathing animation
- Prefer `goto_target` with `duration` over fast `set_target`

For LLM tool exposure: don't expose the full dance library or all
emotions. Curate a small set of gentle, intentional motions
(nod, tilt_head, look_at_direction, antennas_curious, antennas_calm).
Full dance/emotion catalogs are too big and lead to noisy behavior.

## Persona — voice

System prompt baseline is in `~/alita_app_rescue/qwen.txt`. Reuse it,
port to whichever prompt-file location the template uses
(likely `src/alita_app/prompts/` or `prompts_library/`).

**Critical persona rules**:
- Never "How can I help you?" or assistant-tropes
- Never "As an AI..." / "As a language model..."
- No heart-emoji-per-line spam
- This Alita is **standalone** — no references to "Bill", "Bär",
  "Damaszener Klingen", or any external project lore. She is her
  own embodiment, not a clone of a chat persona.

## What NOT to do

- Don't create a new venv (use `~/dev/alita_app/.venv/`)
- Don't write a `requirements.txt` (we use `pyproject.toml`)
- Don't add OpenAI, OpenAI Realtime, or any non-Anthropic LLM API
- Don't add Google APIs (Gemini/Speech-to-Text/etc.)
- Don't reintroduce Vosk wake-word listening — manual wake only
- Don't reintroduce dance/emotion mass-exposure to the LLM after
  curating the tool set
- Don't commit `data/memory.db` or any `.env` files
- Don't change `pyproject.toml` entry points without checking with me

## Files to preserve / port from previous iteration

In `~/alita_app_rescue/`:
- `qwen.py` — Ollama wrapper, tested and working
- `qwen.txt` — system prompt baseline
- `store.py` — SQLite schema (conversations / state / events)

These don't have to be copied verbatim — adapt to the
conversation-template's structure. But the *design* of each
(simple wrapper, three-table schema, persistent history across
sessions) is what we want.

## Status

- 14. Mai 2026: conversation template freshly generated.
  Replacement of OpenAI → Anthropic done; integrated Piper TTS and
  the qwen fallback route on top.
- 19. Mai 2026: STT/TTS migrated to ElevenLabs (Scribe v2 + streaming
  TTS). Vosk wake-word removed; wake is now manual via HTTP or
  SIGUSR1/2. Piper binary + voice file remain on disk as a manual
  fallback only.

---

When in doubt, ask Joschka before refactoring.
