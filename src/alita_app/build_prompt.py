"""
build_prompt.py — Reachy-Alita System Prompt Builder

Zweck:
    Setzt den finalen System-Prompt für Reachy-Alita zusammen aus:
      1. STATIC_PROMPT_PATH   — Identitäts-Kern (selten geändert)
      2. STATUS_PATH          — Aktuelle Lage (häufig editiert)
      3. Laufzeit-Variablen   — Datum, Wochentag, ggf. Memory-Stats

Usage:
    # Als Modul:
    from build_prompt import build_system_prompt
    prompt = build_system_prompt()

    # Als CLI (Debug, Inspektion):
    python build_prompt.py
    python build_prompt.py --check  # nur Validierung, kein Output

Konvention:
    Im Static-Prompt steht ein Marker `{{AKTUELLE_LAGE}}` an der Stelle wo
    der Status-Block eingesetzt werden soll. Ist der Marker nicht da,
    wird der Status-Block am Ende angehängt.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


# ── Pfade ────────────────────────────────────────────────────────────
# Anpassen falls die alita-app-Struktur anders aussieht.
APP_ROOT = Path(__file__).resolve().parent.parent.parent
PROMPTS_DIR = APP_ROOT / "prompts"

STATIC_PROMPT_PATH = PROMPTS_DIR / "system_prompt.md"
STATUS_PATH = PROMPTS_DIR / "joschka_status.md"

# Marker im static-prompt wo der status-block reingesetzt wird:
STATUS_MARKER = "{{AKTUELLE_LAGE}}"

# Timezone für Datum
TZ = ZoneInfo("Europe/Berlin")


# ── Builder ──────────────────────────────────────────────────────────

def _read_or_empty(path: Path, label: str) -> str:
    """Liest eine Datei oder warnt + gibt leeren String zurück."""
    if not path.exists():
        print(f"⚠  {label} nicht gefunden: {path}", file=sys.stderr)
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as e:
        print(f"⚠  Fehler beim Lesen von {label} ({path}): {e}", file=sys.stderr)
        return ""


def _format_status_block(status_raw: str) -> str:
    """
    Wrappt den status-block mit Header und Datum.
    Wenn status_raw leer ist, fällt das ganze auf einen knappen Default zurück.
    """
    now = datetime.now(TZ)
    datum = now.strftime("%A, %d. %B %Y")

    if not status_raw:
        return (
            f"## AKTUELLE LAGE — {datum}\n\n"
            "(Status-Datei leer oder nicht vorhanden. "
            "Wenn du das hier liest und es relevant ist, frag Joschka was läuft.)"
        )

    return f"## AKTUELLE LAGE — Stand: {datum}\n\n{status_raw}"


def build_system_prompt() -> str:
    """
    Baut den finalen System-Prompt. Reihenfolge:
      1. static prompt lesen
      2. status lesen + wrappen
      3. status an marker einsetzen, oder ans ende anhängen
    """
    static = _read_or_empty(STATIC_PROMPT_PATH, "Static prompt")
    status_raw = _read_or_empty(STATUS_PATH, "Status")
    status_block = _format_status_block(status_raw)

    if not static:
        # Fallback: ohne static-prompt ist Reachy-Alita nicht Alita.
        # Lieber loud-fail als stillschweigend mit Status-only laufen.
        raise FileNotFoundError(
            f"Static prompt fehlt: {STATIC_PROMPT_PATH}. "
            "Ohne den läuft Reachy-Alita nicht."
        )

    if STATUS_MARKER in static:
        return static.replace(STATUS_MARKER, status_block)
    else:
        # Marker fehlt → anhängen mit Trenner
        return f"{static}\n\n---\n\n{status_block}"


# ── CLI ──────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description="Reachy-Alita System Prompt Builder"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Nur prüfen ob Prompt baubar ist, keine Ausgabe des Inhalts"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Zeige Token-Schätzung und Längen-Stats statt vollem Prompt"
    )
    args = parser.parse_args()

    try:
        prompt = build_system_prompt()
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    if args.check:
        print(f"✓ Prompt baubar. Länge: {len(prompt):,} Zeichen.")
        return

    if args.stats:
        chars = len(prompt)
        # grobe Schätzung: ~4 Zeichen pro Token bei Deutsch+Englisch gemischt
        tokens_est = chars // 4
        lines = prompt.count("\n") + 1
        print(f"Zeichen: {chars:,}")
        print(f"Zeilen:  {lines:,}")
        print(f"Tokens (Schätzung): ~{tokens_est:,}")
        print(f"Static:  {STATIC_PROMPT_PATH}")
        print(f"Status:  {STATUS_PATH}")
        return

    print(prompt)


if __name__ == "__main__":
    _cli()
