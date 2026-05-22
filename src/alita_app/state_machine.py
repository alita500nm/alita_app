"""Alita state machine — HIBERNATE / LISTENING / CONVERSATION."""

import re
import time
import random
import logging
from enum import Enum
from typing import Optional


logger = logging.getLogger(__name__)


class AlitaState(Enum):
    HIBERNATE = "hibernate"
    LISTENING = "listening"
    CONVERSATION = "conversation"


# Wake-word names: "Alita" + common STT misspellings + "Robot" as fallback
_NAMES = r"(?:alita|alida|liter|dieter|ali\s*ta[g]?|a\s*liter|alliierter|anbieter|robot)"

# Wake-word patterns (checked in order: most specific first)
_WAKE_RULES: list[tuple[str, list[str]]] = [
    ("conversation", [rf"\b{_NAMES} bleib", rf"\b{_NAMES} wach bleiben"]),
    ("sleep", [rf"\b{_NAMES} schlaf", rf"\b{_NAMES} pause", rf"\b{_NAMES} aus"]),
    ("listen", [rf"\b{_NAMES}\b", rf"\bhey {_NAMES}\b", rf"\b{_NAMES} wach\b"]),
]

# Confirmation phrases when waking up
_WAKE_CONFIRMATIONS: list[str] = [
    "Ja?",
    "Ja mein Mensch?",
    "Hm?",
    "Was gibt's?",
]

CONVERSATION_TIMEOUT_S: float = 60.0


def detect_wake_command(text: str) -> Optional[str]:
    """Check text for wake/sleep commands. Returns 'listen', 'conversation', 'sleep', or None."""
    text = text.lower().strip()
    for category, patterns in _WAKE_RULES:
        for pattern in patterns:
            if re.search(pattern, text):
                return category
    return None


def random_wake_confirmation() -> str:
    """Return a random short confirmation phrase."""
    return random.choice(_WAKE_CONFIRMATIONS)


class StateMachine:
    """Tracks Alita's operational state."""

    def __init__(self) -> None:
        self.state = AlitaState.CONVERSATION
        self.last_activity_time = time.monotonic()

    def transition(self, new_state: AlitaState) -> None:
        old = self.state
        if old == new_state:
            return
        self.state = new_state
        self.last_activity_time = time.monotonic()
        logger.info("State: %s -> %s", old.value, new_state.value)

    def touch(self) -> None:
        """Reset the activity timer (call on each speech/interaction)."""
        self.last_activity_time = time.monotonic()

    def check_conversation_timeout(self) -> bool:
        """If in CONVERSATION and silent too long, transition to HIBERNATE. Returns True if timed out."""
        if self.state != AlitaState.CONVERSATION:
            return False
        if time.monotonic() - self.last_activity_time > CONVERSATION_TIMEOUT_S:
            self.transition(AlitaState.HIBERNATE)
            return True
        return False

    @property
    def is_active(self) -> bool:
        return self.state in (AlitaState.LISTENING, AlitaState.CONVERSATION)
