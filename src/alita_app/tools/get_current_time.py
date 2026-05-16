"""Tool: get_current_time — returns current date/time in Europe/Berlin."""

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict

from alita_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Europe/Berlin: UTC+1 (CET) / UTC+2 (CEST)
# Simple DST approximation — last Sunday of March to last Sunday of October
_CET = timezone(timedelta(hours=1))
_CEST = timezone(timedelta(hours=2))


def _last_sunday(year: int, month: int, day: int) -> datetime:
    """Find the last Sunday on or before the given date, at 01:00 UTC."""
    dt = datetime(year, month, day, 1, 0, tzinfo=timezone.utc)
    dt -= timedelta(days=(dt.weekday() + 1) % 7)  # weekday: Mon=0..Sun=6; +1%7 maps Sun→0
    return dt


def _berlin_now() -> datetime:
    """Return current datetime in Europe/Berlin (with basic DST)."""
    utc_now = datetime.now(timezone.utc)
    year = utc_now.year
    # DST: last Sunday of March 01:00 UTC → last Sunday of October 01:00 UTC
    dst_start = _last_sunday(year, 3, 31)
    dst_end = _last_sunday(year, 10, 31)

    if dst_start <= utc_now < dst_end:
        return utc_now.astimezone(_CEST)
    return utc_now.astimezone(_CET)


class GetCurrentTime(Tool):
    """Get the current date and time."""

    name = "get_current_time"
    description = "Get the current date and time in Europe/Berlin timezone."
    parameters_schema: Dict[str, Any] = {"type": "object", "properties": {}}

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        now = _berlin_now()
        return {
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "weekday": now.strftime("%A"),
            "timezone": "Europe/Berlin",
        }
