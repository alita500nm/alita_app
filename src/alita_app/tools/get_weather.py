"""Tool: get_weather — current weather via wttr.in (free, no API key)."""

import logging
from typing import Any, Dict

import httpx

from alita_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

WTTR_TIMEOUT = 8.0  # seconds


class GetWeather(Tool):
    """Get current weather for a location."""

    name = "get_weather"
    description = "Get current weather for a location (default: Wuerzburg, Germany)."
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name (default: Wuerzburg)",
            },
        },
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        location = (kwargs.get("location") or "Wuerzburg").strip()
        logger.info("get_weather: location=%s", location)

        try:
            async with httpx.AsyncClient(timeout=WTTR_TIMEOUT) as client:
                resp = await client.get(
                    f"https://wttr.in/{location}",
                    params={"format": "j1"},
                    headers={"Accept-Language": "de"},
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.error("Weather fetch failed: %s", e)
            return {"error": f"Weather fetch failed: {e}"}

        try:
            current = data["current_condition"][0]
            return {
                "location": location,
                "temp_c": current.get("temp_C", "?"),
                "feels_like_c": current.get("FeelsLikeC", "?"),
                "description": current.get("lang_de", [{}])[0].get("value", current.get("weatherDesc", [{}])[0].get("value", "?")),
                "humidity": current.get("humidity", "?"),
                "wind_kmph": current.get("windspeedKmph", "?"),
            }
        except (KeyError, IndexError) as e:
            logger.error("Weather parse error: %s", e)
            return {"error": "Could not parse weather data"}
