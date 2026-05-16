"""Tool: write_memory — save an important fact or moment to Alita's memory."""

import logging
from typing import Any, Dict, List

from alita_app import store
from alita_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class WriteMemory(Tool):
    """Save something to Alita's long-term memory."""

    name = "write_memory"
    description = "Save an important fact, observation, or moment to Alita's long-term memory."
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "What to remember",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for categorisation",
            },
        },
        "required": ["content"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        content = (kwargs.get("content") or "").strip()
        if not content:
            return {"error": "content must be non-empty"}

        tags: List[str] = kwargs.get("tags") or []
        logger.info("write_memory: content=%r tags=%s", content[:80], tags)

        try:
            row_id = store.save_memory(content, tags=tags or None)
            return {"saved": True, "id": row_id}
        except Exception as e:
            logger.error("write_memory error: %s", e)
            return {"error": str(e)}
