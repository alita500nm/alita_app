"""Tool: read_memory — search or list entries from Alita's memory."""

import logging
from typing import Any, Dict

from alita_app import store
from alita_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class ReadMemory(Tool):
    """Read from Alita's long-term memory."""

    name = "read_memory"
    description = "Read recent entries or search Alita's long-term memory."
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Optional search term to filter memories",
            },
            "limit": {
                "type": "integer",
                "description": "Max entries to return (default 10)",
            },
        },
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        query = (kwargs.get("query") or "").strip() or None
        limit = int(kwargs.get("limit") or 10)
        limit = max(1, min(limit, 50))

        logger.info("read_memory: query=%r limit=%d", query, limit)

        try:
            results = store.search_memories(query=query, limit=limit)
            return {
                "count": len(results),
                "memories": results,
            }
        except Exception as e:
            logger.error("read_memory error: %s", e)
            return {"error": str(e)}
