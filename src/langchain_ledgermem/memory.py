"""LedgerMem-backed conversational memory for LangChain."""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.memory import BaseMemory
from ledgermem import LedgerMem
from pydantic import Field, PrivateAttr


class LedgerMemMemory(BaseMemory):
    """Conversational memory backed by LedgerMem.

    On every turn, ``save_context`` writes the user/AI exchange into LedgerMem,
    and ``load_memory_variables`` retrieves the top-K relevant prior turns by
    semantic search.
    """

    memory_key: str = "history"
    input_key: str = "input"
    output_key: str = "output"
    top_k: int = Field(default=5, ge=1, le=50)
    namespace: Optional[str] = None

    _client: LedgerMem = PrivateAttr()

    def __init__(self, client: LedgerMem, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = client

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def _format_metadata(self, role: str) -> dict[str, Any]:
        meta: dict[str, Any] = {"role": role, "source": "langchain"}
        if self.namespace:
            meta["namespace"] = self.namespace
        return meta

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        query = inputs.get(self.input_key, "")
        if not query:
            return {self.memory_key: ""}
        response = self._client.search(query, limit=self.top_k)
        hits = getattr(response, "hits", []) or []
        lines = []
        for hit in hits:
            content = getattr(hit, "content", None) or getattr(hit, "text", "")
            metadata = getattr(hit, "metadata", {}) or {}
            role = metadata.get("role", "memory")
            lines.append(f"{role}: {content}")
        return {self.memory_key: "\n".join(lines)}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        user_text = inputs.get(self.input_key)
        ai_text = outputs.get(self.output_key)
        if user_text:
            self._client.add(str(user_text), metadata=self._format_metadata("user"))
        if ai_text:
            self._client.add(str(ai_text), metadata=self._format_metadata("assistant"))

    def clear(self) -> None:
        # LedgerMem does not expose a workspace-wide wipe in the SDK; iterate and delete.
        cursor: Optional[str] = None
        while True:
            page = self._client.list(limit=100, cursor=cursor)
            items = getattr(page, "items", []) or getattr(page, "memories", []) or []
            for item in items:
                memory_id = getattr(item, "id", None)
                if memory_id is not None:
                    self._client.delete(memory_id)
            cursor = getattr(page, "next_cursor", None)
            if not cursor:
                break
