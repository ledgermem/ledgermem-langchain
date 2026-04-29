"""Mnemo-backed retriever for LangChain RAG pipelines."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from getmnemo import Mnemo
from pydantic import Field, PrivateAttr


class MnemoRetriever(BaseRetriever):
    """LangChain ``BaseRetriever`` that delegates to ``Mnemo.search``."""

    top_k: int = Field(default=5, ge=1, le=100)

    _client: Mnemo = PrivateAttr()

    def __init__(self, client: Mnemo, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = client

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        response = self._client.search(query, limit=self.top_k)
        hits = getattr(response, "hits", []) or []
        documents: list[Document] = []
        for hit in hits:
            content = getattr(hit, "content", None) or getattr(hit, "text", "")
            metadata = dict(getattr(hit, "metadata", {}) or {})
            score = getattr(hit, "score", None)
            if score is not None:
                metadata["score"] = score
            memory_id = getattr(hit, "id", None)
            if memory_id is not None:
                metadata["memory_id"] = memory_id
            documents.append(Document(page_content=content, metadata=metadata))
        return documents
