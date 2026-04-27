"""Smoke test: import + instantiate with a mocked SDK client."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _install_fake_ledgermem() -> None:
    """Install a fake `ledgermem` module so the package imports without the real SDK."""
    if "ledgermem" in sys.modules:
        return
    fake = types.ModuleType("ledgermem")

    class LedgerMem:  # noqa: D401
        def __init__(self, *args, **kwargs):
            self.calls = []

        def search(self, query, limit=5):
            return types.SimpleNamespace(hits=[])

        def add(self, content, metadata=None):
            return types.SimpleNamespace(id="mem_test")

        def delete(self, memory_id):
            return None

        def list(self, limit=20, cursor=None):
            return types.SimpleNamespace(items=[], next_cursor=None)

    class AsyncLedgerMem(LedgerMem):
        pass

    fake.LedgerMem = LedgerMem
    fake.AsyncLedgerMem = AsyncLedgerMem
    sys.modules["ledgermem"] = fake


_install_fake_ledgermem()

from langchain_ledgermem import LedgerMemMemory, LedgerMemRetriever  # noqa: E402
from ledgermem import LedgerMem  # noqa: E402


def test_imports_resolve() -> None:
    assert LedgerMemMemory is not None
    assert LedgerMemRetriever is not None


def test_memory_save_and_load() -> None:
    client = LedgerMem(api_key="test", workspace_id="ws_test")
    client.add = MagicMock(return_value=None)
    memory = LedgerMemMemory(client=client)
    memory.save_context({"input": "hi"}, {"output": "hello"})
    assert client.add.call_count == 2
    out = memory.load_memory_variables({"input": "hi"})
    assert "history" in out


def test_retriever_returns_documents() -> None:
    client = LedgerMem(api_key="test", workspace_id="ws_test")
    hit = type("Hit", (), {"content": "abc", "metadata": {"k": "v"}, "score": 0.9, "id": "m1"})()
    client.search = MagicMock(return_value=type("Resp", (), {"hits": [hit]})())
    retriever = LedgerMemRetriever(client=client, top_k=3)
    docs = retriever.invoke("query")
    assert len(docs) == 1
    assert docs[0].page_content == "abc"
    assert docs[0].metadata["memory_id"] == "m1"
