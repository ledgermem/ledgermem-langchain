"""Smoke test: import + instantiate with a mocked SDK client."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _install_fake_getmnemo() -> None:
    """Install a fake `getmnemo` module so the package imports without the real SDK."""
    if "getmnemo" in sys.modules:
        return
    fake = types.ModuleType("getmnemo")

    class Mnemo:  # noqa: D401
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

    class AsyncMnemo(Mnemo):
        pass

    fake.Mnemo = Mnemo
    fake.AsyncMnemo = AsyncMnemo
    sys.modules["getmnemo"] = fake


_install_fake_getmnemo()

from langchain_getmnemo import MnemoMemory, MnemoRetriever  # noqa: E402
from getmnemo import Mnemo  # noqa: E402


def test_imports_resolve() -> None:
    assert MnemoMemory is not None
    assert MnemoRetriever is not None


def test_memory_save_and_load() -> None:
    client = Mnemo(api_key="test", workspace_id="ws_test")
    client.add = MagicMock(return_value=None)
    memory = MnemoMemory(client=client)
    memory.save_context({"input": "hi"}, {"output": "hello"})
    assert client.add.call_count == 2
    out = memory.load_memory_variables({"input": "hi"})
    assert "history" in out


def test_retriever_returns_documents() -> None:
    client = Mnemo(api_key="test", workspace_id="ws_test")
    hit = type("Hit", (), {"content": "abc", "metadata": {"k": "v"}, "score": 0.9, "id": "m1"})()
    client.search = MagicMock(return_value=type("Resp", (), {"hits": [hit]})())
    retriever = MnemoRetriever(client=client, top_k=3)
    docs = retriever.invoke("query")
    assert len(docs) == 1
    assert docs[0].page_content == "abc"
    assert docs[0].metadata["memory_id"] == "m1"
