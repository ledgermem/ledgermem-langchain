# ledgermem-langchain

LangChain adapter for [LedgerMem](https://github.com/ledgermem/ledgermem-python) — drop-in conversational memory and retriever backed by the LedgerMem SDK.

## Install

```bash
pip install ledgermem-langchain
```

## Quickstart

```python
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from ledgermem import LedgerMem
from langchain_ledgermem import LedgerMemMemory

mem_client = LedgerMem(api_key="lm_...", workspace_id="ws_...")
chain = ConversationChain(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    memory=LedgerMemMemory(client=mem_client, top_k=5),
)

print(chain.invoke({"input": "Remember that my favourite framework is FastAPI."}))
print(chain.invoke({"input": "What is my favourite framework?"}))
```

## Retriever for RAG

```python
from langchain_ledgermem import LedgerMemRetriever

retriever = LedgerMemRetriever(client=mem_client, top_k=8)
docs = retriever.invoke("show me what I said about deployments")
for doc in docs:
    print(doc.metadata.get("score"), doc.page_content)
```

## What you get

- `LedgerMemMemory` — implements `langchain_core.memory.BaseMemory`. Auto-saves every user / AI turn and rehydrates the top-K relevant prior turns by semantic search.
- `LedgerMemRetriever` — implements `langchain_core.retrievers.BaseRetriever`. Returns LangChain `Document` objects with score and `memory_id` in metadata.

## License

MIT — see [LICENSE](./LICENSE).
