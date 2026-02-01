## Setup

### Venv using uv

```shell
uv venv -p 3.11 .venv && source .venv/bin/activate 
uv pip install --upgrade pip
uv pip install -r requirements.txt
```

## Local LLM setup using ollama
```shell
brew install ollama
OLLAMA_NO_METAL=1 ollama serve
ollama pull phi3:mini
```


┌──────────────┐
│ Tesla PDF(s) │
└──────┬───────┘
       │
       ▼
┌────────────────────┐
│ Azure Blob Storage │
│  (raw documents)   │
└──────┬─────────────┘
       │
       ▼
┌────────────────────┐
│ Document Extractor │
│ (PDF → text + meta)│
└──────┬─────────────┘
       │
       ▼
┌────────────────────┐
│ Chunking Pipeline  │
│ (semantic chunks)  │
└──────┬─────────────┘
       │
       ▼
┌────────────────────┐
│ Embedding Service  │
│ (OpenAI / Azure)   │
└──────┬─────────────┘
       │
       ▼
┌────────────────────┐
│ Azure AI Search    │
│ Vector Index       │
└────────────────────┘

