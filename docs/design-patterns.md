# Complete Design Patterns for AI/ML Pipelines

A comprehensive guide to writing clean, maintainable, and production-ready code for RAG and AI systems.

---

## Table of Contents

1. [Single Responsibility Principle (SRP)](#1-single-responsibility-principle-srp)
2. [Data Classes as Contracts](#2-data-classes-as-contracts)
3. [Pipeline Pattern](#3-pipeline-pattern-functional-composition)
4. [Explicit over Implicit](#4-explicit-over-implicit)
5. [Dependency Injection](#5-dependency-injection)
6. [Strategy Pattern](#6-strategy-pattern)
7. [Factory Pattern](#7-factory-pattern)
8. [Repository Pattern](#8-repository-pattern)
9. [Configuration Management](#9-configuration-management)
10. [Graceful Degradation & Defaults](#10-graceful-degradation--defaults)
11. [Layered Architecture](#11-layered-architecture)
12. [Inspection Points (Observability)](#12-inspection-points-observability)
13. [Idempotency](#13-idempotency)
14. [Fail Fast](#14-fail-fast)
15. [Error Handling & Retry Pattern](#15-error-handling--retry-pattern)
16. [Logging & Observability](#16-logging--observability)

---

## 1. Single Responsibility Principle (SRP)

**The Rule:** A module should have only one reason to change.

### Bad Example

```python
# ❌ One module doing everything
def process_pdf(path):
    # Extract (reason to change: PDF library updates)
    text = extract_pdf(path)
    
    # Chunk (reason to change: chunking strategy)
    chunks = split_text(text)
    
    # Embed (reason to change: embedding model)
    vectors = call_openai(chunks)
    
    # Store (reason to change: database choice)
    save_to_pinecone(vectors)
```

### Good Example

```python
# ✅ Separate modules
# filepath: ingest/extract.py
def extract_text_from_pdf(path: Path) -> ExtractedDocument:
    """Only extracts. Can swap PyMuPDF → Azure Doc Intelligence."""
    ...

# filepath: ingest/embed.py  
def embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    """Only embeds. Can swap OpenAI → Cohere → local model."""
    ...
```

### Application

```
extract.py      → Only extracts text from PDFs
chunk.py        → Only splits text into chunks
embed.py        → Only generates embeddings
blob_storage.py → Only handles Azure Blob operations
```

### Why It Matters for AI

- Embedding models change frequently (OpenAI → Cohere → local)
- You'll experiment with different chunking strategies
- Vector stores evolve (Pinecone → Qdrant → Azure AI Search)

---

## 2. Data Classes as Contracts

**The Rule:** Define explicit data structures that flow between components.

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass(frozen=True)  # frozen=True makes it immutable
class ExtractedDocument:
    """Output of extraction phase."""
    source: str
    text: str
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validation runs after creation."""
        if not self.text.strip():
            raise ValueError(f"Empty document: {self.source}")


@dataclass
class Chunk:
    """Output of chunking phase."""
    id: str
    text: str
    source: str
    chunk_index: int
    token_count: int = 0
    metadata: dict = field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        """Business logic can live on data classes."""
        return len(self.text) >= 50  # Minimum viable chunk


@dataclass
class EmbeddedChunk:
    """Output of embedding phase."""
    chunk: Chunk
    embedding: np.ndarray
    model: str  # Track which model created this
    
    @property
    def dimension(self) -> int:
        return len(self.embedding)
```

### Benefits

| Feature | Benefit |
|---------|---------|
| Type hints | IDE autocompletion, catch errors early |
| `frozen=True` | Prevents accidental mutation |
| `__post_init__` | Validate data at creation |
| Properties | Computed fields without storage |
| Default factories | Safe mutable defaults |

---

## 3. Pipeline Pattern (Functional Composition)

**The Rule:** Data flows through a series of pure transformations.

```python
from typing import Callable, TypeVar
from functools import reduce

T = TypeVar('T')


class Pipeline:
    """
    Composable pipeline builder.
    
    Usage:
        result = (Pipeline(pdf_path)
            .then(extract_text_from_pdf)
            .then(lambda doc: normalize_text(doc.text))
            .then(lambda text: chunk_text(text, source=pdf_path.name))
            .then(embed_chunks)
            .run())
    """
    
    def __init__(self, initial_value):
        self._value = initial_value
        self._steps: list[Callable] = []
    
    def then(self, func: Callable) -> 'Pipeline':
        """Add a transformation step."""
        self._steps.append(func)
        return self
    
    def run(self):
        """Execute all steps in sequence."""
        return reduce(lambda val, func: func(val), self._steps, self._value)


# Alternative: Simple function composition
def compose(*functions):
    """Compose functions right-to-left: compose(f, g, h)(x) = f(g(h(x)))"""
    def inner(arg):
        for f in reversed(functions):
            arg = f(arg)
        return arg
    return inner
```

### Visual Representation

```
┌─────────┐    ┌───────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  Path   │ →  │ Extracted │ →  │ String  │ →  │ Chunks  │ →  │Embedded │
│         │    │ Document  │    │  (norm) │    │         │    │ Chunks  │
└─────────┘    └───────────┘    └─────────┘    └─────────┘    └─────────┘
   extract()     .text +         chunk()        embed()
                normalize()
```

### Pipeline Characteristics

- Takes input, returns output
- No side effects (except explicit I/O)
- Can be tested in isolation

---

## 4. Explicit over Implicit

**The Rule:** All dependencies should be passed as arguments.

```python
# ❌ Bad: Hidden dependencies
def process():
    text = get_global_text()  # Where does this come from?
    
# ✅ Good: Explicit inputs
def process(text: str) -> list[Chunk]:
    ...
```

---

## 5. Dependency Injection

**The Rule:** Pass dependencies in, don't create them inside.

```python
# ❌ Bad: Hard-coded dependency
def embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Hard to test!
    ...

# ✅ Good: Inject the dependency
from typing import Protocol


class Embedder(Protocol):
    """Interface for embedding providers."""
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class OpenAIEmbedder:
    def __init__(self, client, model: str = "text-embedding-3-small"):
        self.client = client
        self.model = model
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [e.embedding for e in response.data]


class LocalEmbedder:
    """For testing or offline use."""
    def __init__(self, model_path: str):
        self.model = load_local_model(model_path)
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.model.encode(t) for t in texts]


# Now you can swap implementations
def embed_chunks(chunks: list[Chunk], embedder: Embedder) -> list[EmbeddedChunk]:
    texts = [c.text for c in chunks]
    embeddings = embedder.embed(texts)
    return [
        EmbeddedChunk(chunk=c, embedding=np.array(e), model=embedder.model)
        for c, e in zip(chunks, embeddings)
    ]
```

### Why It Matters for AI

- Swap OpenAI ↔ Azure OpenAI ↔ local models
- Use mock embedders in tests (fast, free)
- A/B test different models

---

## 6. Strategy Pattern

**The Rule:** Encapsulate algorithms and make them interchangeable.

```python
from typing import Protocol


class ChunkingStrategy(Protocol):
    """Interface for chunking algorithms."""
    def chunk(self, text: str, source: str) -> list[Chunk]:
        ...


class FixedSizeChunker:
    """Simple fixed-size chunks with overlap."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, source: str) -> list[Chunk]:
        chunks = []
        start = 0
        idx = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            chunks.append(Chunk(
                id=f"{source}_{idx}",
                text=chunk_text,
                source=source,
                chunk_index=idx
            ))
            
            start = end - self.overlap
            idx += 1
        
        return chunks


class SemanticChunker:
    """Chunk by semantic boundaries (paragraphs, sections)."""
    
    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size
    
    def chunk(self, text: str, source: str) -> list[Chunk]:
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        idx = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(Chunk(
                        id=f"{source}_{idx}",
                        text=current_chunk.strip(),
                        source=source,
                        chunk_index=idx
                    ))
                    idx += 1
                current_chunk = para
            else:
                current_chunk += "\n\n" + para
        
        if current_chunk:
            chunks.append(Chunk(
                id=f"{source}_{idx}",
                text=current_chunk.strip(),
                source=source,
                chunk_index=idx
            ))
        
        return chunks


class SentenceWindowChunker:
    """For sentence-window retrieval (advanced RAG)."""
    
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
    
    def chunk(self, text: str, source: str) -> list[Chunk]:
        import nltk
        sentences = nltk.sent_tokenize(text)
        
        chunks = []
        for i, sentence in enumerate(sentences):
            window_start = max(0, i - self.window_size)
            window_end = min(len(sentences), i + self.window_size + 1)
            
            chunks.append(Chunk(
                id=f"{source}_{i}",
                text=sentence,
                source=source,
                chunk_index=i,
                metadata={
                    "window": " ".join(sentences[window_start:window_end])
                }
            ))
        
        return chunks


# Usage: Select strategy at runtime
def chunk_text(text: str, source: str, strategy: ChunkingStrategy = None) -> list[Chunk]:
    if strategy is None:
        strategy = SemanticChunker()
    return strategy.chunk(text, source)
```

---

## 7. Factory Pattern

**The Rule:** Centralize object creation logic.

```python
from config.settings import load_settings


class EmbedderFactory:
    """Create embedders based on configuration."""
    
    @staticmethod
    def create(provider: str = None) -> Embedder:
        settings = load_settings()
        provider = provider or settings.get("embedding_provider", "openai")
        
        if provider == "openai":
            from openai import OpenAI
            client = OpenAI()
            return OpenAIEmbedder(client, model=settings.get("embedding_model"))
        
        elif provider == "azure":
            from openai import AzureOpenAI
            client = AzureOpenAI(
                azure_endpoint=settings["azure_endpoint"],
                api_version=settings["api_version"]
            )
            return OpenAIEmbedder(client, model=settings.get("embedding_model"))
        
        elif provider == "local":
            return LocalEmbedder(model_path=settings["local_model_path"])
        
        else:
            raise ValueError(f"Unknown provider: {provider}")


class ChunkerFactory:
    """Create chunkers based on configuration."""
    
    @staticmethod
    def create(strategy: str = None) -> ChunkingStrategy:
        settings = load_settings()
        strategy = strategy or settings.get("chunking_strategy", "semantic")
        
        if strategy == "fixed":
            return FixedSizeChunker(
                chunk_size=settings.get("chunk_size", 512),
                overlap=settings.get("chunk_overlap", 50)
            )
        elif strategy == "semantic":
            return SemanticChunker(
                max_chunk_size=settings.get("max_chunk_size", 1000)
            )
        elif strategy == "sentence_window":
            return SentenceWindowChunker(
                window_size=settings.get("window_size", 3)
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
```

---

## 8. Repository Pattern

**The Rule:** Abstract data access behind a clean interface.

```python
from abc import ABC, abstractmethod
from typing import Protocol


class VectorRepository(Protocol):
    """Interface for vector storage backends."""
    
    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        """Insert or update vectors."""
        ...
    
    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
        """Find similar vectors."""
        ...
    
    def delete(self, ids: list[str]) -> None:
        """Remove vectors by ID."""
        ...


@dataclass
class SearchResult:
    chunk_id: str
    score: float
    text: str
    metadata: dict


class AzureAISearchRepository:
    """Azure AI Search implementation."""
    
    def __init__(self, endpoint: str, index_name: str, credential):
        from azure.search.documents import SearchClient
        self.client = SearchClient(endpoint, index_name, credential)
    
    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        documents = [
            {
                "id": chunk.chunk.id,
                "content": chunk.chunk.text,
                "embedding": chunk.embedding.tolist(),
                "source": chunk.chunk.source,
                "metadata": chunk.chunk.metadata
            }
            for chunk in chunks
        ]
        self.client.upload_documents(documents)
    
    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
        from azure.search.documents.models import VectorizedQuery
        
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="embedding"
        )
        
        results = self.client.search(vector_queries=[vector_query])
        
        return [
            SearchResult(
                chunk_id=r["id"],
                score=r["@search.score"],
                text=r["content"],
                metadata=r.get("metadata", {})
            )
            for r in results
        ]


class InMemoryRepository:
    """For testing and development."""
    
    def __init__(self):
        self._store: dict[str, EmbeddedChunk] = {}
    
    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        for chunk in chunks:
            self._store[chunk.chunk.id] = chunk
    
    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
        import numpy as np
        
        scores = []
        for chunk_id, embedded in self._store.items():
            # Cosine similarity
            score = np.dot(query_vector, embedded.embedding) / (
                np.linalg.norm(query_vector) * np.linalg.norm(embedded.embedding)
            )
            scores.append((chunk_id, score, embedded))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return [
            SearchResult(
                chunk_id=chunk_id,
                score=score,
                text=embedded.chunk.text,
                metadata=embedded.chunk.metadata
            )
            for chunk_id, score, embedded in scores[:top_k]
        ]
```

---

## 9. Configuration Management

**The Rule:** Externalize all tunables.

### Configuration File (settings.yaml)

```yaml
# Chunking
chunking_strategy: semantic  # fixed | semantic | sentence_window
chunk_size: 512
chunk_overlap: 50
max_chunk_size: 1000

# Embedding
embedding_provider: azure  # openai | azure | local
embedding_model: text-embedding-3-small
embedding_batch_size: 100

# Vector Store
vector_store: azure_search  # azure_search | pinecone | qdrant
azure_search_index: tesla-docs

# Retrieval
top_k: 5
score_threshold: 0.7

# Generation
llm_provider: azure
llm_model: gpt-4o
temperature: 0.1
max_tokens: 1000
```

### Configuration Loader

```python
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

load_dotenv()


def load_settings() -> dict:
    """Load settings with environment variable overrides."""
    config_path = Path(__file__).parent / "settings.yaml"
    
    with open(config_path) as f:
        settings = yaml.safe_load(f)
    
    # Environment variables override YAML
    env_overrides = {
        "CHUNK_SIZE": ("chunk_size", int),
        "EMBEDDING_MODEL": ("embedding_model", str),
        "TOP_K": ("top_k", int),
        "LLM_MODEL": ("llm_model", str),
    }
    
    for env_var, (key, type_func) in env_overrides.items():
        if os.getenv(env_var):
            settings[key] = type_func(os.getenv(env_var))
    
    return settings


# Singleton pattern for expensive config loading
_settings = None

def get_settings() -> dict:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings
```

---

## 10. Graceful Degradation & Defaults

**The Rule:** Provide sensible defaults and safe fallbacks.

```python
def upload_pdf(
    file_path: Path,
    container_name: str = "documents",  # Sensible default
    overwrite: bool = False             # Safe default
) -> BlobInfo:
    ...
```

---

## 11. Layered Architecture

**The Rule:** Higher layers depend on lower layers, never the reverse.

```
┌─────────────────────────────────┐
│  scripts/run_ingest.py          │  ← Orchestration (high-level)
├─────────────────────────────────┤
│  ingest/extract.py              │
│  ingest/chunk.py                │  ← Business Logic (mid-level)
│  ingest/embed.py                │
├─────────────────────────────────┤
│  ingest/blob_storage.py         │  ← Infrastructure (low-level)
│  index/search_index.py          │
└─────────────────────────────────┘
```

---

## 12. Inspection Points (Observability)

**The Rule:** Save intermediate results for debugging.

```python
# Save intermediate results for debugging
if save_chunks:
    save_chunks_to_json(chunks, output_path)
```

### Why?

When something breaks, you can inspect each stage.

---

## 13. Idempotency

**The Rule:** Running the same operation twice produces the same result.

```python
# Running twice produces same result
def upload_pdf(..., overwrite: bool = False):
    ...

# Container is created only if missing
if not container_client.exists():
    container_client.create_container()
```

---

## 14. Fail Fast

**The Rule:** Fail immediately with clear error messages.

```python
if len(sys.argv) < 2:
    print("Usage:")
    print("  python -m scripts.run_ingest local <pdf_directory>")
    print("  python -m scripts.run_ingest blob [container_name]")
    sys.exit(1)
```

---

## 15. Error Handling & Retry Pattern

**The Rule:** Handle failures gracefully, especially for external APIs.

```python
import time
import logging
from functools import wraps
from typing import Callable, Type

logger = logging.getLogger(__name__)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,)
):
    """
    Retry decorator with exponential backoff.
    
    Usage:
        @retry(max_attempts=3, exceptions=(RateLimitError, TimeoutError))
        def call_api():
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}"
                    )
                    
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            raise last_exception
        
        return wrapper
    return decorator


# Usage in embedding
from openai import RateLimitError, APITimeoutError


@retry(max_attempts=3, delay=1.0, exceptions=(RateLimitError, APITimeoutError))
def embed_batch(texts: list[str], client, model: str) -> list[list[float]]:
    response = client.embeddings.create(input=texts, model=model)
    return [e.embedding for e in response.data]
```

---

## 16. Logging & Observability

**The Rule:** Log enough to debug, but not so much it's noise.

```python
import logging
import time
from functools import wraps
from contextlib import contextmanager


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


@contextmanager
def log_duration(operation: str, logger: logging.Logger = None):
    """Context manager to log operation duration."""
    logger = logger or logging.getLogger(__name__)
    start = time.perf_counter()
    logger.info(f"Starting: {operation}")
    
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        logger.info(f"Completed: {operation} ({duration:.2f}s)")


def log_pipeline_step(func):
    """Decorator to log pipeline step execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.info(f"→ {func.__name__}")
        
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        
        # Log output size if iterable
        if hasattr(result, '__len__'):
            logger.info(f"  ✓ {func.__name__} → {len(result)} items ({duration:.2f}s)")
        else:
            logger.info(f"  ✓ {func.__name__} ({duration:.2f}s)")
        
        return result
    return wrapper


# Usage
@log_pipeline_step
def chunk_text(text: str, source: str) -> list[Chunk]:
    ...
```

### Example Log Output

```
2026-01-17 10:30:00 | INFO | ingest.extract | → extract_text_from_pdf
2026-01-17 10:30:02 | INFO | ingest.extract |   ✓ extract_text_from_pdf (2.14s)
2026-01-17 10:30:02 | INFO | ingest.chunk | → chunk_text
2026-01-17 10:30:02 | INFO | ingest.chunk |   ✓ chunk_text → 42 items (0.03s)
2026-01-17 10:30:02 | INFO | ingest.embed | → embed_chunks
2026-01-17 10:30:05 | INFO | ingest.embed |   ✓ embed_chunks → 42 items (2.87s)
```

---

## Summary: Quick Reference

| Pattern | Purpose | AI Application |
|---------|---------|----------------|
| **SRP** | One module = one job | Swap models/providers easily |
| **Data Classes** | Explicit contracts | Track lineage, validate data |
| **Pipeline** | Composable transforms | Experiment with orderings |
| **Explicit** | All dependencies passed as arguments | Clear data flow |
| **Dependency Injection** | Swap implementations | Test with mocks, A/B test models |
| **Strategy** | Interchangeable algorithms | Different chunking methods |
| **Factory** | Centralize creation | Config-driven model selection |
| **Repository** | Abstract storage | Swap vector DBs |
| **Configuration** | Externalize tunables | Hyperparameter experiments |
| **Defaults** | Safe, sensible defaults | Reduce configuration burden |
| **Layers** | Orchestration → Logic → Infrastructure | Clear separation of concerns |
| **Observability** | Save intermediate outputs | Debug pipelines |
| **Idempotency** | Safe to re-run | Resilient pipelines |
| **Fail Fast** | Clear error messages | Quick debugging |
| **Retry** | Handle failures | API rate limits, timeouts |
| **Logging** | Track execution | Debug pipelines, track costs |

---

## Applying to RAG Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│                         run_ingest.py                                  │
│                    (Orchestration Layer)                               │
└────────────────────────────────────────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│  extract.py   │      │   chunk.py    │      │   embed.py    │
│  (Strategy:   │  →   │  (Strategy:   │  →   │  (Strategy:   │
│   PyMuPDF /   │      │   Fixed /     │      │   OpenAI /    │
│   Azure DI)   │      │   Semantic)   │      │   Local)      │
└───────────────┘      └───────────────┘      └───────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│ Extracted     │      │   Chunk       │      │ EmbeddedChunk │
│ Document      │      │ (dataclass)   │      │ (dataclass)   │
│ (dataclass)   │      │               │      │               │
└───────────────┘      └───────────────┘      └───────────────┘
                                                      │
                                                      ▼
                              ┌───────────────────────────────────────┐
                              │          VectorRepository             │
                              │  (Repository Pattern: Azure AI Search │
                              │   / Pinecone / In-Memory)             │
                              └───────────────────────────────────────┘
```
