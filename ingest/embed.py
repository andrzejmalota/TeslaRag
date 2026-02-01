"""Embedding generation module."""
from dataclasses import dataclass
import numpy as np

from ingest.chunk import Chunk


@dataclass
class EmbeddedChunk:
    """Chunk with its vector embedding."""
    chunk: Chunk
    embedding: np.ndarray  # or list[float]


def embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    """
    Generate embeddings for chunks.
    
    Args:
        chunks: List of text chunks
        
    Returns:
        List of chunks with embeddings
    """
    # TODO: Use Azure OpenAI, OpenAI, or local model
    raise NotImplementedError


def embed_batch(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Embed texts in batches for efficiency."""
    raise NotImplementedError