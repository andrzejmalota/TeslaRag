"""
Ingestion pipeline orchestrator.

Local PDF / Object Storage → Extract → Normalize → Chunk → JSON → Embed → Index
"""
from pathlib import Path

from ingest.extract import extract_text_from_pdf, ExtractedDocument
from ingest.chunk import normalize_text, chunk_text, save_chunks_to_json, Chunk
from ingest.embed import embed_chunks, EmbeddedChunk
from providers import get_storage_provider
from providers.storage.base import ObjectInfo
from index.search_index import upsert_embeddings
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.filehandler = logging.FileHandler("logs/ingest.log")
logger.addHandler(logger.filehandler)


def ingest_document(pdf_path: Path, save_chunks: bool = True) -> list[EmbeddedChunk]:
    """Full ingestion pipeline for a single document."""
    # 1. Extract
    doc: ExtractedDocument = extract_text_from_pdf(pdf_path)

    logger.info(f"Extracted {len(doc.text)} characters from {pdf_path.name} with metadata {doc.metadata}")
    logger.info(f"Sample text: {doc.text[:2000]}...")

    # 2. Normalize
    normalized_text: str = normalize_text(doc.text)
    logger.info(f"Sample text: {normalized_text[:2000]}...")
    return []  # Placeholder until other functions are implemented
    
    # 3. Chunk
    chunks: list[Chunk] = chunk_text(normalized_text, source=doc.source)
    
    # # 4. Save chunks for inspection (optional)
    # if save_chunks:
    #     output_path = f"data/chunks/{pdf_path.stem}_chunks.json"
    #     save_chunks_to_json(chunks, output_path)
    
    # # 5. Embed
    # embedded_chunks: list[EmbeddedChunk] = embed_chunks(chunks)
    
    # return embedded_chunks


def ingest_from_storage(object_key: str, bucket: str = "documents") -> list[EmbeddedChunk]:
    """Ingest a PDF directly from cloud object storage."""
    storage = get_storage_provider()
    local_path = storage.download(object_key, bucket=bucket)
    return ingest_document(local_path)


def ingest_all_objects(bucket: str = "documents") -> None:
    """Ingest all PDFs from a cloud storage bucket/container."""
    storage = get_storage_provider()
    for key in storage.list_objects(bucket=bucket):
        if key.lower().endswith(".pdf"):
            embedded_chunks = ingest_from_storage(key, bucket)
            upsert_embeddings(embedded_chunks)
            logger.info(f"Ingested: {key} ({len(embedded_chunks)} chunks)")


def upload_and_ingest(pdf_path: Path, bucket: str = "documents") -> list[EmbeddedChunk]:
    """Upload PDF to object storage, then ingest from local file."""
    storage = get_storage_provider()
    obj_info: ObjectInfo = storage.upload(pdf_path, key=pdf_path.name, bucket=bucket)
    logger.info(f"Uploaded to: {obj_info.url}")
    
    # Ingest from local file (faster than re-downloading)
    embedded_chunks = ingest_document(pdf_path)
    upsert_embeddings(embedded_chunks)
    
    return embedded_chunks


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m scripts.run_ingest local <pdf_directory>")
        print("  python -m scripts.run_ingest cloud [bucket_name]")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "local":
        directory = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/")
        for pdf_path in directory.glob("*.pdf"):
            upload_and_ingest(pdf_path)
    elif mode == "cloud":
        bucket = sys.argv[2] if len(sys.argv) > 2 else "documents"
        ingest_all_objects(bucket)