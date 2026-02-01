"""
Ingestion pipeline orchestrator.

Local PDF / Azure Blob → Extract → Normalize → Chunk → JSON → Embed → Index
"""
from pathlib import Path

from ingest.extract import extract_text_from_pdf, ExtractedDocument
from ingest.chunk import normalize_text, chunk_text, save_chunks_to_json, Chunk
from ingest.embed import embed_chunks, EmbeddedChunk
from ingest.blob import upload_pdf, download_blob, list_blobs, BlobInfo
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


def ingest_from_blob(blob_name: str, container: str = "documents") -> list[EmbeddedChunk]:
    """Ingest a PDF directly from Azure Blob Storage."""
    # Download blob to temp location
    local_path = download_blob(blob_name, container)
    
    # Run standard pipeline
    embedded_chunks = ingest_document(local_path)
    
    return embedded_chunks


def ingest_all_blobs(container: str = "documents") -> None:
    """Ingest all PDFs from a blob container."""
    for blob_name in list_blobs(container):
        if blob_name.lower().endswith(".pdf"):
            embedded_chunks = ingest_from_blob(blob_name, container)
            upsert_embeddings(embedded_chunks)
            logger.info(f"Ingested from blob: {blob_name} ({len(embedded_chunks)} chunks)")


def upload_and_ingest(pdf_path: Path, container: str = "documents") -> list[EmbeddedChunk]:
    """Upload PDF to blob storage, then ingest."""
    # Upload to blob storage for archival
    blob_info: BlobInfo = upload_pdf(pdf_path, container)
    logger.info(f"Uploaded to: {blob_info.url}")
    
    # Ingest from local file (faster than re-downloading)
    embedded_chunks = ingest_document(pdf_path)
    upsert_embeddings(embedded_chunks)
    
    return embedded_chunks


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m scripts.run_ingest local <pdf_directory>")
        print("  python -m scripts.run_ingest blob [container_name]")
        sys.exit(1)
    
    mode = sys.argv[1]
    # mode = 'local'
    if mode == "local":
        directory = Path("/Users/andrzej.malota/Desktop/andrzej/TeslaRAG/data/")
        # directory = Path(sys.argv[2])
        for pdf_path in directory.glob("*.pdf"):
            upload_and_ingest(pdf_path)
    elif mode == "blob":
        container = sys.argv[2] if len(sys.argv) > 2 else "documents"
        ingest_all_blobs(container)