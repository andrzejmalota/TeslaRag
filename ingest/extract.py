from dataclasses import dataclass
from pathlib import Path
import re
import pdfplumber

@dataclass
class ExtractedDocument:
    """Represents extracted content from a PDF."""
    source: str
    text: str
    metadata: dict  # page count, file size, etc.


def _normalize_toc_text(raw_text: str) -> str:
    # Insert newlines between concatenated entries like "138Tire"
    text = re.sub(r"(\d)([A-Z])", r"\1\n\2", raw_text)
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    return text


def _extract_toc_top_level(raw_text: str) -> list[dict]:
    """
    Extract top-level TOC entries (those with dot leaders).
    Returns list of dicts: {section_title, page_start}
    """
    text = _normalize_toc_text(raw_text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Try to isolate content after "Contents"
    if "Contents" in lines:
        start_idx = lines.index("Contents") + 1
        lines = lines[start_idx:]

    entries = []
    # Dot-leader pattern: "Doors.................................. 4"
    dot_leader = re.compile(r"^(?P<title>.+?)\.{2,}\s*(?P<page>\d{1,4})$")

    for ln in lines:
        m = dot_leader.match(ln)
        if m:
            title = m.group("title").strip()
            page = int(m.group("page"))
            entries.append({"section_title": title, "page_start": page})

    return entries


def _compute_section_ranges(toc_entries: list[dict], page_count: int) -> list[dict]:
    """
    Given top-level entries with page_start, compute page_end from next entry.
    """
    if not toc_entries:
        return []

    # Keep order as in TOC
    sections = []
    for i, entry in enumerate(toc_entries):
        start = entry["page_start"]
        if i + 1 < len(toc_entries):
            next_start = toc_entries[i + 1]["page_start"]
            end = max(start, next_start - 1)
        else:
            end = page_count
        sections.append(
            {
                "section_title": entry["section_title"],
                "page_start": start,
                "page_end": end,
            }
        )
    return sections


def extract_text_from_pdf(pdf_path: Path) -> ExtractedDocument:
    """
    Extract text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        ExtractedDocument with raw text and metadata
    """
    pdf_text = ""
    page_texts: list[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            page_texts.append(page_text)
            pdf_text += f"[[PAGE={i}]]\n{page_text}\n"

        metadata = {"page_count": len(pdf.pages)}

    # Extract TOC-based section ranges into metadata
    toc_entries = _extract_toc_top_level(pdf_text)
    sections = _compute_section_ranges(toc_entries, metadata["page_count"])
    metadata["sections"] = sections

    return ExtractedDocument(source=str(pdf_path), text=pdf_text, metadata=metadata)


def extract_from_blob(blob_url: str) -> ExtractedDocument:
    """Extract text from a PDF stored in Azure Blob Storage."""
    # TODO: Download blob, then extract
    raise NotImplementedError