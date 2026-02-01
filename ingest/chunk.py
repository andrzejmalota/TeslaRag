from dataclasses import dataclass
import re
import unicodedata


@dataclass
class Chunk:
    """Represents a text chunk."""
    id: str
    text: str
    source: str
    chunk_index: int
    metadata: dict  # start_char, end_char, page_number, etc.


def normalize_text(text: str) -> str:
    """
    Normalize extracted text.
    
    - Remove excessive whitespace
    - Fix encoding issues
    - Standardize line breaks
    """
    if text is None:
        return ""
    
    # Normalize unicode (e.g., smart quotes, ligatures)
    text = unicodedata.normalize("NFKC", text)
    
    # Standardize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Collapse spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    
    # Trim whitespace on each line
    text = "\n".join(line.strip() for line in text.split("\n"))
    
    # Final trim
    return text.strip()


def _pages_from_markers(text: str) -> list[tuple[int, str]]:
    """
    Parse [[PAGE=n]] markers and return list of (page_num, page_text).
    If no markers, returns [(1, text)].
    """
    marker = re.compile(r"\[\[PAGE=(\d+)\]\]")
    matches = list(marker.finditer(text))
    if not matches:
        return [(1, text)]

    pages: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        page_num = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        page_text = text[start:end]
        pages.append((page_num, page_text))
    return pages


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    sections: list[dict] | None = None
) -> list[Chunk]:
    """
    Split text into chunks.

    If sections are provided, creates one chunk per section (based on page ranges).
    Otherwise, falls back to fixed-size chunking.
    """
    pages = _pages_from_markers(text)

    # Chunk per section
    if sections:
        chunks: list[Chunk] = []
        for idx, s in enumerate(sections):
            page_start = s["page_start"]
            page_end = s["page_end"]
            section_pages = [
                ptext for pnum, ptext in pages
                if page_start <= pnum <= page_end
            ]
            section_text = normalize_text("\n".join(section_pages)).strip()
            if not section_text:
                continue

            metadata = {
                "section_title": s["section_title"],
                "page_start": page_start,
                "page_end": page_end,
            }
            chunks.append(
                Chunk(
                    id=f"{source}::section{idx}",
                    text=section_text,
                    source=source,
                    chunk_index=idx,
                    metadata=metadata,
                )
            )
        return chunks

    # Fallback: fixed-size chunking
    clean_text = normalize_text(
        re.sub(r"\[\[PAGE=\d+\]\]", "", text)
    )

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    chunks: list[Chunk] = []
    step = chunk_size - chunk_overlap
    chunk_index = 0

    for start in range(0, len(clean_text), step):
        end = min(start + chunk_size, len(clean_text))
        chunk_str = clean_text[start:end].strip()
        if not chunk_str:
            continue

        metadata = {"start_char": start, "end_char": end}
        chunks.append(
            Chunk(
                id=f"{source}::chunk{chunk_index}",
                text=chunk_str,
                source=source,
                chunk_index=chunk_index,
                metadata=metadata,
            )
        )
        chunk_index += 1

    return chunks


def save_chunks_to_json(chunks: list[Chunk], output_path: str) -> None:
    """Save chunks to JSON for inspection/debugging."""
    raise NotImplementedError