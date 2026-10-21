import re

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SEPARATOR = "\n\n"


def _joined_length(parts: list[str]) -> int:
    return sum(map(len, parts)) + len(_SEPARATOR) * max(len(parts) - 1, 0)


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Greedy paragraph packing; consecutive chunks share trailing paragraphs worth up to `overlap` chars
    so that relations described across a chunk boundary are still seen together."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")

    pieces: list[str] = []
    for paragraph in (p.strip() for p in _PARAGRAPH_BREAK.split(text)):
        if not paragraph:
            continue
        pieces.extend(paragraph[i : i + size] for i in range(0, len(paragraph), size))

    chunks: list[str] = []
    current: list[str] = []
    for piece in pieces:
        if current and _joined_length([*current, piece]) > size:
            chunks.append(_SEPARATOR.join(current))
            tail: list[str] = []
            for previous in reversed(current):
                if _joined_length([previous, *tail]) > overlap:
                    break
                tail.insert(0, previous)
            while tail and _joined_length([*tail, piece]) > size:
                tail.pop(0)
            current = tail
        current.append(piece)

    if current:
        chunks.append(_SEPARATOR.join(current))
    return chunks
