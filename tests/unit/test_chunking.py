import pytest

from app.pipeline.chunking import chunk_text


def test_short_text_is_a_single_chunk() -> None:
    assert chunk_text("Hello world.", size=100, overlap=10) == ["Hello world."]


def test_empty_text_yields_no_chunks() -> None:
    assert chunk_text("  \n\n ", size=100, overlap=10) == []


def test_chunks_respect_paragraph_boundaries_and_size() -> None:
    paragraphs = [f"Paragraph {i} " + "x" * 30 for i in range(10)]

    chunks = chunk_text("\n\n".join(paragraphs), size=100, overlap=0)

    assert all(len(chunk) <= 100 for chunk in chunks)
    assert "".join(chunks).count("Paragraph") == 10
    assert all(chunk.startswith("Paragraph") for chunk in chunks)


def test_chunks_overlap_by_trailing_paragraphs() -> None:
    paragraphs = [f"P{i} " + "y" * 40 for i in range(6)]

    chunks = chunk_text("\n\n".join(paragraphs), size=100, overlap=50)

    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert previous.split("\n\n")[-1] == current.split("\n\n")[0]


def test_oversized_paragraph_is_split_hard() -> None:
    chunks = chunk_text("z" * 250, size=100, overlap=0)

    assert [len(c) for c in chunks] == [100, 100, 50]


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        chunk_text("text", size=10, overlap=10)
