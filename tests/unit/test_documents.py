import pytest

from app.pipeline.documents import DocumentError, PdfOptions, VisualKind, parse_markdown, parse_pdf
from tests.factories import build_pdf

OPTIONS = PdfOptions(
    max_visual_items=10,
    min_image_side_px=100,
    render_dpi=72,
    scanned_page_max_chars=10,
    drawings_render_threshold=20,
)


def test_markdown_title_is_taken_from_first_heading() -> None:
    document = parse_markdown("Intro line\n\n## Chapter 2\n\nText")

    assert document.title == "Chapter 2"
    assert document.pages[0].text == "Intro line\n\n## Chapter 2\n\nText"
    assert document.pages[0].visuals == []


def test_markdown_without_heading_has_no_title() -> None:
    assert parse_markdown("Just text").title is None


def test_pdf_text_and_embedded_images_are_extracted() -> None:
    pdf = build_pdf([{"text": "Harry Potter went to Hogwarts.", "image_size": (300, 200)}])

    document = parse_pdf(pdf, OPTIONS)

    assert len(document.pages) == 1
    page = document.pages[0]
    assert "Harry Potter went to Hogwarts." in page.text
    assert [v.kind for v in page.visuals] == [VisualKind.EMBEDDED_IMAGE]
    assert page.visuals[0].mime_type in {"image/png", "image/jpeg"}
    assert page.visuals[0].data


def test_tiny_images_are_ignored() -> None:
    pdf = build_pdf([{"text": "Text", "image_size": (20, 20)}])

    assert parse_pdf(pdf, OPTIONS).pages[0].visuals == []


def test_scanned_page_is_rendered_as_a_whole() -> None:
    pdf = build_pdf([{"image_size": (600, 800)}])

    visuals = parse_pdf(pdf, OPTIONS).pages[0].visuals

    assert [v.kind for v in visuals] == [VisualKind.PAGE_RENDER]
    assert visuals[0].mime_type == "image/png"


def test_page_with_vector_graphics_is_rendered_instead_of_individual_images() -> None:
    pdf = build_pdf([{"text": "A chart follows.", "image_size": (300, 300), "drawings": 40}])

    visuals = parse_pdf(pdf, OPTIONS).pages[0].visuals

    assert [v.kind for v in visuals] == [VisualKind.PAGE_RENDER]


def test_same_image_reused_on_several_pages_is_analyzed_once() -> None:
    import pymupdf

    from tests.factories import solid_pixmap

    document = pymupdf.open()
    first = document.new_page()
    first.insert_text((50, 50), "Page one text")
    xref = first.insert_image(pymupdf.Rect(100, 100, 300, 300), pixmap=solid_pixmap(300, 300))
    second = document.new_page()
    second.insert_text((50, 50), "Page two text")
    second.insert_image(pymupdf.Rect(100, 100, 300, 300), xref=xref)
    pdf = document.tobytes()

    parsed = parse_pdf(pdf, OPTIONS)

    assert [len(p.visuals) for p in parsed.pages] == [1, 0]


def test_visual_items_are_capped() -> None:
    pdf = build_pdf([{"text": f"Page {i}", "image_size": (300, 300 + i)} for i in range(5)])
    options = PdfOptions(**{**OPTIONS.__dict__, "max_visual_items": 2})

    parsed = parse_pdf(pdf, options)

    assert sum(len(p.visuals) for p in parsed.pages) == 2


def test_pdf_title_falls_back_to_metadata() -> None:
    import pymupdf

    document = pymupdf.open()
    document.new_page().insert_text((50, 50), "Body")
    document.set_metadata({"title": "The Philosopher's Stone"})

    assert parse_pdf(document.tobytes(), OPTIONS).title == "The Philosopher's Stone"


def test_invalid_pdf_raises_document_error() -> None:
    with pytest.raises(DocumentError):
        parse_pdf(b"%PDF-1.7 not really", OPTIONS)
