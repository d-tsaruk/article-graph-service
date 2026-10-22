import logging
import math
import re
from dataclasses import dataclass, field
from enum import StrEnum

import pymupdf

from app.core.config import Settings

logger = logging.getLogger(__name__)

_MARKDOWN_HEADING = re.compile(r"^ {0,3}#{1,6}[ \t]+(.+?)[ \t#]*$", re.MULTILINE)
_PASSTHROUGH_IMAGE_TYPES = {"png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg"}


class DocumentError(Exception):
    """The uploaded document cannot be read."""


class VisualKind(StrEnum):
    EMBEDDED_IMAGE = "embedded_image"
    # Whole page rasterised: scanned pages and vector graphics (charts, diagrams) that are not bitmaps.
    PAGE_RENDER = "page_render"


@dataclass(frozen=True)
class Visual:
    kind: VisualKind
    page_number: int
    data: bytes
    mime_type: str


@dataclass
class Page:
    number: int
    text: str
    visuals: list[Visual] = field(default_factory=list)


@dataclass
class ParsedDocument:
    title: str | None
    pages: list[Page]


@dataclass(frozen=True)
class PdfOptions:
    max_visual_items: int
    min_image_side_px: int
    render_dpi: int
    scanned_page_max_chars: int
    drawings_render_threshold: int
    max_pages: int
    max_render_side_px: int

    @classmethod
    def from_settings(cls, settings: Settings) -> "PdfOptions":
        return cls(
            max_visual_items=settings.pdf_max_visual_items,
            min_image_side_px=settings.pdf_min_image_side_px,
            render_dpi=settings.pdf_render_dpi,
            scanned_page_max_chars=settings.pdf_scanned_page_max_chars,
            drawings_render_threshold=settings.pdf_drawings_render_threshold,
            max_pages=settings.max_pdf_pages,
            max_render_side_px=settings.pdf_max_render_side_px,
        )


def parse_markdown(text: str) -> ParsedDocument:
    heading = _MARKDOWN_HEADING.search(text)
    return ParsedDocument(title=heading.group(1) if heading else None, pages=[Page(number=1, text=text.strip())])


def parse_pdf(data: bytes, options: PdfOptions) -> ParsedDocument:
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several unrelated types for malformed input
        raise DocumentError(f"cannot open PDF: {exc}") from exc

    with document:
        if document.needs_pass:
            raise DocumentError("encrypted PDFs are not supported")
        if document.page_count > options.max_pages:
            raise DocumentError(
                f"PDF has {document.page_count} pages; the limit is {options.max_pages} (MAX_PDF_PAGES)"
            )
        try:
            pages = _PdfReader(document, options).read()
        except DocumentError:
            raise
        except Exception as exc:
            raise DocumentError(f"cannot read PDF: {exc}") from exc
        title = (document.metadata or {}).get("title") or None

    if not pages:
        raise DocumentError("PDF has no pages")
    return ParsedDocument(title=title.strip() if title else None, pages=pages)


class _PdfReader:
    def __init__(self, document: pymupdf.Document, options: PdfOptions) -> None:
        self._document = document
        self._options = options
        self._budget = options.max_visual_items
        self._seen_xrefs: set[int] = set()

    def read(self) -> list[Page]:
        pages = [self._read_page(page) for page in self._document]
        if self._budget == 0:
            logger.warning("Visual item limit (%d) reached; remaining images skipped", self._options.max_visual_items)
        return pages

    def _read_page(self, page: pymupdf.Page) -> Page:
        number = page.number + 1
        text = page.get_text("text", sort=True).strip()
        result = Page(number=number, text=text)
        if self._budget == 0:
            return result

        large_xrefs = [
            image[0]
            for image in page.get_images(full=True)
            if min(image[2], image[3]) >= self._options.min_image_side_px
        ]
        is_scanned = bool(large_xrefs) and len(text) <= self._options.scanned_page_max_chars
        has_vector_graphics = len(page.get_drawings()) >= self._options.drawings_render_threshold

        if is_scanned or has_vector_graphics:
            pixmap = page.get_pixmap(matrix=self._render_matrix(page))
            result.visuals.append(Visual(VisualKind.PAGE_RENDER, number, pixmap.tobytes("png"), "image/png"))
            self._budget -= 1
            self._seen_xrefs.update(large_xrefs)
            return result

        for xref in large_xrefs:
            if self._budget == 0:
                break
            if xref in self._seen_xrefs:
                continue
            self._seen_xrefs.add(xref)
            if visual := self._extract_image(xref, number):
                result.visuals.append(visual)
                self._budget -= 1
        return result

    def _render_matrix(self, page: pymupdf.Page) -> pymupdf.Matrix:
        longest_side = max(page.rect.width, page.rect.height)
        # Floored so that the rounded-up pixel size never exceeds the limit.
        fit = math.floor(self._options.max_render_side_px / longest_side * 1000) / 1000
        zoom = min(self._options.render_dpi / 72, fit)
        return pymupdf.Matrix(zoom, zoom)

    def _extract_image(self, xref: int, page_number: int) -> Visual | None:
        info = self._document.extract_image(xref)
        if not info:
            return None
        longest = max(info["width"], info["height"])
        oversized = longest > self._options.max_render_side_px
        if not oversized and (mime_type := _PASSTHROUGH_IMAGE_TYPES.get(info["ext"])):
            return Visual(VisualKind.EMBEDDED_IMAGE, page_number, info["image"], mime_type)

        # Oversized images and formats vision APIs do not accept (JPX, JBIG2, CMYK, ...) are re-encoded as PNG.
        pixmap = pymupdf.Pixmap(self._document, xref)
        if pixmap.colorspace and pixmap.colorspace.n > 3:
            pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
        if oversized:
            # shrink(n) divides each side by 2**n.
            pixmap.shrink(math.ceil(math.log2(longest / self._options.max_render_side_px)))
        return Visual(VisualKind.EMBEDDED_IMAGE, page_number, pixmap.tobytes("png"), "image/png")
