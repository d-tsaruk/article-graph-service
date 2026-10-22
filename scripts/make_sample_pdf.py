"""Generates samples/03-bakery-chart.pdf: a PDF whose relations exist only inside an embedded chart image.

Run from the repository root: .venv/bin/python scripts/make_sample_pdf.py
"""

from pathlib import Path

import pymupdf

TITLE = "The Varga Bakery"
BODY = (
    "Elena Varga lives in Trieste and rises before dawn. Mateo Rossi is a twelve-year-old who wants to become a "
    "pilot and fills notebooks with drawings of aeroplanes. Ivo Petrov is a quiet man with a warm laugh who hums "
    "while he walks. The chart below shows how they are connected."
)
OUTPUT = Path(__file__).resolve().parents[1] / "samples" / "03-bakery-chart.pdf"

BOX_W, BOX_H = 130, 44
ELENA = pymupdf.Rect(235, 40, 235 + BOX_W, 40 + BOX_H)
MATEO = pymupdf.Rect(60, 190, 60 + BOX_W, 190 + BOX_H)
IVO = pymupdf.Rect(410, 190, 410 + BOX_W, 190 + BOX_H)


def _box(page: pymupdf.Page, rect: pymupdf.Rect, label: str) -> None:
    page.draw_rect(rect, width=1.5)
    page.insert_text(pymupdf.Point(rect.x0 + 14, rect.y0 + 27), label, fontsize=14)


def _arrow(page: pymupdf.Page, start: pymupdf.Point, end: pymupdf.Point, label: str) -> None:
    page.draw_line(start, end, width=1.5)
    direction = (end - start) / abs(end - start)
    normal = pymupdf.Point(-direction.y, direction.x)
    for side in (1, -1):
        page.draw_line(end, end - direction * 12 + normal * (6 * side), width=1.5)
    middle = (start + end) / 2
    page.insert_text(middle + pymupdf.Point(8, -6), label, fontsize=12)


def _chart_png() -> bytes:
    document = pymupdf.open()
    page = document.new_page(width=600, height=300)
    for rect, label in ((ELENA, "Elena Varga"), (MATEO, "Mateo Rossi"), (IVO, "Ivo Petrov")):
        _box(page, rect, label)
    _arrow(page, pymupdf.Point(260, 84), pymupdf.Point(160, 190), "mother of")
    _arrow(page, pymupdf.Point(470, 190), pymupdf.Point(340, 84), "works for")
    page.insert_text(pymupdf.Point(170, 285), "Varga household and bakery staff", fontsize=13)
    png = page.get_pixmap(dpi=150).tobytes("png")
    document.close()
    return png


def build() -> bytes:
    png = _chart_png()
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(pymupdf.Point(50, 70), TITLE, fontsize=20)
    # insert_textbox wraps the text; insert_text does not.
    page.insert_textbox(pymupdf.Rect(50, 90, 545, 300), BODY, fontsize=11)
    page.insert_image(pymupdf.Rect(50, 320, 545, 570), stream=png)
    document.set_metadata({"title": TITLE})
    data = document.tobytes(garbage=4, deflate=True, no_new_id=True)
    document.close()
    return data


def main() -> None:
    OUTPUT.write_bytes(build())
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
