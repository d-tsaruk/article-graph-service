import importlib.util
import re
from pathlib import Path

import pymupdf
import pytest

from app.core.config import Settings
from app.pipeline.documents import PdfOptions, VisualKind, parse_pdf

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "make_sample_pdf.py"
COMMITTED = ROOT / "samples" / "03-bakery-chart.pdf"
RELATION_WORDS = re.compile(r"\b(mother|son|works?|employ\w*|parent)\b", re.IGNORECASE)


def load_script():
    spec = importlib.util.spec_from_file_location("make_sample_pdf", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["committed", "generated"])
def parsed(request: pytest.FixtureRequest):
    data = COMMITTED.read_bytes() if request.param == "committed" else load_script().build()
    options = PdfOptions.from_settings(Settings(_env_file=None, app_tokens=["t"], jwt_secret="x" * 32))
    return parse_pdf(data, options)


def test_pdf_has_one_page_with_the_names(parsed) -> None:
    assert len(parsed.pages) == 1
    text = " ".join(parsed.pages[0].text.split())
    for name in ("Elena Varga", "Mateo Rossi", "Ivo Petrov"):
        assert name in text


def test_relations_are_not_stated_in_the_text(parsed) -> None:
    assert RELATION_WORDS.search(parsed.pages[0].text) is None


def test_the_chart_is_the_only_visual(parsed) -> None:
    visuals = parsed.pages[0].visuals

    assert [v.kind for v in visuals] == [VisualKind.EMBEDDED_IMAGE]
    chart = pymupdf.Pixmap(visuals[0].data)
    assert min(chart.width, chart.height) >= 100


def test_title_comes_from_the_metadata(parsed) -> None:
    assert parsed.title == "The Varga Bakery"


def test_body_does_not_hint_at_the_employment_relation() -> None:
    assert re.search(r"\b(bak\w*|employ\w*|boss|staff|hired?|job|colleague\w*)\b", load_script().BODY, re.I) is None
