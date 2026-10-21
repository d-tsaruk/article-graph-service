from pathlib import Path

import pytest

from app.llm import prompts

README = Path(__file__).resolve().parents[2] / "README.md"

PROMPTS = {name: value for name, value in vars(prompts).items() if name.isupper() and isinstance(value, str)}


@pytest.mark.parametrize("name", sorted(PROMPTS))
def test_readme_documents_the_prompts_actually_used(name: str) -> None:
    assert PROMPTS[name].strip() in README.read_text(encoding="utf-8")
