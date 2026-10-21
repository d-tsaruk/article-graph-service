from pathlib import Path

import pytest

from app.llm import prompts

README = Path(__file__).resolve().parents[2] / "README.md"


@pytest.mark.parametrize(
    "prompt",
    [
        prompts.EXTRACTION_SYSTEM_PROMPT,
        prompts.EXTRACTION_USER_TEMPLATE,
        prompts.VISION_SYSTEM_PROMPT,
        prompts.VISION_USER_TEMPLATE,
    ],
)
def test_readme_documents_the_prompts_actually_used(prompt: str) -> None:
    assert prompt.strip() in README.read_text(encoding="utf-8")
