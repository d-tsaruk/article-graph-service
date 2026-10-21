import logging

import pytest

from app.llm.base import LLMError
from app.pipeline.ingestion import unwrap_plan_failure


def test_llm_error_is_preferred_and_every_failure_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    group = ExceptionGroup("plans", [RuntimeError("neo4j down"), LLMError("boom")])

    with caplog.at_level(logging.ERROR, logger="app.pipeline.ingestion"):
        chosen = unwrap_plan_failure(group)

    assert isinstance(chosen, LLMError)
    assert "neo4j down" in caplog.text
    assert "boom" in caplog.text


def test_first_failure_is_chosen_without_an_llm_error() -> None:
    group = ExceptionGroup("plans", [RuntimeError("first"), ValueError("second")])

    assert str(unwrap_plan_failure(group)) == "first"
