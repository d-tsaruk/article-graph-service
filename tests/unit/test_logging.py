import logging

import pytest

from app.core.config import DEV_APP_TOKEN, DEV_JWT_SECRET, Settings
from app.main import create_app
from tests.fakes import FakeLLMClient


def test_create_app_enables_info_logging_for_the_app_namespace() -> None:
    settings = Settings(_env_file=None, app_tokens=["t"], jwt_secret="x" * 32)

    create_app(settings, llm_client=FakeLLMClient(8))

    assert logging.getLogger("app").getEffectiveLevel() == logging.INFO


def test_dev_credentials_warn_at_startup(caplog: pytest.LogCaptureFixture) -> None:
    settings = Settings(_env_file=None, app_tokens=[DEV_APP_TOKEN], jwt_secret=DEV_JWT_SECRET)

    with caplog.at_level(logging.WARNING, logger="app.main"):
        create_app(settings, llm_client=FakeLLMClient(8))

    messages = [record.getMessage() for record in caplog.records]
    assert sum("JWT_SECRET" in m for m in messages) == 1
    assert sum("APP_TOKENS" in m for m in messages) == 1


def test_custom_credentials_do_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    settings = Settings(_env_file=None, app_tokens=["real-token"], jwt_secret="y" * 32)

    with caplog.at_level(logging.WARNING, logger="app.main"):
        create_app(settings, llm_client=FakeLLMClient(8))

    messages = [record.getMessage() for record in caplog.records]
    assert not any("JWT_SECRET" in m or "APP_TOKENS" in m for m in messages)
