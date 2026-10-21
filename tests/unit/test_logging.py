import logging

from app.core.config import Settings
from app.main import create_app
from tests.fakes import FakeLLMClient


def test_create_app_enables_info_logging_for_the_app_namespace() -> None:
    settings = Settings(_env_file=None, app_tokens=["t"], jwt_secret="x" * 32)

    create_app(settings, llm_client=FakeLLMClient(8))

    assert logging.getLogger("app").getEffectiveLevel() == logging.INFO
