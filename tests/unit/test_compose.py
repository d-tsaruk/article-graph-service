import re
from pathlib import Path

import pytest

from app.core.config import DEV_APP_TOKEN, DEV_JWT_SECRET, Settings

ROOT = Path(__file__).resolve().parents[2]
NOT_DOCUMENTED = {"NEO4J_URI", "NEO4J_USER"}
SKIPPED_DEFAULTS = {"APP_TOKENS", "JWT_SECRET", "NEO4J_PASSWORD", "OPENAI_API_KEY", "OPENAI_BASE_URL"}


def api_environment() -> dict[str, str]:
    text = (ROOT / "docker-compose.yml").read_text()
    block = re.search(r"^  api:\n(?:.*\n)*?    environment:\n((?:      .*\n|\n)+)", text, re.M)
    assert block, "api.environment not found"
    entries = re.findall(r"^      ([A-Z0-9_]+): (.*)$", block.group(1), re.M)
    return dict(entries)


def setting_names() -> set[str]:
    return {name.upper() for name in Settings.model_fields}


def test_every_setting_reaches_the_container() -> None:
    assert setting_names() - set(api_environment()) == set()


def test_compose_defaults_match_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in setting_names():
        monkeypatch.delenv(name, raising=False)
    base = {"app_tokens": ["t"], "jwt_secret": "x" * 32}
    checked = 0
    for name, value in api_environment().items():
        match = re.fullmatch(r"\$\{" + name + r":-(.*)\}", value)
        if not match or name in SKIPPED_DEFAULTS:
            continue
        field = name.lower()
        overridden = Settings(_env_file=None, **base, **{field: match.group(1)})
        assert getattr(overridden, field) == getattr(Settings(_env_file=None, **base), field), name
        checked += 1
    assert checked > 15


def test_compose_dev_defaults_match_constants() -> None:
    env = api_environment()

    assert env["APP_TOKENS"] == "${APP_TOKENS:-" + DEV_APP_TOKEN + "}"
    assert env["JWT_SECRET"] == "${JWT_SECRET:-" + DEV_JWT_SECRET + "}"


@pytest.mark.parametrize("path", ["README.md", ".env.example"])
def test_every_setting_is_documented(path: str) -> None:
    text = (ROOT / path).read_text()

    assert [name for name in sorted(setting_names() - NOT_DOCUMENTED) if name not in text] == []
