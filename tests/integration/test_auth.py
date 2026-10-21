import time

import jwt
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from tests.integration.conftest import APP_TOKEN

PROTECTED_ENDPOINTS = [
    ("GET", "/characters"),
    ("GET", "/characters/by_article/some-id"),
    ("GET", "/relations/types"),
    ("GET", "/relations/connected/some-id?type=FRIEND_OF"),
    ("POST", "/articles/upload"),
]


def test_app_token_is_exchanged_for_token_pair(client: TestClient, settings: Settings) -> None:
    response = client.post("/auth/token", json={"app_token": APP_TOKEN})

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == settings.access_token_ttl_seconds
    assert body["refresh_expires_in"] == settings.refresh_token_ttl_seconds
    secret = settings.jwt_secret.get_secret_value()
    access = jwt.decode(body["access_token"], secret, algorithms=["HS256"])
    refresh = jwt.decode(body["refresh_token"], secret, algorithms=["HS256"])
    assert access["type"] == "access"
    assert refresh["type"] == "refresh"
    assert access["sub"] == refresh["sub"]


def test_unknown_app_token_is_rejected(client: TestClient) -> None:
    response = client.post("/auth/token", json={"app_token": "wrong"})

    assert response.status_code == 401


@pytest.mark.parametrize(("method", "path"), PROTECTED_ENDPOINTS)
def test_protected_endpoints_require_access_token(client: TestClient, method: str, path: str) -> None:
    response = client.request(method, path)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_refresh_token_cannot_be_used_as_access_token(client: TestClient) -> None:
    tokens = client.post("/auth/token", json={"app_token": APP_TOKEN}).json()

    response = client.get("/characters", headers={"Authorization": f"Bearer {tokens['refresh_token']}"})

    assert response.status_code == 401


def test_refresh_token_yields_new_working_pair(client: TestClient) -> None:
    tokens = client.post("/auth/token", json={"app_token": APP_TOKEN}).json()

    response = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})

    assert response.status_code == 200
    new_access = response.json()["access_token"]
    assert client.get("/characters", headers={"Authorization": f"Bearer {new_access}"}).status_code == 200


def test_access_token_cannot_be_used_to_refresh(client: TestClient) -> None:
    tokens = client.post("/auth/token", json={"app_token": APP_TOKEN}).json()

    response = client.post("/auth/refresh", json={"refresh_token": tokens["access_token"]})

    assert response.status_code == 401


def test_expired_access_token_is_rejected(client: TestClient, settings: Settings) -> None:
    now = int(time.time())
    expired = jwt.encode(
        {"sub": "app", "type": "access", "iat": now - 120, "exp": now - 60, "jti": "x"},
        settings.jwt_secret.get_secret_value(),
        algorithm="HS256",
    )

    response = client.get("/characters", headers={"Authorization": f"Bearer {expired}"})

    assert response.status_code == 401
