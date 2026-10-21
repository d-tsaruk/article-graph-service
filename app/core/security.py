import hashlib
import hmac
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum

import jwt

from app.core.config import Settings


class TokenType(StrEnum):
    ACCESS = "access"
    REFRESH = "refresh"


class InvalidTokenError(Exception):
    pass


@dataclass(frozen=True)
class IssuedTokens:
    access_token: str
    refresh_token: str
    access_ttl_seconds: int
    refresh_ttl_seconds: int


class TokenService:
    """Stateless JWTs. Each token has a `jti`, so a revocation list can be added without format changes."""

    def __init__(self, settings: Settings) -> None:
        self._app_tokens = [token.encode() for token in settings.app_tokens]
        self._secret = settings.jwt_secret.get_secret_value()
        self._algorithm = settings.jwt_algorithm
        self._ttl = {
            TokenType.ACCESS: settings.access_token_ttl_seconds,
            TokenType.REFRESH: settings.refresh_token_ttl_seconds,
        }

    def authenticate_app(self, app_token: str) -> str | None:
        """Returns the client subject for a valid App Token."""
        candidate = app_token.encode()
        # Compare against every token to keep timing independent of which one matched.
        matched = [hmac.compare_digest(candidate, known) for known in self._app_tokens]
        if not any(matched):
            return None
        return "app:" + hashlib.sha256(candidate).hexdigest()[:16]

    def issue(self, subject: str) -> IssuedTokens:
        return IssuedTokens(
            access_token=self._encode(subject, TokenType.ACCESS),
            refresh_token=self._encode(subject, TokenType.REFRESH),
            access_ttl_seconds=self._ttl[TokenType.ACCESS],
            refresh_ttl_seconds=self._ttl[TokenType.REFRESH],
        )

    def verify(self, token: str, expected_type: TokenType) -> str:
        try:
            claims = jwt.decode(
                token, self._secret, algorithms=[self._algorithm], options={"require": ["exp", "sub", "type"]}
            )
        except jwt.PyJWTError as exc:
            raise InvalidTokenError(str(exc)) from exc
        if claims["type"] != expected_type:
            raise InvalidTokenError(f"expected {expected_type} token")
        return claims["sub"]

    def _encode(self, subject: str, token_type: TokenType) -> str:
        now = int(time.time())
        claims = {
            "sub": subject,
            "type": token_type.value,
            "iat": now,
            "exp": now + self._ttl[token_type],
            "jti": uuid.uuid4().hex,
        }
        return jwt.encode(claims, self._secret, algorithm=self._algorithm)
