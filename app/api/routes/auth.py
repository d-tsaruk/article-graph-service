from fastapi import APIRouter

from app.api.deps import ServicesDep, unauthorized
from app.core.security import InvalidTokenError, IssuedTokens, TokenType
from app.schemas.api import RefreshRequest, TokenPair, TokenRequest

router = APIRouter(prefix="/auth", tags=["auth"])


def _to_response(tokens: IssuedTokens) -> TokenPair:
    return TokenPair(
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        expires_in=tokens.access_ttl_seconds,
        refresh_expires_in=tokens.refresh_ttl_seconds,
    )


@router.post("/token", summary="Exchange an App Token for access and refresh tokens")
async def issue_tokens(body: TokenRequest, services: ServicesDep) -> TokenPair:
    subject = services.tokens.authenticate_app(body.app_token)
    if subject is None:
        raise unauthorized("Invalid app token")
    return _to_response(services.tokens.issue(subject))


@router.post("/refresh", summary="Exchange a refresh token for a new token pair")
async def refresh_tokens(body: RefreshRequest, services: ServicesDep) -> TokenPair:
    try:
        subject = services.tokens.verify(body.refresh_token, TokenType.REFRESH)
    except InvalidTokenError as exc:
        raise unauthorized("Invalid or expired refresh token") from exc
    return _to_response(services.tokens.issue(subject))
