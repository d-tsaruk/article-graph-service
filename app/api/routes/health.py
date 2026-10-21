from fastapi import APIRouter

from app.api.deps import ServicesDep

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness and Neo4j connectivity check")
async def health(services: ServicesDep) -> dict[str, str]:
    await services.repository.ping()
    return {"status": "ok"}
