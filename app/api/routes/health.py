import logging

from fastapi import APIRouter, Response, status
from neo4j.exceptions import DriverError, Neo4jError

from app.api.deps import ServicesDep

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get(
    "/health",
    summary="Liveness and Neo4j connectivity check",
    responses={503: {"description": "Neo4j is unreachable"}},
)
async def health(services: ServicesDep, response: Response) -> dict[str, str]:
    try:
        await services.repository.ping()
    except (DriverError, Neo4jError) as exc:
        logger.warning("Neo4j health check failed: %r", exc)
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unavailable"}
    return {"status": "ok"}
