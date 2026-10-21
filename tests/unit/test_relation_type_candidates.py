from app.graph.repository import RelationTypeMatch
from app.pipeline.resolution import RELATION_TYPE_CANDIDATES, RelationTypeResolver


class StubRepository:
    """Returns as many ranked matches as asked for, like an approximate vector index."""

    def __init__(self, matches: list[RelationTypeMatch]) -> None:
        self._matches = matches
        self.limits: list[int] = []

    async def find_similar_relation_types(self, embedding: list[float], limit: int) -> list[RelationTypeMatch]:
        self.limits.append(limit)
        return self._matches[:limit]


async def test_candidates_are_over_fetched_so_approximate_search_has_headroom_but_capped_after_filtering() -> None:
    matches = [RelationTypeMatch(f"TYPE_{i}", 1.0 - i * 0.01) for i in range(30)]
    repository = StubRepository(matches)
    resolver = RelationTypeResolver(repository, llm=None, candidate_score=0.75)  # type: ignore[arg-type]

    found = await resolver._candidates([0.0], exclude_inverse_of="OTHER")

    assert repository.limits[0] > RELATION_TYPE_CANDIDATES
    assert found == [f"TYPE_{i}" for i in range(RELATION_TYPE_CANDIDATES)]
