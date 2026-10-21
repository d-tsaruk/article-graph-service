"""Turns the LLM judge's verdicts into merge groups. Anything the judge gets wrong degrades to "new"."""

import logging
from dataclasses import dataclass

from app.llm.schemas import Judgement

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Group:
    members: tuple[str, ...]  # N refs in item order
    canonical: str  # N ref whose name the group keeps when it is created
    stored: str | None  # S ref the whole group merges into


def group_verdicts(
    candidates: dict[str, tuple[str, ...]],
    judgement: Judgement | None,
    *,
    kind: str = "item",
    labels: dict[str, str] | None = None,
) -> list[Group]:
    """`candidates` maps each N ref, in item order, to the S refs offered for it.
    `kind` and `labels` (ref -> name) only make the warnings readable."""
    describe = _Describer(kind, labels or {})
    stored: dict[str, str] = {}
    same_as: dict[str, str] = {}
    if judgement is not None:
        stored, same_as = _valid_decisions(candidates, judgement, describe)

    parent = {item: item for item in candidates}
    for item, match in same_as.items():
        parent[_find(parent, item)] = _find(parent, match)

    components: dict[str, list[str]] = {}
    for item in candidates:
        components.setdefault(_find(parent, item), []).append(item)

    groups: list[Group] = []
    for members in components.values():  # ordered by each component's first member
        # Every item has at most one outgoing edge, so a component has one member without it, or is a cycle.
        root = next((item for item in members if item not in same_as), None)
        if root is None:
            groups.append(Group(tuple(members), canonical=members[0], stored=None))
        else:
            groups.append(Group(tuple(members), canonical=root, stored=stored.get(root)))
    return groups


class _Describer:
    def __init__(self, kind: str, labels: dict[str, str]) -> None:
        self.kind = kind
        self._labels = labels

    def ref(self, ref: str | None) -> str:
        return f"{self._labels[ref]} ({ref})" if ref in self._labels else str(ref)


def _valid_decisions(
    candidates: dict[str, tuple[str, ...]], judgement: Judgement, describe: _Describer
) -> tuple[dict[str, str], dict[str, str]]:
    stored: dict[str, str] = {}
    same_as: dict[str, str] = {}
    seen: set[str] = set()
    for verdict in judgement.verdicts:
        item = verdict.item
        if item not in candidates:
            logger.warning("%s judge returned a verdict for unknown item %r; ignored", describe.kind, item)
            continue
        if item in seen:
            logger.warning(
                "%s judge returned a second verdict for %s (%s); ignored",
                describe.kind,
                describe.ref(item),
                verdict.reason,
            )
            continue
        seen.add(item)

        if verdict.decision == "stored" and verdict.match in candidates[item]:
            stored[item] = verdict.match
        elif verdict.decision == "same_as_new" and verdict.match in candidates and verdict.match != item:
            same_as[item] = verdict.match
        elif verdict.decision != "new":
            logger.warning(
                "%s judge: invalid verdict for %s (%s -> %s, reason: %s); treated as new",
                describe.kind,
                describe.ref(item),
                verdict.decision,
                describe.ref(verdict.match),
                verdict.reason,
            )

    for item in (ref for ref in candidates if ref not in seen):
        logger.warning("%s judge returned no verdict for %s; treated as new", describe.kind, describe.ref(item))
    return stored, same_as


def _find(parent: dict[str, str], item: str) -> str:
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item
