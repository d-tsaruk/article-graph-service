import hashlib
import math
import re
from collections.abc import Callable

from app.llm.judging import CharacterCard, RelationTypeCard
from app.llm.schemas import ExtractionResult, ImageAnalysis, Judgement, Verdict

ExtractionFactory = Callable[[str], ExtractionResult]


def hashed_bag_of_words(text: str, dim: int) -> list[float]:
    """Deterministic embedding: texts sharing most tokens end up close in cosine space."""
    vector = [0.0] * dim
    for token in re.findall(r"\w+", text.lower()):
        digest = int(hashlib.sha256(token.encode()).hexdigest(), 16)
        vector[digest % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class FakeLLMClient:
    def __init__(self, embedding_dimensions: int) -> None:
        self.embedding_dimensions = embedding_dimensions
        self.extraction: ExtractionResult | ExtractionFactory = ExtractionResult(characters=[], relations=[])
        self.image_analysis = ImageAnalysis(is_relevant=True, description="An illustration.")
        # Lets a test declare two phrases as synonyms, e.g. {"friends with": "friend of"}.
        self.embedding_overrides: dict[str, str] = {}
        self.extract_calls: list[tuple[str, list[str]]] = []
        self.image_calls: list[tuple[bytes, str, str]] = []
        # Each set holds names (canonical or alias) of ONE individual; cards sharing a set are "the same".
        self.same_characters: list[set[str]] = []
        # (new type, type it maps to), e.g. ("FRIENDS_WITH", "FRIEND_OF").
        self.same_relation_types: set[tuple[str, str]] = set()
        self.judge_error: Exception | None = None
        self.character_judge_calls: list[tuple[list[CharacterCard], list[CharacterCard]]] = []
        self.relation_type_judge_calls: list[tuple[list[RelationTypeCard], list[RelationTypeCard]]] = []

    async def extract(self, text: str, known_relation_types: list[str]) -> ExtractionResult:
        self.extract_calls.append((text, known_relation_types))
        if callable(self.extraction):
            return self.extraction(text)
        return self.extraction

    async def analyze_image(self, image: bytes, mime_type: str, context: str) -> ImageAnalysis:
        self.image_calls.append((image, mime_type, context))
        return self.image_analysis

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            hashed_bag_of_words(self.embedding_overrides.get(text, text), self.embedding_dimensions) for text in texts
        ]

    async def judge_characters(self, new: list[CharacterCard], known: list[CharacterCard]) -> Judgement:
        self.character_judge_calls.append((new, known))
        if self.judge_error:
            raise self.judge_error
        known_by_ref = {card.ref: card for card in known}

        def same(left: CharacterCard, right: CharacterCard) -> bool:
            left_names, right_names = {left.name, *left.aliases}, {right.name, *right.aliases}
            return any(names & left_names and names & right_names for names in self.same_characters)

        verdicts = []
        for index, card in enumerate(new):
            stored = next((ref for ref in card.candidates if same(card, known_by_ref[ref])), None)
            earlier = next((other.ref for other in new[:index] if same(card, other)), None)
            verdicts.append(_verdict(card.ref, stored, earlier))
        return Judgement(verdicts=verdicts)

    async def judge_relation_types(self, new: list[RelationTypeCard], known: list[RelationTypeCard]) -> Judgement:
        self.relation_type_judge_calls.append((new, known))
        if self.judge_error:
            raise self.judge_error
        known_by_ref = {card.ref: card for card in known}

        verdicts = []
        for card in new:
            stored = next(
                (ref for ref in card.candidates if (card.name, known_by_ref[ref].name) in self.same_relation_types),
                None,
            )
            earlier = next(
                (
                    other.ref
                    for other in new
                    if other is not card and (card.name, other.name) in self.same_relation_types
                ),
                None,
            )
            verdicts.append(_verdict(card.ref, stored, earlier))
        return Judgement(verdicts=verdicts)


def _verdict(item: str, stored: str | None, same_as_new: str | None) -> Verdict:
    if stored:
        return Verdict(item=item, reason="fake", decision="stored", match=stored)
    if same_as_new:
        return Verdict(item=item, reason="fake", decision="same_as_new", match=same_as_new)
    return Verdict(item=item, reason="fake", decision="new", match=None)
