import hashlib
import math
import re
from collections.abc import Callable

from app.llm.schemas import ExtractionResult, ImageAnalysis

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
