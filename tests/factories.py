import pymupdf

from app.llm.schemas import ExtractedCharacter, ExtractedRelation, ExtractionResult

HARRY_DESCRIPTION = "A young wizard who attends Hogwarts School of Witchcraft and Wizardry"
RON_DESCRIPTION = "Red haired boy from a large and poor wizarding family"
HERMIONE_DESCRIPTION = "Brilliant muggle born witch who loves books and rules"


def character(name: str, description: str = "", aliases: tuple[str, ...] = ()) -> ExtractedCharacter:
    return ExtractedCharacter(name=name, aliases=list(aliases), description=description)


def relation(source: str, target: str, type_: str, description: str = "") -> ExtractedRelation:
    return ExtractedRelation(source=source, target=target, type=type_, description=description)


def extraction(
    characters: list[ExtractedCharacter], relations: list[ExtractedRelation] | None = None
) -> ExtractionResult:
    return ExtractionResult(characters=characters, relations=relations or [])


def harry_and_ron() -> ExtractionResult:
    return extraction(
        [
            character("Harry Potter", HARRY_DESCRIPTION, aliases=("Harry", "The Boy Who Lived")),
            character("Ron Weasley", RON_DESCRIPTION, aliases=("Ron",)),
        ],
        [relation("Harry Potter", "Ron Weasley", "FRIEND_OF", "Best friends since the first train ride")],
    )


def solid_pixmap(width: int, height: int, gray: int = 120) -> pymupdf.Pixmap:
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), False)
    pixmap.clear_with(gray)
    return pixmap


def build_pdf(pages: list[dict]) -> bytes:
    """Each page spec may contain: text, image_size (w, h), drawings (number of lines)."""
    document = pymupdf.open()
    for spec in pages:
        page = document.new_page()
        if text := spec.get("text"):
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 400), text, fontsize=11)
        if image_size := spec.get("image_size"):
            page.insert_image(pymupdf.Rect(100, 420, 300, 620), pixmap=solid_pixmap(*image_size))
        for index in range(spec.get("drawings", 0)):
            page.draw_line(pymupdf.Point(50, 650 + index), pymupdf.Point(550, 650 + index))
    data = document.tobytes()
    document.close()
    return data
