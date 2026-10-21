from pathlib import PurePath
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status

from app.api.deps import ServicesDep, require_access_token
from app.pipeline.ingestion import SourceType, UploadedDocument
from app.schemas.api import ArticleUploadResponse, CharacterOut, ErrorResponse, RelationOut

router = APIRouter(prefix="/articles", tags=["articles"], dependencies=[Depends(require_access_token)])

_MARKDOWN_EXTENSIONS = {".md", ".markdown", ".txt"}
_MARKDOWN_CONTENT_TYPES = {"text/markdown", "text/x-markdown", "text/plain"}
_READ_BLOCK = 1024 * 1024


def _detect_source_type(filename: str | None, content_type: str | None, head: bytes) -> SourceType:
    extension = PurePath(filename or "").suffix.lower()
    if head.startswith(b"%PDF") or extension == ".pdf" or content_type == "application/pdf":
        return SourceType.PDF
    if extension in _MARKDOWN_EXTENSIONS or content_type in _MARKDOWN_CONTENT_TYPES:
        return SourceType.MARKDOWN
    raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only Markdown and PDF files are supported")


def _too_large(limit: int) -> HTTPException:
    return HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"Upload exceeds {limit} bytes")


async def _read_limited(file: UploadFile, limit: int) -> bytes:
    data = bytearray()
    while block := await file.read(_READ_BLOCK):
        data.extend(block)
        if len(data) > limit:
            raise _too_large(limit)
    return bytes(data)


@router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
    summary="Upload a Markdown text or a Markdown/PDF file and extract characters and relations",
    responses={
        200: {"model": ArticleUploadResponse, "description": "Identical content was already processed"},
        409: {"model": ErrorResponse, "description": "Identical content is being processed right now"},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"description": "Invalid input or unreadable document"},
        502: {"model": ErrorResponse, "description": "LLM backend failure"},
        503: {"model": ErrorResponse, "description": "LLM backend is not configured"},
    },
)
async def upload_article(
    response: Response,
    services: ServicesDep,
    file: Annotated[UploadFile | None, File(description="Markdown (.md) or PDF file")] = None,
    text: Annotated[str | None, Form(description="Raw Markdown text, alternative to `file`")] = None,
    title: Annotated[str | None, Form(max_length=500)] = None,
) -> ArticleUploadResponse:
    limit = services.settings.max_upload_bytes
    text = text or None
    if (file is None) == (text is None):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Provide exactly one of `file` or `text`")

    if text is not None:
        content = text.encode("utf-8")
        if len(content) > limit:
            raise _too_large(limit)
        upload = UploadedDocument(content=content, source_type=SourceType.MARKDOWN, title=title)
    else:
        content = await _read_limited(file, limit)
        if not content:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Uploaded file is empty")
        source_type = _detect_source_type(file.filename, file.content_type, content[:8])
        upload = UploadedDocument(content=content, source_type=source_type, title=title, filename=file.filename)

    result = await services.ingestion.ingest(upload)
    if result.duplicate:
        response.status_code = status.HTTP_200_OK
    return ArticleUploadResponse(
        article_id=result.article_id,
        title=result.title,
        source_type=result.source_type,
        duplicate=result.duplicate,
        characters=[CharacterOut(**vars(c)) for c in result.characters],
        relations=[RelationOut(**vars(r)) for r in result.relations],
    )
