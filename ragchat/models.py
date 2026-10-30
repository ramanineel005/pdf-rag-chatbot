from __future__ import annotations

from dataclasses import dataclass, field


class UserFacingError(Exception):
    """Error whose message is safe and useful to show in the UI."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


class PDFError(UserFacingError):
    pass


class EmbeddingModelUnavailable(UserFacingError):
    pass


class LLMUnavailable(UserFacingError):
    pass


@dataclass
class PageText:
    page_number: int  # 1-based, matches the PDF viewer
    text: str
    ocr: bool = False


@dataclass
class ExtractedDocument:
    doc_id: str
    filename: str
    page_count: int
    pages: list[PageText]  # only pages that contain text

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    filename: str
    page: int
    index: int
    text: str


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    filename: str
    page: int
    text: str
    similarity: float


@dataclass
class Source:
    label: str  # "S1"
    filename: str
    page: int
    excerpt: str
    similarity: float
    cited: bool


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    grounded: bool = True
    citations_verified: bool = True  # False: model cited nothing; sources are just what was retrieved
