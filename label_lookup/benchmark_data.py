"""Deterministic benchmark questions using real package label records."""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance

from .corpus import Record
from .images import load_image

QUESTIONS = {
    "name": "What product name is listed on this label?",
    "codes": "Which package codes are listed for this product?",
    "package": "What packaging descriptions are listed for this product?",
}


@dataclass(frozen=True)
class Case:
    key: str
    kind: str
    question: str
    field: str | None
    record: Record
    image: bool = False
    expected: str = "answer"


def answerable(record: Record) -> list[Case]:
    return [
        Case(
            record.id + ":text", "text",
            f"Which package codes are listed for {record.name} ({record.codes[0]})?",
            "codes", record,
        ),
        Case(record.id + ":image", "image", QUESTIONS["name"], "name", record, image=True),
        Case(record.id + ":multimodal", "multimodal", QUESTIONS["package"], "package", record, image=True),
    ]


def benchmark_cases(records: list[Record]) -> list[Case]:
    cases = [case for record in records for case in answerable(record)]
    for record in records:
        cases.extend((
            Case(
                record.id + ":unsupported", "unsupported", "Does this package have a recycling symbol?",
                None, record, image=True, expected="abstain",
            ),
            Case(
                record.id + ":ambiguous", "ambiguous", QUESTIONS["codes"],
                "codes", record, expected="clarify",
            ),
        ))
    return cases


def query_image(case: Case, work: Path) -> Image.Image | None:
    if not case.image:
        return None
    source = load_image(work / case.record.image)
    width, height = source.size
    border_x, border_y = max(1, width // 16), max(1, height // 16)
    cropped = source.crop((border_x, border_y, width - border_x, height - border_y))
    cropped.thumbnail((448, 448))
    return ImageEnhance.Brightness(cropped).enhance(0.90)
