"""Bound the local model to one answer drawn from a packaging record."""

from functools import cached_property
import operator
from pathlib import Path
import re
from typing import Annotated

from langgraph.graph import END, START, StateGraph
from PIL import Image
from typing_extensions import NotRequired, TypedDict

from .models import Flan
from .ranker import load_ranker, score
from .retrieval import Index

FIELDS = {
    "name": r"\b(?:product name|brand name|brand|name)\b",
    "codes": r"\b(?:ndcs?|(?:package|product) (?:code|codes|identifier|identifiers))\b",
    "package": r"\b(?:packaging description|packaging descriptions|package size|pack size|packaging)\b",
}
FIELD_LABELS = {"name": "Product name", "codes": "Package codes", "package": "Packaging descriptions"}


class State(TypedDict):
    question: str
    image: Image.Image | None
    trace: Annotated[list[str], operator.add]
    field: NotRequired[str]
    position: NotRequired[int]
    result: NotRequired[dict]


def _clean(value: str) -> str:
    return " ".join(value.casefold().strip(" .").split())


def _stop(status: str, answer: str) -> dict:
    return {"status": status, "answer": answer, "citations": [], "source_type": None}


class Lookup:
    def __init__(self, index: Index) -> None:
        self.index = index
        self.ranker = load_ranker(index, "mlp")
        graph = StateGraph(State)
        graph.add_node("classify", self._classify)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("ground", self._ground)
        graph.add_edge(START, "classify")
        graph.add_conditional_edges("classify", self._next, {"stop": END, "continue": "retrieve"})
        graph.add_conditional_edges("retrieve", self._next, {"stop": END, "continue": "ground"})
        graph.add_edge("ground", END)
        self.graph = graph.compile()

    @cached_property
    def generator(self) -> Flan:
        return Flan(self.index.work)

    def ask(self, question: str, image: Image.Image | None = None) -> dict:
        state = self.graph.invoke({"question": question.strip(), "image": image, "trace": []})
        return {**state["result"], "trace": state["trace"]}

    @staticmethod
    def _next(state: State) -> str:
        return "stop" if "result" in state else "continue"

    @staticmethod
    def _classify(state: State) -> dict:
        fields = [field for field, pattern in FIELDS.items() if re.search(pattern, state["question"], re.I)]
        update = {"trace": ["classify"]}
        if len(fields) == 1:
            update["field"] = fields[0]
        elif fields:
            update["result"] = _stop("clarify", "Ask for one packaging field at a time.")
        else:
            update["result"] = _stop(
                "abstain", "The available evidence covers product names, package codes, and packaging descriptions.",
            )
        return update

    def _retrieve(self, state: State) -> dict:
        question, image = state["question"], state["image"]
        query = _clean(question)
        code_matches, name_matches = set(), set()
        for i, record in enumerate(self.index.records):
            if any(re.search(rf"(?<![\d-]){re.escape(code)}(?![\d-])", query) for code in record.codes):
                code_matches.add(i)
            if re.search(rf"(?<!\w){re.escape(_clean(record.name))}(?!\w)", query):
                name_matches.add(i)
        supplied = code_matches or name_matches
        update = {"trace": ["retrieve"]}
        if image is None and len(supplied) != 1:
            update["result"] = _stop("clarify", "Provide a package image or an exact name or package code.")
            return update
        if len(code_matches) > 1 or (code_matches and name_matches and not code_matches <= name_matches):
            update["result"] = _stop("clarify", "The question refers to more than one package.")
            return update

        features = self.index.features(question, image)
        ranking = self.index.order(score(self.ranker, features.matrix()))
        if image is None:
            update["position"] = next(iter(supplied))
            return update
        visual = self.index.order(features.visual)
        first, second = visual[:2]
        confident = features.visual[first] >= 0.70 and features.visual[first] - features.visual[second] >= 0.025
        if not confident or (supplied and first not in supplied) or ranking[0] != first:
            update["result"] = _stop("clarify", "I could not identify a single matching package. Provide its code.")
        else:
            update["position"] = first
        return update

    def _ground(self, state: State) -> dict:
        record = self.index.records[state["position"]]
        field = state["field"]
        value = record.evidence(field)
        passage = f"{FIELD_LABELS[field]}: {value}."
        generated = self.generator.answer(state["question"], passage)
        accepted = _clean(generated) == _clean(value)
        return {
            "result": {
                "status": "answer",
                "answer": generated if accepted else value,
                "source_type": "verified_generation" if accepted else "direct_evidence",
                "citations": [{"id": record.id, "url": record.source, "field": field, "value": value}],
            },
            "trace": ["ground", "generate", "verify"],
        }
