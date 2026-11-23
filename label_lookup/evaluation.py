"""Measure ranking, answer grounding, and routing on held-out product families."""

from collections import Counter, defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .answering import Lookup
from .benchmark_data import benchmark_cases, query_image
from .models import REVISIONS
from .ranker import load_ranker, score
from .retrieval import Index


def evaluate(index: Index, split: str) -> dict:
    families = set(index.splits[split])
    records = [record for record in index.records if record.family in families]
    linear, mlp = (load_ranker(index, kind) for kind in ("linear", "mlp"))
    lookup = Lookup(index)
    retrieval = defaultdict(lambda: defaultdict(list))
    timings = []
    counts = Counter()
    failures, rows = [], []
    for case in benchmark_cases(records):
        image = query_image(case, index.work)
        row = {"id": case.key, "kind": case.kind, "question": case.question, "expected": case.expected}
        if case.expected == "answer":
            both = index.features(case.question, image)
            text = index.features(case.question) if image is not None else both
            position = index.positions[case.record.id]
            methods = {
                "bm25": index.order(text.lexical),
                "dense_text": index.order(text.dense),
                "rrf": index.fused(both),
                "linear": index.order(score(linear, both.matrix())),
                "mlp_text": index.order(score(mlp, text.matrix())),
                "mlp": index.order(score(mlp, both.matrix())),
            }
            row["ranks"] = {}
            for name, ranking in methods.items():
                rank = ranking.index(position) + 1
                row["ranks"][name] = rank
                retrieval["all"][name].append(rank)
                retrieval[case.kind][name].append(rank)
        started = perf_counter()
        result = lookup.ask(case.question, image)
        elapsed = (perf_counter() - started) * 1000
        timings.append(elapsed)
        counts["route_correct"] += int(result["status"] == case.expected)
        row.update({
            "status": result["status"], "trace": result["trace"],
            "source_type": result["source_type"], "inference_ms": round(elapsed, 2),
        })
        if case.expected == "answer":
            counts["answerable"] += 1
            correct = bool(result["citations"] and result["citations"][0]["id"] == case.record.id)
            counts["correct_grounded"] += int(correct)
            counts["answered"] += int(result["status"] == "answer")
            counts["verified_generation"] += int(result["source_type"] == "verified_generation")
            if not correct:
                failures.append({**row, "got": (
                    result["citations"][0]["id"] if result["citations"] else result["status"]
                )})
            elif row["ranks"]["bm25"] > 3 or row["ranks"]["mlp"] > 1:
                failures.append({**row, "got": "correct answer; retrieval methods differed"})
        else:
            counts[case.expected] += 1
            counts[f"{case.expected}_correct"] += int(result["status"] == case.expected)
            if result["status"] != case.expected:
                failures.append({**row, "got": result["status"]})
        rows.append(row)
    summary = {
        "split": split, "records": len(records), "families": len(families),
        "corpus_fingerprint": index.fingerprint, "model_revisions": REVISIONS,
        "retrieval": {
            query_type: {name: _metrics(ranks) for name, ranks in methods.items()}
            for query_type, methods in retrieval.items()
        },
        "answering": {
            **counts, "total": len(rows), "latency_p50_ms": round(float(np.median(timings)), 2),
            "latency_p95_ms": round(float(np.percentile(timings, 95)), 2),
        },
    }
    _write(index.work, split, {**summary, "cases": rows, "failures_and_disagreements": failures})
    return summary


def _metrics(ranks: list[int]) -> dict:
    return {
        "n": len(ranks), "recall_at_1": round(sum(rank == 1 for rank in ranks) / len(ranks), 4),
        "recall_at_3": round(sum(rank <= 3 for rank in ranks) / len(ranks), 4),
        "mrr": round(sum(1 / rank for rank in ranks) / len(ranks), 4),
    }


def _write(work: Path, split: str, report: dict) -> None:
    (work / f"evaluation_{split}.json").write_text(json.dumps(report, indent=2) + "\n")
