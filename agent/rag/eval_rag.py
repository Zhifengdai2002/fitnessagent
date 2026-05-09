"""Offline RAG recall evaluation.

Runs a labeled query set against retrieve_knowledge() and reports Recall@k.

Usage:
    python -m agent.rag.eval_rag            # default k=4
    python -m agent.rag.eval_rag --k 6
    python -m agent.rag.eval_rag --k 4 --verbose
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Labeled evaluation set
# Each entry: query + topic/goal/level context + expected keywords (at least
# one chunk in top-k must contain ALL keywords in any single keyword group).
# ---------------------------------------------------------------------------

EVAL_SET: list[dict[str, Any]] = [
    # --- Training / technique ---
    {
        "query": "how to squat with proper form",
        "topic": "training",
        "goal": "strength",
        "level": "beginner",
        "keyword_groups": [["squat", "knee"], ["squat", "back"], ["squat", "depth"]],
    },
    {
        "query": "barbell deadlift technique and common mistakes",
        "topic": "training",
        "goal": "strength",
        "level": "",
        "keyword_groups": [["deadlift", "back"], ["deadlift", "hip"], ["deadlift", "form"]],
    },
    {
        "query": "bench press form chest exercise",
        "topic": "training",
        "goal": "hypertrophy",
        "level": "",
        "keyword_groups": [["bench", "chest"], ["bench", "press"], ["pectoral"]],
    },
    {
        "query": "pull-up technique lat muscles",
        "topic": "training",
        "goal": "strength",
        "level": "",
        "keyword_groups": [["pull", "lat"], ["pullup", "back"], ["chin"]],
    },
    {
        "query": "overhead press shoulder exercise",
        "topic": "training",
        "goal": "strength",
        "level": "",
        "keyword_groups": [["press", "shoulder"], ["overhead", "deltoid"], ["military press"]],
    },
    {
        "query": "progressive overload for strength gains",
        "topic": "training",
        "goal": "strength",
        "level": "",
        "keyword_groups": [["progressive", "overload"], ["progressive", "load"], ["adaptation", "training"]],
    },
    {
        "query": "how many sets and reps for muscle hypertrophy",
        "topic": "training",
        "goal": "hypertrophy",
        "level": "",
        "keyword_groups": [["sets", "rep"], ["volume", "hypertrophy"], ["sets", "muscle"]],
    },
    {
        "query": "training program design periodization",
        "topic": "training",
        "goal": "strength",
        "level": "",
        "keyword_groups": [["periodization"], ["program", "design"], ["training", "cycle"]],
    },
    {
        "query": "beginner weight training guidelines",
        "topic": "training",
        "goal": "health",
        "level": "beginner",
        "keyword_groups": [["beginner", "training"], ["resistance", "guideline"], ["beginner", "strength"]],
    },
    # --- Nutrition ---
    {
        "query": "how much protein do I need to build muscle",
        "topic": "nutrition",
        "goal": "hypertrophy",
        "level": "",
        "keyword_groups": [["protein", "muscle"], ["protein", "g/kg"], ["protein", "intake"]],
    },
    {
        "query": "calorie deficit for weight loss",
        "topic": "nutrition",
        "goal": "fat_loss",
        "level": "",
        "keyword_groups": [["calorie", "deficit"], ["energy", "balance"], ["weight", "loss"]],
    },
    {
        "query": "healthy diet guidelines macronutrients",
        "topic": "nutrition",
        "goal": "health",
        "level": "",
        "keyword_groups": [["diet", "guideline"], ["macronutrient"], ["carbohydrate", "protein", "fat"]],
    },
    # --- Recovery ---
    {
        "query": "how to recover from muscle soreness after workout",
        "topic": "recovery",
        "goal": "",
        "level": "",
        "keyword_groups": [["recovery", "muscle"], ["soreness", "doms"], ["rest", "recovery"]],
    },
    {
        "query": "sleep and exercise performance recovery",
        "topic": "recovery",
        "goal": "",
        "level": "",
        "keyword_groups": [["sleep", "recovery"], ["sleep", "performance"], ["rest", "exercise"]],
    },
    {
        "query": "warm up before lifting weights",
        "topic": "recovery",
        "goal": "",
        "level": "",
        "keyword_groups": [["warm", "up"], ["warmup", "exercise"], ["warm", "lift"]],
    },
    {
        "query": "signs of overtraining and how to avoid it",
        "topic": "recovery",
        "goal": "",
        "level": "",
        "keyword_groups": [["overtraining"], ["fatigue", "training"], ["overtrain", "recovery"]],
    },
    # --- Injury ---
    {
        "query": "knee pain squatting injury prevention",
        "topic": "training",
        "goal": "",
        "level": "",
        "keyword_groups": [["knee", "squat"], ["knee", "injury"], ["knee", "pain"]],
    },
    {
        "query": "exercise injury prevention strength training safety",
        "topic": "injury",
        "goal": "",
        "level": "",
        "keyword_groups": [["injury", "prevention"], ["safety", "exercise"], ["injury", "training"]],
    },
    # --- Cardio ---
    {
        "query": "cardio for fat loss how much",
        "topic": "training",
        "goal": "fat_loss",
        "level": "",
        "keyword_groups": [["cardio", "fat"], ["cardio", "loss"], ["aerobic", "exercise"]],
    },
    {
        "query": "physical activity guidelines adults weekly exercise",
        "topic": "training",
        "goal": "health",
        "level": "",
        "keyword_groups": [["physical activity", "guideline"], ["150", "minute"], ["exercise", "week"]],
    },
]


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    query: str
    hit: bool
    matched_group: list[str] = field(default_factory=list)
    top_titles: list[str] = field(default_factory=list)
    top_sources: list[str] = field(default_factory=list)


def _chunk_matches_group(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    return all(kw.lower() in text_lower for kw in keywords)


def _evaluate_query(entry: dict[str, Any], k: int) -> QueryResult:
    from agent.rag.retriever import retrieve_knowledge

    results = retrieve_knowledge(
        query=entry["query"],
        topic=entry.get("topic") or None,
        goal=entry.get("goal") or None,
        level=entry.get("level") or None,
        limit=k,
    )

    top_titles = [r.get("title", "")[:60] for r in results]
    top_sources = [r.get("source", "") for r in results]
    keyword_groups: list[list[str]] = entry.get("keyword_groups", [])

    for result in results:
        text = (result.get("text") or "") + " " + (result.get("title") or "")
        for group in keyword_groups:
            if _chunk_matches_group(text, group):
                return QueryResult(
                    query=entry["query"],
                    hit=True,
                    matched_group=group,
                    top_titles=top_titles,
                    top_sources=top_sources,
                )

    return QueryResult(
        query=entry["query"],
        hit=False,
        top_titles=top_titles,
        top_sources=top_sources,
    )


def run_eval(k: int = 4, verbose: bool = False) -> None:
    results: list[QueryResult] = []
    hits = 0

    print(f"\nRAG Recall Evaluation  (k={k}, {len(EVAL_SET)} queries)\n{'─' * 60}")

    for entry in EVAL_SET:
        result = _evaluate_query(entry, k)
        results.append(result)
        status = "✓" if result.hit else "✗"
        if result.hit:
            hits += 1

        if verbose or not result.hit:
            print(f"\n{status} {result.query!r}")
            if result.hit:
                print(f"   matched: {result.matched_group}")
            else:
                print(f"   MISS — top {k} results:")
                for title, src in zip(result.top_titles, result.top_sources):
                    print(f"     [{src}] {title}")

    recall = hits / len(EVAL_SET) * 100
    print(f"\n{'─' * 60}")
    print(f"Recall@{k}: {hits}/{len(EVAL_SET)} = {recall:.1f}%")

    # Per-topic breakdown
    topic_counts: dict[str, list[bool]] = {}
    for entry, result in zip(EVAL_SET, results):
        topic = entry.get("topic") or "other"
        topic_counts.setdefault(topic, []).append(result.hit)

    print("\nPer-topic breakdown:")
    for topic, hits_list in sorted(topic_counts.items()):
        t_hits = sum(hits_list)
        print(f"  {topic:<12} {t_hits}/{len(hits_list)} = {t_hits/len(hits_list)*100:.0f}%")

    if recall < 60:
        print("\n[!] Recall below 60% — consider expanding knowledge sources or tuning chunk_size.")
    elif recall < 80:
        print("\n[~] Recall moderate — knowledge base coverage could be improved.")
    else:
        print("\n[✓] Recall looks healthy.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG recall offline.")
    parser.add_argument("--k", type=int, default=4, help="Top-k chunks to retrieve per query")
    parser.add_argument("--verbose", action="store_true", help="Print details for all queries, not just misses")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_eval(k=args.k, verbose=args.verbose)
