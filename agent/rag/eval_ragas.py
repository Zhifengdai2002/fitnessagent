"""RAGAS four-metric RAG evaluation.

Metrics:
  - Context Precision  : of retrieved chunks, what fraction are relevant?
  - Context Recall     : of info needed to answer, what fraction was retrieved?
  - Faithfulness       : does the answer stay within the retrieved context?
  - Answer Relevance   : does the answer actually address the question?

Usage:
    python -m agent.rag.eval_ragas
    python -m agent.rag.eval_ragas --samples 10   # run first N samples only
    python -m agent.rag.eval_ragas --no-answer     # skip answer generation (context metrics only)
"""

from __future__ import annotations

import argparse
import json
import warnings
from typing import Any

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Evaluation dataset — query + ground_truth reference answers
# ---------------------------------------------------------------------------

# Queries are scoped to content already in the corpus:
# WHO (physical activity + healthy diet), Fitness Wiki (getting started,
# weight loss, muscle building), Dietary Guidelines PDF, PubMed abstracts
# (resistance training, protein, sleep, injury prevention, periodization,
# hypertrophy volume, progressive overload).
# ExRx exercise-technique queries are excluded until those pages are ingested.

EVAL_SET: list[dict[str, Any]] = [
    # --- WHO / Physical activity guidelines ---
    {
        "query": "how much physical activity do adults need per week",
        "topic": "training", "goal": "health", "level": "",
        "ground_truth": (
            "WHO recommends adults do at least 150-300 minutes of moderate-intensity "
            "aerobic activity per week, or 75-150 minutes of vigorous-intensity activity, "
            "plus muscle-strengthening activities on 2 or more days per week."
        ),
    },
    {
        "query": "benefits of regular physical activity for health",
        "topic": "training", "goal": "health", "level": "",
        "ground_truth": (
            "Regular physical activity reduces the risk of cardiovascular disease, type 2 "
            "diabetes, and several cancers. It also improves mental health, bone density, "
            "and overall quality of life."
        ),
    },
    # --- WHO / Healthy diet ---
    {
        "query": "healthy diet guidelines macronutrients",
        "topic": "nutrition", "goal": "health", "level": "",
        "ground_truth": (
            "A healthy diet includes a balance of carbohydrates, proteins, and fats. "
            "WHO recommends limiting free sugars to less than 10% of total energy intake "
            "and keeping saturated fats below 10% of total energy."
        ),
    },
    {
        "query": "how much sugar and fat should I eat daily",
        "topic": "nutrition", "goal": "health", "level": "",
        "ground_truth": (
            "WHO recommends limiting free sugars to less than 10% of total energy intake "
            "and reducing saturated fat to less than 10%. Trans fats should be eliminated "
            "from the diet entirely."
        ),
    },
    # --- Dietary Guidelines ---
    {
        "query": "daily protein recommendation for adults",
        "topic": "nutrition", "goal": "health", "level": "",
        "ground_truth": (
            "The Dietary Guidelines for Americans recommend adults consume 0.8 grams of "
            "protein per kilogram of bodyweight per day as a minimum. Higher intakes "
            "support muscle maintenance, especially in older adults."
        ),
    },
    {
        "query": "recommended daily calorie intake for adults",
        "topic": "nutrition", "goal": "health", "level": "",
        "ground_truth": (
            "Calorie needs vary by age, sex, and activity level. The Dietary Guidelines "
            "for Americans suggest 1600-2400 kcal per day for adult women and 2000-3000 "
            "kcal per day for adult men depending on activity level."
        ),
    },
    # --- Fitness Wiki / Muscle building ---
    {
        "query": "how to build muscle for beginners",
        "topic": "training", "goal": "hypertrophy", "level": "beginner",
        "ground_truth": (
            "Beginners should focus on progressive overload with compound movements, "
            "training each muscle group 2-3 times per week. Adequate protein intake "
            "and consistent sleep are essential for muscle growth."
        ),
    },
    {
        "query": "progressive overload for muscle growth",
        "topic": "training", "goal": "hypertrophy", "level": "",
        "ground_truth": (
            "Progressive overload means gradually increasing training stress by adding "
            "weight, reps, or sets over time. It is the primary driver of muscle "
            "hypertrophy and strength adaptation."
        ),
    },
    # --- Fitness Wiki / Weight loss ---
    {
        "query": "calorie deficit for weight loss",
        "topic": "nutrition", "goal": "fat_loss", "level": "",
        "ground_truth": (
            "Weight loss requires consuming fewer calories than you expend. A deficit "
            "of 300-500 kcal per day leads to gradual, sustainable fat loss of "
            "approximately 0.25-0.5 kg per week."
        ),
    },
    {
        "query": "how to lose weight with diet and exercise",
        "topic": "nutrition", "goal": "fat_loss", "level": "",
        "ground_truth": (
            "Effective weight loss combines a moderate caloric deficit with regular "
            "exercise. Resistance training helps preserve muscle mass during a deficit, "
            "and protein intake should be kept high."
        ),
    },
    # --- PubMed / Resistance training ---
    {
        "query": "how many times per week should I train each muscle",
        "topic": "training", "goal": "hypertrophy", "level": "",
        "ground_truth": (
            "Research suggests training each muscle group 2 times per week produces "
            "superior hypertrophy compared to once per week. Frequency allows more "
            "total weekly volume distributed across sessions."
        ),
    },
    {
        "query": "resistance training benefits for health",
        "topic": "training", "goal": "health", "level": "",
        "ground_truth": (
            "Resistance training improves muscular strength, bone density, metabolic "
            "rate, and insulin sensitivity. It reduces risk of sarcopenia, osteoporosis, "
            "and cardiovascular disease."
        ),
    },
    # --- PubMed / Protein + nutrition ---
    {
        "query": "protein intake for muscle hypertrophy research",
        "topic": "nutrition", "goal": "hypertrophy", "level": "",
        "ground_truth": (
            "Meta-analyses show 1.6 g/kg/day of protein maximizes muscle protein "
            "synthesis. Intakes beyond 2.2 g/kg/day show diminishing returns. "
            "Protein distribution across meals also matters."
        ),
    },
    # --- PubMed / Sleep and recovery ---
    {
        "query": "sleep and exercise recovery performance",
        "topic": "recovery", "goal": "", "level": "",
        "ground_truth": (
            "Sleep deprivation impairs strength, reaction time, and recovery. "
            "Adults need 7-9 hours per night. Poor sleep elevates cortisol and "
            "reduces growth hormone secretion, hindering muscle repair."
        ),
    },
    # --- PubMed / Injury prevention ---
    {
        "query": "injury prevention in strength training",
        "topic": "injury", "goal": "", "level": "",
        "ground_truth": (
            "Injury prevention in resistance training requires progressive loading, "
            "proper technique, adequate warm-up, and sufficient recovery. "
            "Avoiding rapid load increases reduces overuse injury risk."
        ),
    },
]


# ---------------------------------------------------------------------------
# Build answer with the project's own LLM
# ---------------------------------------------------------------------------

def _generate_answer(query: str, contexts: list[str]) -> str:
    from agent.llm import call_model_text
    context_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    return call_model_text(
        system_prompt=(
            "You are a professional fitness coach. Answer the user's question "
            "using ONLY the provided reference excerpts. Be concise (2-4 sentences)."
        ),
        user_prompt=f"Reference excerpts:\n{context_text}\n\nQuestion: {query}",
        temperature=0.1,
        max_tokens=200,
    ).strip()


# ---------------------------------------------------------------------------
# Build RAGAS dataset
# ---------------------------------------------------------------------------

def _build_samples(entries: list[dict[str, Any]], generate_answers: bool) -> list[Any]:
    from ragas import SingleTurnSample
    from agent.rag.retriever import retrieve_knowledge

    samples = []
    for i, entry in enumerate(entries):
        print(f"  [{i+1}/{len(entries)}] {entry['query'][:55]}...")
        results = retrieve_knowledge(
            query=entry["query"],
            topic=entry.get("topic") or None,
            goal=entry.get("goal") or None,
            level=entry.get("level") or None,
            limit=4,
        )
        contexts = [r.get("text", "") for r in results if r.get("text")]

        if not contexts:
            print(f"    WARNING: no contexts retrieved, skipping")
            continue

        if generate_answers:
            answer = _generate_answer(entry["query"], contexts)
        else:
            answer = "N/A"

        samples.append(SingleTurnSample(
            user_input=entry["query"],
            retrieved_contexts=contexts,
            response=answer,
            reference=entry["ground_truth"],
        ))
    return samples


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

def run_ragas_eval(*, n_samples: int | None = None, generate_answers: bool = True) -> None:
    from ragas import evaluate, EvaluationDataset
    import warnings
    warnings.filterwarnings("ignore")

    from openai import OpenAI
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings as LCEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
    from agent.config import load_settings

    settings = load_settings()

    lc_llm = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.model_api_key,
        base_url=settings.model_base_url,
        temperature=0.1,
    )
    ragas_llm = LangchainLLMWrapper(lc_llm)

    lc_emb = LCEmbeddings(
        model=settings.embedding_model_name,
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
    )
    ragas_emb = LangchainEmbeddingsWrapper(lc_emb)

    # inject LLM/embeddings into metric instances
    context_precision.llm = ragas_llm
    context_recall.llm = ragas_llm
    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_emb

    entries = EVAL_SET[:n_samples] if n_samples else EVAL_SET
    print(f"\nBuilding evaluation dataset ({len(entries)} samples)...")
    samples = _build_samples(entries, generate_answers=generate_answers)

    if not samples:
        print("No samples built — check retriever configuration.")
        return

    dataset = EvaluationDataset(samples=samples)

    metrics: list[Any] = [context_precision, context_recall]
    if generate_answers:
        metrics += [faithfulness, answer_relevancy]

    from ragas.run_config import RunConfig
    run_config = RunConfig(timeout=120, max_workers=2, max_retries=3)

    print(f"\nRunning RAGAS evaluation on {len(samples)} samples...")
    result = evaluate(dataset=dataset, metrics=metrics, run_config=run_config)

    print(f"\n{'─'*60}")
    print("RAGAS Results:")
    print(f"{'─'*60}")
    scores = result.to_pandas() if hasattr(result, "to_pandas") else {}
    for metric_name, score in result.items() if hasattr(result, "items") else []:
        print(f"  {metric_name:<35} {score:.4f}")

    print(f"\n{result}")

    # Save to JSON
    output = {
        "n_samples": len(samples),
        "generate_answers": generate_answers,
        "scores": {k: float(v) for k, v in result.items()} if hasattr(result, "items") else {},
    }
    out_path = "/Users/zhifengdai/FitnessAgent/data/knowledge/ragas_eval_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAGAS four-metric RAG evaluation.")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to evaluate (default: all 15)")
    parser.add_argument("--no-answer", action="store_true", help="Skip answer generation; run context metrics only")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_ragas_eval(
        n_samples=args.samples,
        generate_answers=not args.no_answer,
    )
