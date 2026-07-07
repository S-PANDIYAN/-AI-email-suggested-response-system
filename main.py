"""
main.py — end-to-end entry point.

Pipeline:
    1. Ensure dataset exists (generate if missing).
    2. Split into knowledge base (retrieval index) + eval holdout (test set).
    3. Build embedder -> retriever (index the KB).
    4. Build LLM provider -> generator.
    5. Generate a grounded reply for every holdout email.
    6. Evaluate generated vs. reference with all metrics -> weighted score.
    7. Write outputs (generated.csv, scores.csv, results.json, summary.txt).

WHY a KB / holdout split
------------------------
If we evaluated on emails that are IN the retrieval index, the model could
lean on near-duplicates (or, without self-exclusion, the answer itself). We
hold out a slice to simulate genuinely NEW incoming emails. The KB (the other
rows) is the "history" the system retrieves from. This is standard train/test
hygiene applied to RAG.

Run:
    python main.py                 # full pipeline with config defaults
    python main.py --limit 8       # quick smoke run on 8 emails
    python main.py --provider mock # force offline generation
"""
from __future__ import annotations

import argparse

import pandas as pd

from config.settings import load_settings
from dataset.build_dataset import build_dataset
from embeddings.embedder import build_embedder
from evaluation.evaluator import Evaluator
from evaluation.metrics.llm_judge import LLMJudge
from evaluation.reporting import write_all
from generator.generator import ReplyGenerator
from generator.providers import build_provider
from retrieval.retriever import Retriever
from utils.io import ensure_dir
from utils.logger import configure_logging, get_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hiver AI Email Eval pipeline")
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--limit", type=int, default=None,
                   help="evaluate only the first N holdout emails (smoke test)")
    p.add_argument("--provider", default=None,
                   help="override generation.provider (ollama|groq|mock|...)")
    p.add_argument("--rebuild-dataset", action="store_true",
                   help="regenerate the dataset even if it exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    configure_logging(settings.logging["level"])
    log = get_logger("main")
    if args.provider:                       # CLI override wins over yaml
        settings.generation["provider"] = args.provider

    # 1. dataset ----------------------------------------------------------
    csv_path = settings.paths["dataset_csv"]
    if args.rebuild_dataset or not csv_path.exists():
        df = build_dataset(settings.dataset["n_pairs"], settings.seed)
        ensure_dir(csv_path.parent)
        df.to_csv(csv_path, index=False)
        log.info("Generated dataset -> %s (%d rows)", csv_path, len(df))
    else:
        df = pd.read_csv(csv_path)
        log.info("Loaded dataset <- %s (%d rows)", csv_path, len(df))

    # 2. KB / holdout split ----------------------------------------------
    holdout_n = settings.dataset["eval_holdout"]
    if args.limit:
        holdout_n = min(holdout_n, args.limit)
    # Deterministic, stratified-ish holdout: sample per category for coverage.
    holdout = (df.groupby("category", group_keys=False)
                 .apply(lambda g: g.sample(min(2, len(g)), random_state=settings.seed))
                 .head(holdout_n)
                 .reset_index(drop=True))
    kb = df[~df["id"].isin(holdout["id"])].reset_index(drop=True)
    log.info("Split: %d knowledge-base rows, %d holdout rows", len(kb), len(holdout))

    # 3. embedder + retriever (index the KB only) ------------------------
    corpus = kb["customer_email"].tolist() + kb["support_reply"].tolist()
    embedder = build_embedder(settings, corpus)
    retriever = Retriever(kb, embedder, settings.retrieval["exclude_self"])

    # 4. provider + generator --------------------------------------------
    provider = build_provider(settings)
    generator = ReplyGenerator(retriever, provider, settings)

    # 5. generate replies for the holdout --------------------------------
    generations = []
    for _, row in holdout.iterrows():
        gen = generator.generate(row["customer_email"], query_id=int(row["id"]))
        generations.append(gen)
    log.info("Generated %d replies via %s", len(generations), provider.name)

    # 6. evaluate ---------------------------------------------------------
    judge = LLMJudge(provider, settings, embedder)
    evaluator = Evaluator(embedder, judge, settings)
    references = dict(zip(holdout["id"], holdout["support_reply"]))
    categories = dict(zip(holdout["id"], holdout["category"]))
    records = [{
        "id": int(g.query_id),
        "category": categories[g.query_id],
        "customer_email": g.query_email,
        "generated": g.reply,
        "reference": references[g.query_id],
    } for g in generations]
    report = evaluator.evaluate(records)

    # 7. write outputs ----------------------------------------------------
    write_all(report, generations, references, categories, settings.paths["outputs_dir"])

    o = report.summary["overall"]
    log.info("DONE. Overall final score: mean=%.1f median=%.1f (n=%d)",
             o["mean"], o["median"], o["count"])
    print(f"\n  Overall score: {o['mean']}/100  "
          f"(median {o['median']}, min {o['min']}, max {o['max']})")
    print(f"  See {settings.paths['outputs_dir']} for full reports.\n")


if __name__ == "__main__":
    main()
