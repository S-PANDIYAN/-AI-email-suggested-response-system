"""
Reporting — persist an EvalReport to the four required output artifacts.

    generated.csv  : id, category, customer_email, generated_reply, reference_reply
    scores.csv     : per-email metric columns + final_score (spreadsheet-friendly)
    results.json   : full structured results (per-email + summary + meta)
    summary.txt    : human-readable digest (overall stats + per-category table)

Why four formats? Different consumers:
    * CSV  -> open in Excel/Sheets, sort, pivot.
    * JSON -> programmatic consumption / dashboards / re-analysis.
    * TXT  -> the 10-second "how did we do" glance for a human.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluation.evaluator import EvalReport
from generator.generator import Generation
from utils.io import ensure_dir, write_json, write_text
from utils.logger import get_logger

log = get_logger(__name__)


def write_generated_csv(generations: list[Generation], references: dict[int, str],
                        categories: dict[int, str], path: Path) -> None:
    rows = [{
        "id": g.query_id,
        "category": categories.get(g.query_id, ""),
        "customer_email": g.query_email,
        "generated_reply": g.reply,
        "reference_reply": references.get(g.query_id, ""),
        "provider": g.provider,
        "retrieved_ids": "|".join(str(c.id) for c in g.retrieved),
    } for g in generations]
    ensure_dir(path.parent)
    pd.DataFrame(rows).to_csv(path, index=False)
    log.info("Wrote %s", path)


def write_scores_csv(report: EvalReport, path: Path) -> None:
    rows = []
    for r in report.per_email:
        row = {k: v for k, v in r.items()
               if k not in ("judge_dimensions", "judge_rationale")}
        row.update({f"judge_{d}": s for d, s in r["judge_dimensions"].items()})
        rows.append(row)
    ensure_dir(path.parent)
    pd.DataFrame(rows).to_csv(path, index=False)
    log.info("Wrote %s", path)


def write_results_json(report: EvalReport, path: Path) -> None:
    write_json({"meta": report.meta, "summary": report.summary,
                "per_email": report.per_email}, path)
    log.info("Wrote %s", path)


def write_summary_txt(report: EvalReport, path: Path) -> None:
    s = report.summary
    o = s["overall"]
    cm = s["component_means"]
    lines = [
        "=" * 66,
        " HIVER AI EMAIL EVAL — SUMMARY",
        "=" * 66,
        f" Emails evaluated : {report.meta['n_evaluated']}",
        f" Embedder         : {report.meta['embedder']}",
        f" Judge backend    : {report.meta['judge_backend']}",
        f" BERTScore backend: {report.meta['bertscore_backend']}",
        f" Weights          : {report.meta['weights']}",
        "-" * 66,
        " OVERALL FINAL SCORE (0-100)",
        f"   mean={o['mean']}  median={o['median']}  min={o['min']}  "
        f"max={o['max']}  stdev={o['stdev']}",
        "-" * 66,
        " COMPONENT MEANS (0-1)",
        f"   LLM judge (norm) : {cm['judge_normalized']}",
        f"   BERTScore F1     : {cm['bertscore_f1']}",
        f"   Cosine           : {cm['cosine']}",
        f"   ROUGE-L F1 (diag): {cm['rouge_l_f1']}",
        "-" * 66,
        " PER-CATEGORY FINAL SCORE (mean / count)",
    ]
    for cat, st in s["per_category"].items():
        lines.append(f"   {cat:<20} {st['mean']:>6}   (n={st['count']})")
    lines.append("=" * 66)
    write_text("\n".join(lines) + "\n", path)
    log.info("Wrote %s", path)


def write_all(report: EvalReport, generations: list[Generation],
              references: dict[int, str], categories: dict[int, str],
              outputs_dir: Path) -> None:
    ensure_dir(outputs_dir)
    write_generated_csv(generations, references, categories, outputs_dir / "generated.csv")
    write_scores_csv(report, outputs_dir / "scores.csv")
    write_results_json(report, outputs_dir / "results.json")
    write_summary_txt(report, outputs_dir / "summary.txt")
