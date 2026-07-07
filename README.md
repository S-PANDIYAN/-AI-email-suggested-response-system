# Hiver AI Email Evaluation

End-to-end retrieval-augmented generation (RAG) pipeline for customer-support email replies, with an evaluation framework that prioritizes reply quality over surface-level overlap.

The project takes an incoming email, retrieves similar historical cases, generates a grounded response, and scores that response against a reference using a weighted, multi-metric evaluator. It is designed to run in three modes:

- Fully offline with deterministic fallbacks.
- Locally with Ollama.
- With a cloud LLM provider such as Groq, OpenAI, or Gemini.

## What it evaluates

The submission is judged on four points:

- Accuracy and evaluation quality, weighted most heavily.
- Quality and honesty of the dataset.
- Whether the generator is sensible and runs end to end.
- Whether the README clearly explains the approach, trade-offs, and usage.

This repository is built around those criteria.

## Pipeline

```text
incoming email
  -> retrieve similar historical tickets
  -> build a few-shot prompt
  -> generate a grounded reply
  -> evaluate against the gold reference
  -> write CSV, JSON, and text reports
```

## Scoring approach

The final score is a weighted combination of three normalized signals:

- LLM-as-a-Judge: 0.50
- BERTScore: 0.30
- Cosine similarity: 0.20

ROUGE-L is reported for diagnostics but is not part of the final score.

Final score:

```text
final = 100 * (0.50 * judge_norm + 0.30 * bert_f1 + 0.20 * cosine)
```

Why this design:

- The LLM judge is the best proxy for human review, so it carries the most weight.
- BERTScore adds reference-grounded semantic alignment.
- Cosine similarity provides a cheap sanity floor.

This balances correctness, semantic similarity, and robustness against overly literal scoring.

## Dataset

The dataset is synthetic and deterministic. That keeps the repo shippable, reproducible, and free of PII while still producing varied support scenarios across multiple categories such as shipping, refunds, billing, account issues, and technical support.

The generator uses seeded templates, so the same seed always produces the same dataset and the same evaluation split.

## Implementation overview

- [main.py](main.py) orchestrates dataset creation, retrieval, generation, evaluation, and reporting.
- [config/config.yaml](config/config.yaml) stores all tunable settings.
- [dataset/build_dataset.py](dataset/build_dataset.py) creates the synthetic support corpus.
- [embeddings/embedder.py](embeddings/embedder.py) selects Ollama, sentence-transformers, or TF-IDF.
- [retrieval/retriever.py](retrieval/retriever.py) performs self-excluded retrieval over the knowledge base.
- [generator/providers.py](generator/providers.py) adapts Ollama, Groq, OpenAI, Gemini, or mock generation behind one interface.
- [evaluation/](evaluation/) computes cosine, BERTScore, LLM-judge scoring, and final aggregation.
- [evaluation/reporting.py](evaluation/reporting.py) writes the output artifacts.

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run offline

```bash
python main.py --provider mock
```

### 3. Run with Ollama

```bash
ollama pull qwen2.5:7b
ollama pull embeddinggemma:latest
python main.py --provider ollama
```

### 4. Run with a cloud provider

```bash
pip install groq
python main.py
```

If you prefer OpenAI or Gemini, install the corresponding SDK and set the matching API key in `.env`.

### Useful flags

```bash
python main.py --limit 4
python main.py --provider groq
python main.py --rebuild-dataset
```

## Outputs

Each run writes the following files to `outputs/`:

- `generated.csv` - generated replies, references, providers, and retrieved ticket ids.
- `scores.csv` - per-email metric values and final score.
- `results.json` - structured evaluation output and metadata.
- `summary.txt` - a short human-readable summary.

## Trade-offs

- The project prefers evaluation honesty over perfect realism. Synthetic data is easier to ship and verify than private inbox data.
- The weighted score is transparent and easy to inspect, but it is still a hand-tuned heuristic rather than a learned quality model.
- The pipeline degrades gracefully when optional models or API keys are unavailable, which makes it reproducible in restricted environments.

## Requirements

The core pipeline runs with the base dependencies in `requirements.txt`. Optional packages upgrade specific fallbacks:

- `sentence-transformers` for transformer embeddings.
- `bert-score` for real BERTScore.
- `groq`, `openai`, or `google-generativeai` for cloud generation.
- Ollama for local generation and embeddings.

## AI tools used

GitHub Copilot was used to help draft and refine the README structure and wording. The implementation in this repository was then checked against the actual code and runtime outputs so the documentation matches the current behavior.

## Project layout

```text
hiver-ai-email-eval/
├── main.py
├── config/
├── dataset/
├── embeddings/
├── generator/
├── evaluation/
├── retrieval/
├── utils/
└── outputs/
```
