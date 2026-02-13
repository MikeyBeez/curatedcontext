#!/usr/bin/env python3
"""
Context Curation Experiment
Compares raw (noisy) context vs curated context for QA using Ollama.

Two conditions per question:
  BASELINE: noisy context (relevant + irrelevant paragraphs) -> answer model
  CURATED:  noisy context -> curator model extracts relevant text -> answer model

Scores: exact match and F1 token overlap (standard SQuAD metrics).
"""

import json
import random
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

try:
    import requests
except ImportError:
    # Fall back to urllib if requests not available
    import urllib.request
    import urllib.error

    class _Requests:
        """Minimal requests-like wrapper around urllib."""

        class _Response:
            def __init__(self, resp):
                self.status_code = resp.status
                self._body = resp.read().decode("utf-8")

            def json(self):
                return json.loads(self._body)

            @property
            def text(self):
                return self._body

        def post(self, url, json=None, timeout=None):
            data = json_.dumps(json).encode("utf-8") if json else None
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}
            )
            try:
                resp = urllib.request.urlopen(req, timeout=timeout)
                return self._Response(resp)
            except urllib.error.HTTPError as e:
                r = self._Response(e)
                r.status_code = e.code
                return r

    json_ = json
    requests = _Requests()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
SQUAD_PATH = Path(__file__).parent / "dev-v2.0.json"
OUTPUT_PATH = Path(__file__).parent / "results.jsonl"
SUMMARY_PATH = Path(__file__).parent / "summary.json"

# Models — using llama3.1:8b for both by default
ANSWER_MODEL = "llama3.1:8b"
CURATOR_MODEL = "qwen2.5:latest"  # smaller/faster model for curation

NUM_QUESTIONS = 200       # how many questions to run
NUM_DISTRACTORS = 6       # irrelevant paragraphs mixed in (4-9 range)
SEED = 42

# Ollama generation parameters
OLLAMA_OPTS = {
    "temperature": 0.0,
    "num_predict": 256,
    "top_p": 1.0,
}


# ---------------------------------------------------------------------------
# SQuAD metrics (standard implementations)
# ---------------------------------------------------------------------------

def normalize_answer(s):
    """Lower text, remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens) if pred_tokens else 0
    recall = num_same / len(gt_tokens) if gt_tokens else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def best_score(prediction, ground_truths, metric_fn):
    """Take the max score across all accepted ground truth answers."""
    return max(metric_fn(prediction, gt) for gt in ground_truths)


# ---------------------------------------------------------------------------
# Ollama API
# ---------------------------------------------------------------------------

def call_ollama(model, prompt, timeout=120):
    """Call Ollama generate API. Returns (response_text, elapsed_seconds, prompt_tokens, response_tokens)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": OLLAMA_OPTS,
    }
    t0 = time.time()
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    except Exception as e:
        return f"[ERROR: {e}]", time.time() - t0, 0, 0

    elapsed = time.time() - t0

    if resp.status_code != 200:
        return f"[HTTP {resp.status_code}]", elapsed, 0, 0

    data = resp.json()
    text = data.get("response", "").strip()
    prompt_tokens = data.get("prompt_eval_count", 0)
    response_tokens = data.get("eval_count", 0)
    return text, elapsed, prompt_tokens, response_tokens


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

CURATOR_PROMPT = """Given the following context and question, extract ONLY the sentences from the context that are relevant to answering the question. Return just the relevant text, nothing else.

Context: {context}

Question: {question}

Relevant text:"""

ANSWER_PROMPT = """Answer the following question based only on the provided context. Give a short, precise answer.

Context: {context}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def load_squad(path):
    """Load SQuAD 2.0 and return list of (question, answers, context, article_title)."""
    with open(path) as f:
        data = json.load(f)

    entries = []
    all_paragraphs = []  # for sampling distractors

    for article in data["data"]:
        for para in article["paragraphs"]:
            all_paragraphs.append(para["context"])
            for qa in para["qas"]:
                if qa.get("is_impossible", False):
                    continue  # skip unanswerable
                answers = list({a["text"] for a in qa["answers"]})
                if not answers:
                    continue
                entries.append({
                    "id": qa["id"],
                    "question": qa["question"],
                    "answers": answers,
                    "gold_context": para["context"],
                    "title": article["title"],
                })

    return entries, all_paragraphs


def build_noisy_context(gold_context, all_paragraphs, n_distractors, rng):
    """Build a shuffled context with the gold paragraph + n distractor paragraphs."""
    # Sample distractors that aren't the gold paragraph
    candidates = [p for p in all_paragraphs if p != gold_context]
    distractors = rng.sample(candidates, min(n_distractors, len(candidates)))
    paragraphs = [gold_context] + distractors
    rng.shuffle(paragraphs)
    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment():
    print("=" * 70)
    print("CONTEXT CURATION EXPERIMENT")
    print("=" * 70)
    print(f"Answer model:  {ANSWER_MODEL}")
    print(f"Curator model: {CURATOR_MODEL}")
    print(f"Questions:     {NUM_QUESTIONS}")
    print(f"Distractors:   {NUM_DISTRACTORS}")
    print(f"Output:        {OUTPUT_PATH}")
    print("=" * 70)

    # Load dataset
    print("\nLoading SQuAD 2.0 dev set...")
    entries, all_paragraphs = load_squad(SQUAD_PATH)
    print(f"  Answerable questions: {len(entries)}")
    print(f"  Total paragraphs:     {len(all_paragraphs)}")

    # Sample questions
    rng = random.Random(SEED)
    sample = rng.sample(entries, min(NUM_QUESTIONS, len(entries)))

    # Warm up models
    print(f"\nWarming up {ANSWER_MODEL}...")
    call_ollama(ANSWER_MODEL, "Hello", timeout=60)
    print(f"Warming up {CURATOR_MODEL}...")
    call_ollama(CURATOR_MODEL, "Hello", timeout=60)
    print("Models ready.\n")

    # Accumulators
    baseline_em, baseline_f1, baseline_times = [], [], []
    curated_em, curated_f1, curated_times = [], [], []
    curator_times = []
    baseline_prompt_tokens, curated_prompt_tokens = [], []

    # Open output file
    out_f = open(OUTPUT_PATH, "w")

    header = f"{'#':>4}  {'Baseline EM':>11}  {'Baseline F1':>11}  {'Curated EM':>10}  {'Curated F1':>10}  {'Curator s':>9}  {'Noisy tok':>9}  {'Curated tok':>11}"
    print(header)
    print("-" * len(header))

    for i, entry in enumerate(sample):
        qid = entry["id"]
        question = entry["question"]
        answers = entry["answers"]
        gold_ctx = entry["gold_context"]

        # Build noisy context
        noisy_ctx = build_noisy_context(gold_ctx, all_paragraphs, NUM_DISTRACTORS, rng)

        # --- BASELINE: noisy context -> answer model ---
        baseline_prompt = ANSWER_PROMPT.format(context=noisy_ctx, question=question)
        b_answer, b_time, b_ptokens, b_rtokens = call_ollama(ANSWER_MODEL, baseline_prompt)

        b_em = best_score(b_answer, answers, exact_match_score)
        b_f1 = best_score(b_answer, answers, f1_score)
        baseline_em.append(b_em)
        baseline_f1.append(b_f1)
        baseline_times.append(b_time)
        baseline_prompt_tokens.append(b_ptokens)

        # --- CURATED: noisy context -> curator -> answer model ---
        curator_prompt = CURATOR_PROMPT.format(context=noisy_ctx, question=question)
        curated_text, c_time, c_ptokens, c_rtokens = call_ollama(CURATOR_MODEL, curator_prompt)
        curator_times.append(c_time)

        # Use curated text as context for the answer model
        curated_answer_prompt = ANSWER_PROMPT.format(context=curated_text, question=question)
        ca_answer, ca_time, ca_ptokens, ca_rtokens = call_ollama(ANSWER_MODEL, curated_answer_prompt)

        c_em = best_score(ca_answer, answers, exact_match_score)
        c_f1 = best_score(ca_answer, answers, f1_score)
        curated_em.append(c_em)
        curated_f1.append(c_f1)
        curated_times.append(c_time + ca_time)  # total time for curated pipeline
        curated_prompt_tokens.append(ca_ptokens)

        # Log to JSONL
        record = {
            "i": i,
            "id": qid,
            "question": question,
            "answers": answers,
            "gold_context_len": len(gold_ctx),
            "noisy_context_len": len(noisy_ctx),
            "curated_context_len": len(curated_text),
            "baseline_answer": b_answer,
            "baseline_em": b_em,
            "baseline_f1": round(b_f1, 4),
            "baseline_time": round(b_time, 2),
            "baseline_prompt_tokens": b_ptokens,
            "curated_extract": curated_text[:500],  # truncate for readability
            "curated_answer": ca_answer,
            "curated_em": c_em,
            "curated_f1": round(c_f1, 4),
            "curated_total_time": round(c_time + ca_time, 2),
            "curator_time": round(c_time, 2),
            "curated_prompt_tokens": ca_ptokens,
        }
        out_f.write(json.dumps(record) + "\n")
        out_f.flush()

        # Running stats
        n = i + 1
        avg_b_em = sum(baseline_em) / n
        avg_b_f1 = sum(baseline_f1) / n
        avg_c_em = sum(curated_em) / n
        avg_c_f1 = sum(curated_f1) / n
        avg_cur_t = sum(curator_times) / n

        print(
            f"{n:>4}  "
            f"{avg_b_em:>11.1%}  "
            f"{avg_b_f1:>11.1%}  "
            f"{avg_c_em:>10.1%}  "
            f"{avg_c_f1:>10.1%}  "
            f"{avg_cur_t:>8.1f}s  "
            f"{b_ptokens:>9}  "
            f"{ca_ptokens:>11}"
        )

    out_f.close()

    # --- Summary statistics ---
    n = len(sample)
    summary = {
        "num_questions": n,
        "answer_model": ANSWER_MODEL,
        "curator_model": CURATOR_MODEL,
        "num_distractors": NUM_DISTRACTORS,
        "baseline": {
            "exact_match": round(sum(baseline_em) / n, 4),
            "f1": round(sum(baseline_f1) / n, 4),
            "avg_time_s": round(sum(baseline_times) / n, 2),
            "avg_prompt_tokens": round(sum(baseline_prompt_tokens) / n, 1),
        },
        "curated": {
            "exact_match": round(sum(curated_em) / n, 4),
            "f1": round(sum(curated_f1) / n, 4),
            "avg_time_s": round(sum(curated_times) / n, 2),
            "avg_prompt_tokens": round(sum(curated_prompt_tokens) / n, 1),
            "avg_curator_time_s": round(sum(curator_times) / n, 2),
        },
        "deltas": {
            "em_delta": round(sum(curated_em) / n - sum(baseline_em) / n, 4),
            "f1_delta": round(sum(curated_f1) / n - sum(baseline_f1) / n, 4),
            "token_reduction_pct": round(
                (1 - sum(curated_prompt_tokens) / max(sum(baseline_prompt_tokens), 1)) * 100, 1
            ),
        },
    }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Questions evaluated: {n}")
    print(f"Answer model:       {ANSWER_MODEL}")
    print(f"Curator model:      {CURATOR_MODEL}")
    print()
    print(f"{'Metric':<25} {'Baseline':>10} {'Curated':>10} {'Delta':>10}")
    print("-" * 57)
    print(f"{'Exact Match':<25} {summary['baseline']['exact_match']:>10.1%} {summary['curated']['exact_match']:>10.1%} {summary['deltas']['em_delta']:>+10.1%}")
    print(f"{'F1':<25} {summary['baseline']['f1']:>10.1%} {summary['curated']['f1']:>10.1%} {summary['deltas']['f1_delta']:>+10.1%}")
    print(f"{'Avg Time (s)':<25} {summary['baseline']['avg_time_s']:>10.2f} {summary['curated']['avg_time_s']:>10.2f}")
    print(f"{'Avg Prompt Tokens':<25} {summary['baseline']['avg_prompt_tokens']:>10.1f} {summary['curated']['avg_prompt_tokens']:>10.1f} {summary['deltas']['token_reduction_pct']:>+9.1f}%")
    print(f"{'Avg Curator Time (s)':<25} {'—':>10} {summary['curated']['avg_curator_time_s']:>10.2f}")
    print()
    print(f"Results log: {OUTPUT_PATH}")
    print(f"Summary:     {SUMMARY_PATH}")


if __name__ == "__main__":
    run_experiment()
