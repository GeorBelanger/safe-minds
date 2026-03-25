"""
safe-minds — Benchmark Evaluation
==================================
Evaluates the safe-minds two-stage detection pipeline against three
publicly available HuggingFace datasets covering suicidal ideation
and mental health crisis detection.

Datasets
--------
1. vibhorag101/suicide_prediction_dataset_phr
   - 186k train / 23k test, binary (suicide / non-suicide)
   - Source: Reddit (cleaned + lemmatized)

2. AIMH/SWMH
   - 54k posts, multi-class (SuicideWatch, depression, anxiety, bipolar)
   - Source: Reddit r/SuicideWatch + mental health subreddits

3. thePixel42/depression-detection
   - 200k posts, binary (suicide / non-suicide)
   - Source: Reddit r/teenagers, r/SuicideWatch, r/depression
   - Specifically includes youth (r/teenagers) — most relevant to safe-minds

Metrics
-------
- Precision, Recall, F1 (macro + per-class)
- AUC-ROC
- False Negative Rate  ← primary safety metric (missed crises)
- False Positive Rate  ← secondary (alarm fatigue)
- Confusion matrix

Usage
-----
  # Quick eval on 200 samples (fast, no LLM needed)
  python evaluate.py --dataset 1 --stage prefilter --samples 200

  # Full eval with LLM stage on dataset 3 (youth-focused)
  python evaluate.py --dataset 3 --stage llm --samples 500

  # Eval all datasets, pre-filter only
  python evaluate.py --dataset all --stage prefilter --samples 300

  # Save results to JSON
  python evaluate.py --dataset all --stage prefilter --samples 300 --output results.json
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from datasets import load_dataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

from detector import assess, pre_filter, RiskLevel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------

DATASETS = {
    1: {
        "name": "vibhorag101/suicide_prediction_dataset_phr",
        "hf_id": "vibhorag101/suicide_prediction_dataset_phr",
        "split": "test",
        "text_col": "text",
        "label_col": "label",
        # label mapping: "suicide" → 1 (positive), "non-suicide" → 0
        "positive_label": "suicide",
        "description": "Reddit binary (suicide/non-suicide), 23k test samples",
        "youth_relevant": False,
    },
    2: {
        "name": "AIMH/SWMH",
        "hf_id": "AIMH/SWMH",
        "split": "test",
        "text_col": "text",
        "label_col": "label",
        # 0=SuicideWatch, 1=depression, 2=anxiety, 3=bipolar
        # We treat SuicideWatch (0) as positive
        "positive_label": 0,
        "description": "Reddit multi-class mental health, 54k posts",
        "youth_relevant": False,
    },
    3: {
        "name": "thePixel42/depression-detection",
        "hf_id": "thePixel42/depression-detection",
        "split": "test",
        "text_col": "text",
        "label_col": "label",
        # "suicide" → 1, "non-suicide" → 0
        "positive_label": "suicide",
        "description": "Reddit r/teenagers + r/SuicideWatch + r/depression, 60k test — YOUTH RELEVANT",
        "youth_relevant": True,
    },
}

# ---------------------------------------------------------------------------
# Risk level → binary mapping
# ---------------------------------------------------------------------------

def risk_to_binary(risk: RiskLevel) -> int:
    """Map our 5-level risk to binary positive (1) / negative (0)."""
    return 1 if risk in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRISIS) else 0


def risk_to_binary_strict(risk: RiskLevel) -> int:
    """Strict: only HIGH and CRISIS count as positive."""
    return 1 if risk in (RiskLevel.HIGH, RiskLevel.CRISIS) else 0


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    dataset_name: str
    stage: str
    n_samples: int
    threshold: str                      # "standard" or "strict"
    precision: float
    recall: float
    f1: float
    auc_roc: float
    false_negative_rate: float          # missed crises — primary safety metric
    false_positive_rate: float
    confusion_matrix: list
    classification_report: str
    evaluated_at: str
    elapsed_seconds: float
    samples_per_second: float

    def summary(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"Dataset : {self.dataset_name}\n"
            f"Stage   : {self.stage}  |  Threshold: {self.threshold}\n"
            f"Samples : {self.n_samples}\n"
            f"{'─'*60}\n"
            f"Precision : {self.precision:.3f}\n"
            f"Recall    : {self.recall:.3f}\n"
            f"F1        : {self.f1:.3f}\n"
            f"AUC-ROC   : {self.auc_roc:.3f}\n"
            f"{'─'*60}\n"
            f"⚠️  False Negative Rate (missed crises) : {self.false_negative_rate:.3f}\n"
            f"   False Positive Rate (alarm fatigue)  : {self.false_positive_rate:.3f}\n"
            f"{'─'*60}\n"
            f"Speed   : {self.samples_per_second:.1f} samples/sec\n"
            f"{'='*60}"
        )


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_samples(cfg: dict, n: Optional[int] = None) -> list[tuple[str, int]]:
    """Load (text, binary_label) pairs from a HuggingFace dataset."""
    logger.info("Loading dataset: %s (split=%s)", cfg["hf_id"], cfg["split"])

    try:
        ds = load_dataset(cfg["hf_id"], split=cfg["split"])
    except Exception as e:
        logger.error("Failed to load %s: %s", cfg["hf_id"], e)
        raise

    samples = []
    for row in ds:
        text = str(row[cfg["text_col"]]).strip()
        raw_label = row[cfg["label_col"]]
        # Normalise to binary
        is_positive = (raw_label == cfg["positive_label"]) or (raw_label == 1 and cfg["positive_label"] == 1)
        label = 1 if is_positive else 0
        if text:
            samples.append((text, label))

    if n and n < len(samples):
        # Stratified subsample: keep class balance
        pos = [(t, l) for t, l in samples if l == 1]
        neg = [(t, l) for t, l in samples if l == 0]
        half = n // 2
        import random; random.seed(42)
        sub = random.sample(pos, min(half, len(pos))) + random.sample(neg, min(n - min(half, len(pos)), len(neg)))
        random.shuffle(sub)
        samples = sub

    pos_count = sum(l for _, l in samples)
    logger.info("Loaded %d samples (%d positive, %d negative)", len(samples), pos_count, len(samples) - pos_count)
    return samples


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate_prefilter(
    samples: list[tuple[str, int]],
    threshold: str = "standard",
) -> tuple[list[int], list[int]]:
    """Run Stage 1 pre-filter only — fast, no model needed."""
    y_true, y_pred = [], []
    for text, label in samples:
        risk, _ = pre_filter(text)
        pred = risk_to_binary(risk) if threshold == "standard" else risk_to_binary_strict(risk)
        y_true.append(label)
        y_pred.append(pred)
    return y_true, y_pred


def evaluate_llm(
    samples: list[tuple[str, int]],
    threshold: str = "standard",
) -> tuple[list[int], list[int]]:
    """Run full two-stage pipeline (pre-filter + LLM). Slower but more accurate."""
    y_true, y_pred = [], []
    for i, (text, label) in enumerate(samples):
        if (i + 1) % 25 == 0:
            logger.info("LLM eval: %d/%d", i + 1, len(samples))
        result = assess(text)
        pred = risk_to_binary(result.risk_level) if threshold == "standard" else risk_to_binary_strict(result.risk_level)
        y_true.append(label)
        y_pred.append(pred)
    return y_true, y_pred


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    dataset_name: str,
    stage: str,
    n_samples: int,
    threshold: str,
    elapsed: float,
) -> EvalResult:
    """Compute all metrics from predictions."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0   # missed crisis rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # alarm fatigue rate

    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = float("nan")

    report = classification_report(y_true, y_pred, target_names=["non-crisis", "crisis"])

    return EvalResult(
        dataset_name=dataset_name,
        stage=stage,
        n_samples=n_samples,
        threshold=threshold,
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        auc_roc=auc,
        false_negative_rate=fnr,
        false_positive_rate=fpr,
        confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
        classification_report=report,
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=round(elapsed, 2),
        samples_per_second=round(n_samples / elapsed, 1) if elapsed > 0 else 0.0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eval(
    dataset_ids: list[int],
    stage: str,
    n_samples: Optional[int],
    threshold: str,
    output_path: Optional[str],
):
    all_results = []

    for ds_id in dataset_ids:
        cfg = DATASETS[ds_id]
        print(f"\n▶ Dataset {ds_id}: {cfg['name']}")
        print(f"  {cfg['description']}")
        if cfg["youth_relevant"]:
            print("  ★ Youth-relevant — most aligned with safe-minds target population")

        try:
            samples = load_samples(cfg, n=n_samples)
        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
            continue

        t0 = time.time()

        if stage == "prefilter":
            y_true, y_pred = evaluate_prefilter(samples, threshold)
        else:
            y_true, y_pred = evaluate_llm(samples, threshold)

        elapsed = time.time() - t0
        result = compute_metrics(y_true, y_pred, cfg["name"], stage, len(samples), threshold, elapsed)
        all_results.append(result)

        print(result.summary())
        print("\nClassification Report:")
        print(result.classification_report)

        cm = result.confusion_matrix
        print(f"Confusion Matrix (rows=actual, cols=predicted):")
        print(f"             Pred NEG  Pred POS")
        print(f"  Actual NEG   {cm[0][0]:6d}    {cm[0][1]:6d}")
        print(f"  Actual POS   {cm[1][0]:6d}    {cm[1][1]:6d}")

    # Summary table
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("SUMMARY TABLE")
        print("="*60)
        print(f"{'Dataset':<45} {'F1':>6} {'Recall':>8} {'FNR':>8}")
        print("─"*60)
        for r in all_results:
            name = r.dataset_name.split("/")[-1][:44]
            print(f"{name:<45} {r.f1:>6.3f} {r.recall:>8.3f} {r.false_negative_rate:>8.3f}")
        print("─"*60)
        print("FNR = False Negative Rate (missed crises) — lower is better")

    # Save results
    if output_path:
        out = [asdict(r) for r in all_results]
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="safe-minds benchmark evaluation")
    parser.add_argument(
        "--dataset", type=str, default="all",
        help="Dataset ID (1, 2, 3) or 'all'. Default: all",
    )
    parser.add_argument(
        "--stage", type=str, default="prefilter",
        choices=["prefilter", "llm"],
        help="Which stage to evaluate. 'prefilter' is fast; 'llm' loads Phi-3-mini. Default: prefilter",
    )
    parser.add_argument(
        "--samples", type=int, default=300,
        help="Number of samples per dataset (stratified). Default: 300",
    )
    parser.add_argument(
        "--threshold", type=str, default="standard",
        choices=["standard", "strict"],
        help="standard: MEDIUM/HIGH/CRISIS = positive. strict: HIGH/CRISIS only. Default: standard",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional path to save JSON results (e.g. results.json)",
    )

    args = parser.parse_args()

    if args.dataset == "all":
        ids = [1, 2, 3]
    else:
        ids = [int(x.strip()) for x in args.dataset.split(",")]

    print("\n" + "="*60)
    print("safe-minds — Benchmark Evaluation")
    print("="*60)
    print(f"Stage     : {args.stage}")
    print(f"Threshold : {args.threshold}")
    print(f"Samples   : {args.samples} per dataset (stratified)")
    print(f"Datasets  : {ids}")

    run_eval(
        dataset_ids=ids,
        stage=args.stage,
        n_samples=args.samples,
        threshold=args.threshold,
        output_path=args.output,
    )
