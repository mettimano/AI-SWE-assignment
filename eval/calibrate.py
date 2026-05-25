"""Human-baseline calibration — computes Cohen's kappa (judge ↔ human) per axis.

Usage:
  1. Run `python -m eval.run --judge` to generate replies and judge scores.
  2. Open eval/human_baseline.json and fill in reply_text + human_ratings (1-5).
  3. Run: python -m eval.calibrate <path/to/report_*.json>

Outputs per-axis Cohen's κ and flags axes with κ < 0.4 as untrusted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _linear_weights(n: int) -> list[list[float]]:
    """Linear weighting matrix for Cohen's kappa (size n×n for ratings 1..n)."""
    W = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            W[i][j] = 1.0 - abs(i - j) / (n - 1)
    return W


def _weighted_kappa(ratings_a: list[int], ratings_b: list[int], n: int = 5) -> float:
    """Weighted Cohen's kappa with linear weights for 1..n scale."""
    if len(ratings_a) != len(ratings_b) or not ratings_a:
        raise ValueError("rating lists must be equal length and non-empty")

    W = _linear_weights(n)
    k = len(ratings_a)
    # Observed agreement
    Po = sum(W[a - 1][b - 1] for a, b in zip(ratings_a, ratings_b)) / k
    # Expected agreement
    freq_a = [sum(1 for r in ratings_a if r == v) / k for v in range(1, n + 1)]
    freq_b = [sum(1 for r in ratings_b if r == v) / k for v in range(1, n + 1)]
    Pe = sum(W[i][j] * freq_a[i] * freq_b[j] for i in range(n) for j in range(n))

    if Pe == 1.0:
        return 1.0
    return (Po - Pe) / (1.0 - Pe)


def calibrate(report_path: Path, baseline_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    baseline: list[dict] = [
        b for b in json.loads(baseline_path.read_text(encoding="utf-8"))
        if "_note" not in b
    ]

    # Index report judge scores by (case_id, turn)
    judge_index: dict[tuple[str, int], dict] = {}
    for case in report["cases"]:
        for turn in case["turns"]:
            if turn.get("judge"):
                judge_index[(case["id"], turn["turn"])] = turn["judge"]

    axes = ("relevance", "brand_voice", "whatsapp_feel")
    axis_pairs: dict[str, tuple[list[int], list[int]]] = {ax: ([], []) for ax in axes}

    skipped = []
    for sample in baseline:
        key = (sample["case_id"], sample["turn"])
        if key not in judge_index:
            skipped.append(key)
            continue
        human_ratings = sample.get("human_ratings") or {}
        judge_scores = judge_index[key]

        for ax in axes:
            h = human_ratings.get(ax)
            j = judge_scores.get(ax)
            if h is not None and j is not None:
                axis_pairs[ax][0].append(int(h))
                axis_pairs[ax][1].append(int(j))

    print("\n══ Human-baseline calibration ══════════════════════════════════\n")

    if skipped:
        print(f"  ⚠  Skipped {len(skipped)} samples (not found in report or missing judge scores):")
        for k in skipped:
            print(f"      {k[0]} turn {k[1]}")
        print()

    results: dict[str, float | None] = {}
    for ax in axes:
        human, judge = axis_pairs[ax]
        if len(human) < 3:
            print(f"  {ax:15s}: insufficient data ({len(human)} pair(s)) — fill human_baseline.json")
            results[ax] = None
        else:
            kappa = _weighted_kappa(human, judge)
            trust = "✓ trusted" if kappa >= 0.4 else "⚠ UNTRUSTED (κ < 0.4)"
            print(f"  {ax:15s}: κ = {kappa:+.3f}  ({len(human)} pairs)  {trust}")
            results[ax] = kappa

    print()
    untrusted = [ax for ax, k in results.items() if k is not None and k < 0.4]
    if untrusted:
        print(f"  Action needed: judge scores for {untrusted} should not be reported as ground truth.")
    else:
        print("  All scored axes meet κ ≥ 0.4 threshold.")
    print()

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m eval.calibrate <report_*.json>")
        sys.exit(1)
    report_path = Path(sys.argv[1])
    baseline_path = Path(__file__).parent / "human_baseline.json"
    calibrate(report_path, baseline_path)
