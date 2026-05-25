"""Evaluation runner for Lumé.

Usage:
  python -m eval.run                    # deterministic checks only
  python -m eval.run --judge            # + 3-axis LLM judge
  python -m eval.run --case C01 C03     # run specific cases only
  python -m eval.run --out eval/runs/my_run
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from eval.deterministic import Result, run_checks
from lume.agents.graph import run_turn
from lume.catalog.loader import load_products
from lume.catalog.normalize import normalize_all
from lume.config import CATALOG_PATH
from lume.schemas import Reply

CASES_FILE = Path(__file__).parent / "cases.yaml"
DEFAULT_OUT_DIR = Path(__file__).parent / "runs"


# ── Result structures ─────────────────────────────────────────────────────────

@dataclass
class TurnResult:
    turn_index: int
    message: str
    expected_mode: str | None
    reply_mode: str
    reply_text: str
    latency_ms: int
    check_results: list[dict]
    judge_scores: dict | None = None
    error: str | None = None


@dataclass
class CaseResult:
    id: str
    description: str
    user_id: str | None
    multi_turn: bool
    turns: list[TurnResult] = field(default_factory=list)
    error: str | None = None

    @property
    def checks_passed(self) -> int:
        return sum(
            1
            for t in self.turns
            for c in t.check_results
            if c["result"] == "pass"
        )

    @property
    def checks_failed(self) -> int:
        return sum(
            1
            for t in self.turns
            for c in t.check_results
            if c["result"] == "fail"
        )

    @property
    def checks_skipped(self) -> int:
        return sum(
            1
            for t in self.turns
            for c in t.check_results
            if c["result"] == "skip"
        )

    @property
    def passed(self) -> bool:
        return self.error is None and self.checks_failed == 0

    def mean_judge_score(self, axis: str) -> float | None:
        scores = [
            t.judge_scores[axis]
            for t in self.turns
            if t.judge_scores and axis in t.judge_scores
        ]
        return sum(scores) / len(scores) if scores else None


# ── Main runner ───────────────────────────────────────────────────────────────

def _load_cases(path: Path = CASES_FILE) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["cases"]


def _build_catalog_ids() -> set[str]:
    products = normalize_all(load_products(CATALOG_PATH))
    return {p.product_id for p in products}


def _run_case(
    case: dict,
    catalog_ids: set[str],
    with_judge: bool,
) -> CaseResult:
    result = CaseResult(
        id=case["id"],
        description=case["description"],
        user_id=case.get("user_id"),
        multi_turn=case.get("multi_turn", False),
    )

    # Cross-turn state
    state: dict[str, Any] = {
        "user_id": case.get("user_id"),
        "current_intent": None,
        "last_shown_product_ids": [],
        "last_shown_products": [],
        "topic_history": [],
        "clarify_count": 0,
        "last_action": None,
        "last_shown_mode": None,
    }

    turns: list[dict] = case.get("turns", [])
    for i, turn in enumerate(turns):
        message: str = turn["message"]
        expected_mode: str | None = turn.get("expected_mode")
        checks_def: dict = turn.get("checks") or {}

        t_start = time.perf_counter()
        try:
            output = run_turn(message, **state)
        except Exception:
            tr = TurnResult(
                turn_index=i,
                message=message,
                expected_mode=expected_mode,
                reply_mode="error",
                reply_text="",
                latency_ms=int((time.perf_counter() - t_start) * 1000),
                check_results=[],
                error=traceback.format_exc(),
            )
            result.turns.append(tr)
            break

        latency_ms = int((time.perf_counter() - t_start) * 1000)

        # Update cross-turn state from output
        for key in (
            "current_intent",
            "last_shown_product_ids",
            "last_shown_products",
            "topic_history",
            "clarify_count",
            "last_action",
            "last_shown_mode",
        ):
            if key in output:
                state[key] = output[key]

        reply: Reply | None = output.get("reply")
        if reply is None:
            tr = TurnResult(
                turn_index=i,
                message=message,
                expected_mode=expected_mode,
                reply_mode="error",
                reply_text="",
                latency_ms=latency_ms,
                check_results=[],
                error="graph returned no reply",
            )
            result.turns.append(tr)
            break

        check_results = run_checks(reply, checks_def, catalog_ids)

        judge_scores = None
        if with_judge:
            from eval.judge import judge_reply  # noqa: PLC0415

            scores = judge_reply(message, reply, case["description"])
            if scores:
                judge_scores = {
                    "relevance": scores.relevance,
                    "brand_voice": scores.brand_voice,
                    "whatsapp_feel": scores.whatsapp_feel,
                    "reasoning": scores.reasoning,
                }

        result.turns.append(
            TurnResult(
                turn_index=i,
                message=message,
                expected_mode=expected_mode,
                reply_mode=reply.mode,
                reply_text=reply.reply_text,
                latency_ms=latency_ms,
                check_results=[asdict(c) for c in check_results],
                judge_scores=judge_scores,
            )
        )

    return result


def _serialize_case(cr: CaseResult) -> dict:
    return {
        "id": cr.id,
        "description": cr.description,
        "user_id": cr.user_id,
        "multi_turn": cr.multi_turn,
        "passed": cr.passed,
        "checks_passed": cr.checks_passed,
        "checks_failed": cr.checks_failed,
        "checks_skipped": cr.checks_skipped,
        "error": cr.error,
        "turns": [
            {
                "turn": t.turn_index,
                "message": t.message,
                "expected_mode": t.expected_mode,
                "reply_mode": t.reply_mode,
                "reply_text": t.reply_text,
                "latency_ms": t.latency_ms,
                "checks": t.check_results,
                "judge": t.judge_scores,
                "error": t.error,
            }
            for t in cr.turns
        ],
    }


def _write_report(
    case_results: list[CaseResult],
    out_dir: Path,
    elapsed_s: float,
    with_judge: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    total = len(case_results)
    passed = sum(1 for c in case_results if c.passed)
    failed = total - passed

    total_checks = sum(c.checks_passed + c.checks_failed for c in case_results)
    total_passed = sum(c.checks_passed for c in case_results)
    pass_rate = (total_passed / total_checks * 100) if total_checks else 0

    mean_latency = (
        sum(t.latency_ms for c in case_results for t in c.turns) / max(
            sum(len(c.turns) for c in case_results), 1
        )
    )

    judge_axes = ("relevance", "brand_voice", "whatsapp_feel")
    judge_means: dict[str, float | None] = {}
    if with_judge:
        for axis in judge_axes:
            scores = [
                t.judge_scores[axis]
                for c in case_results
                for t in c.turns
                if t.judge_scores and axis in t.judge_scores
            ]
            judge_means[axis] = round(sum(scores) / len(scores), 2) if scores else None

    # ── JSON report ──
    report_json = {
        "timestamp": ts,
        "summary": {
            "total_cases": total,
            "cases_passed": passed,
            "cases_failed": failed,
            "check_pass_rate_pct": round(pass_rate, 1),
            "mean_latency_ms": round(mean_latency, 0),
            "judge_means": judge_means if with_judge else None,
        },
        "cases": [_serialize_case(c) for c in case_results],
    }
    json_path = out_dir / f"report_{ts}.json"
    json_path.write_text(json.dumps(report_json, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Markdown report ──
    lines: list[str] = [
        f"# Lumé Eval Report — {ts}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Cases passed | {passed}/{total} |",
        f"| Check pass rate | {pass_rate:.1f}% |",
        f"| Mean latency | {mean_latency:.0f} ms |",
    ]
    if with_judge:
        for axis, val in judge_means.items():
            lines.append(f"| Judge {axis} | {val if val is not None else 'N/A'} / 5 |")

    lines += ["", "## Case Results", ""]
    lines.append("| ID | Description | Mode | Pass? | Checks | Latency |")
    lines.append("|----|-------------|------|-------|--------|---------|")
    for cr in case_results:
        last_turn = cr.turns[-1] if cr.turns else None
        mode = last_turn.reply_mode if last_turn else "error"
        check_summary = f"{cr.checks_passed}✓ {cr.checks_failed}✗ {cr.checks_skipped}–"
        latency = f"{last_turn.latency_ms}ms" if last_turn else "—"
        passed_icon = "PASS" if cr.passed else "FAIL"
        lines.append(
            f"| {cr.id} | {cr.description[:45]} | {mode} | {passed_icon} | {check_summary} | {latency} |"
        )

    lines += ["", "## Failed Checks", ""]
    any_failed = False
    for cr in case_results:
        for t in cr.turns:
            failed_checks = [c for c in t.check_results if c["result"] == "fail"]
            if failed_checks:
                any_failed = True
                lines.append(f"**{cr.id}** turn {t.turn_index} (`{t.reply_mode}`):")
                for c in failed_checks:
                    lines.append(f"  - `{c['name']}` FAIL: {c.get('detail', '')}")
    if not any_failed:
        lines.append("_All checks passed._")

    if with_judge:
        lines += ["", "## Judge Scores by Case", ""]
        lines.append("| ID | Relevance | Brand Voice | WhatsApp Feel |")
        lines.append("|----|-----------|-------------|---------------|")
        for cr in case_results:
            rel = cr.mean_judge_score("relevance")
            bv = cr.mean_judge_score("brand_voice")
            wf = cr.mean_judge_score("whatsapp_feel")
            fmt = lambda v: f"{v:.1f}" if v is not None else "—"
            lines.append(f"| {cr.id} | {fmt(rel)} | {fmt(bv)} | {fmt(wf)} |")

    lines += ["", f"_Run took {elapsed_s:.1f}s total._", ""]

    md_path = out_dir / f"report_{ts}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  Cases:  {passed}/{total} passed")
    print(f"  Checks: {pass_rate:.1f}% pass rate ({total_passed}/{total_checks})")
    print(f"  Latency: {mean_latency:.0f}ms mean")
    if with_judge:
        for axis, val in judge_means.items():
            print(f"  Judge {axis}: {val if val is not None else 'N/A'}/5")
    print(f"  Reports: {json_path}")
    print(f"          {md_path}")
    print(f"{sep}\n")


def run_eval(
    case_ids: list[str] | None = None,
    with_judge: bool = False,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> list[CaseResult]:
    print("Loading catalog…")
    catalog_ids = _build_catalog_ids()
    print(f"  {len(catalog_ids)} products loaded")

    all_cases = _load_cases()
    if case_ids:
        all_cases = [c for c in all_cases if c["id"] in case_ids]
    print(f"  Running {len(all_cases)} cases{' (with LLM judge)' if with_judge else ''}…\n")

    results: list[CaseResult] = []
    t_total = time.perf_counter()

    for case in all_cases:
        print(f"  [{case['id']}] {case['description'][:60]}…", end=" ", flush=True)
        cr = _run_case(case, catalog_ids, with_judge=with_judge)
        results.append(cr)
        icon = "OK" if cr.passed else ("FAIL" if cr.error is None else "ERR")
        fail_detail = ""
        if cr.checks_failed:
            fail_detail = f" ({cr.checks_failed} checks failed)"
        print(f"{icon}{fail_detail}")

    elapsed = time.perf_counter() - t_total
    _write_report(results, out_dir, elapsed, with_judge)
    return results


# ── CLI entry ─────────────────────────────────────────────────────────────────

def _parse_args() -> tuple[list[str] | None, bool, Path]:
    import argparse

    parser = argparse.ArgumentParser(description="Run Lumé eval harness")
    parser.add_argument("--judge", action="store_true", help="Enable LLM judge scoring")
    parser.add_argument("--case", nargs="+", metavar="ID", help="Run only these case IDs")
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT_DIR),
        metavar="DIR",
        help="Output directory for reports",
    )
    args = parser.parse_args()
    return args.case, args.judge, Path(args.out)


if __name__ == "__main__":
    case_ids, with_judge, out_dir = _parse_args()
    results = run_eval(case_ids=case_ids, with_judge=with_judge, out_dir=out_dir)
    failed = sum(1 for r in results if not r.passed)
    sys.exit(1 if failed else 0)
