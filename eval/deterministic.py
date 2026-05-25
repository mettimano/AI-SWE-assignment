"""Deterministic checks — each check returns PASS, FAIL, or SKIP.

Checks are registered by name and dispatched by run_checks().
All checks take (reply, check_value, catalog_ids) and return CheckResult.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from lume.schemas import Reply


class Result(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class CheckResult:
    name: str
    result: Result
    detail: str = ""

    def passed(self) -> bool:
        return self.result == Result.PASS

    def failed(self) -> bool:
        return self.result == Result.FAIL


# ── Regex helpers shared with guard ──────────────────────────────────────────

_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"
    "\U00002702-\U000027B0"
    "\U0000FE00-\U0000FE0F"
    "\U00002500-\U00002BEF"
    "✨🌸💋✅⭐❤🎁🎄🛒💄"
    "]",
    flags=re.UNICODE,
)

_MARKDOWN_RE = re.compile(
    r"\*\*[^*]+\*\*"
    r"|__[^_]+__"
    r"|\*[^*]+\*"
    r"|^#{1,6}\s"
    r"|^\s*-\s"
    r"|^\s*\d+\.\s",
    re.MULTILINE,
)

_MEDICAL_RE = re.compile(
    r"\b(cura|guarisce|tratta(?:mento)?|terapeutic[oa]|clinicament[ei]|"
    r"medicinale|farmaco|diagnosi|prescriz|dermatolog|allergen|ipoallergen|"
    r"cures?|heals?|treats?|therapeutic|clinically[ -]proven)\b",
    re.IGNORECASE,
)

_OUD_RE = re.compile(r"\b(oud|aoud)\b", re.IGNORECASE)

_ITALIAN_WORDS = re.compile(
    r"\b(il|la|le|di|che|per|un|una|ho|hai|mi|ti|si|ci|vi|con|non|ma|è|"
    r"questo|quello|ecco|molto|bene|certo|anche|poi|qui|dove|quando|come|"
    r"prodotto|profumo|prezzo|disponibile|regalo|budget)\b",
    re.IGNORECASE,
)
_ENGLISH_WORDS = re.compile(
    r"\b(the|and|for|with|your|you|this|that|our|we|have|here|some|one|"
    r"of|in|to|is|it|are|can|would|could|great|perfect|available|price|"
    r"budget|fragrance|perfume|gift)\b",
    re.IGNORECASE,
)


def _parse_min_price(price_display: str) -> float | None:
    """Extract the minimum (first) EUR price from a display string like €49.00 or €49.00–€120.00."""
    m = re.search(r"€(\d+(?:[.,]\d+)?)", price_display)
    if m:
        return float(m.group(1).replace(",", "."))
    return None


# ── Individual check functions ────────────────────────────────────────────────

def _check_mode_matches(reply: Reply, expected: str, _: set[str]) -> CheckResult:
    if reply.mode == expected:
        return CheckResult("mode_matches", Result.PASS)
    return CheckResult("mode_matches", Result.FAIL, f"expected={expected} got={reply.mode}")


def _check_budget_respected(reply: Reply, _: Any, __: set[str]) -> CheckResult:
    intent = reply.intent
    if intent is None or intent.budget_max is None:
        return CheckResult("budget_respected", Result.SKIP, "no budget in intent")
    violations = []
    for rec in reply.recommendations:
        price = _parse_min_price(rec.price_display)
        if price is not None and price > intent.budget_max:
            violations.append(f"{rec.product_id}={price:.2f}>{intent.budget_max:.2f}")
    if violations:
        return CheckResult("budget_respected", Result.FAIL, "; ".join(violations))
    return CheckResult("budget_respected", Result.PASS)


def _check_in_stock_first(reply: Reply, _: Any, __: set[str]) -> CheckResult:
    if reply.mode not in ("answer", "specification"):
        return CheckResult("in_stock_first", Result.SKIP, f"mode={reply.mode}")
    if not reply.recommendations:
        return CheckResult("in_stock_first", Result.SKIP, "no recommendations")
    first = reply.recommendations[0]
    if first.available:
        return CheckResult("in_stock_first", Result.PASS)
    return CheckResult("in_stock_first", Result.FAIL, f"first={first.product_id} is OOS")


def _check_cite_only_catalog(reply: Reply, _: Any, catalog_ids: set[str]) -> CheckResult:
    bad = [r.product_id for r in reply.recommendations if r.product_id not in catalog_ids]
    if bad:
        return CheckResult("cite_only_catalog", Result.FAIL, f"hallucinated: {bad}")
    return CheckResult("cite_only_catalog", Result.PASS)


def _check_no_markdown(reply: Reply, _: Any, __: set[str]) -> CheckResult:
    if _MARKDOWN_RE.search(reply.reply_text):
        sample = _MARKDOWN_RE.findall(reply.reply_text)[:2]
        return CheckResult("no_markdown", Result.FAIL, f"found: {sample}")
    return CheckResult("no_markdown", Result.PASS)


def _check_no_medical_claims(reply: Reply, _: Any, __: set[str]) -> CheckResult:
    if _MEDICAL_RE.search(reply.reply_text):
        m = _MEDICAL_RE.search(reply.reply_text)
        return CheckResult("no_medical_claims", Result.FAIL, f"found: {m.group()}")
    return CheckResult("no_medical_claims", Result.PASS)


def _check_max_emoji_1(reply: Reply, _: Any, __: set[str]) -> CheckResult:
    count = len(_EMOJI_RE.findall(reply.reply_text))
    if count <= 1:
        return CheckResult("max_emoji_1", Result.PASS, f"count={count}")
    return CheckResult("max_emoji_1", Result.FAIL, f"count={count}")


def _check_language_match(reply: Reply, expected_lang: str, _: set[str]) -> CheckResult:
    intent_lang = (reply.intent.language if reply.intent else None) or "it"
    if intent_lang == expected_lang:
        return CheckResult("language_match", Result.PASS, f"intent.language={intent_lang}")
    return CheckResult("language_match", Result.FAIL, f"expected={expected_lang} got={intent_lang}")


def _check_clarify_question_count_le_2(reply: Reply, _: Any, __: set[str]) -> CheckResult:
    if reply.mode != "clarify_question":
        return CheckResult("clarify_question_count_le_2", Result.SKIP, f"mode={reply.mode}")
    count = len(reply.questions)
    if count <= 2:
        return CheckResult("clarify_question_count_le_2", Result.PASS, f"count={count}")
    return CheckResult("clarify_question_count_le_2", Result.FAIL, f"count={count}>2")


def _check_probe_count_in_range_if_probe(reply: Reply, _: Any, __: set[str]) -> CheckResult:
    if reply.mode != "clarify_probe":
        return CheckResult("probe_count_in_range_if_probe", Result.SKIP, f"mode={reply.mode}")
    count = len(reply.probes)
    if 2 <= count <= 4:
        return CheckResult("probe_count_in_range_if_probe", Result.PASS, f"count={count}")
    return CheckResult("probe_count_in_range_if_probe", Result.FAIL, f"count={count} not in [2,4]")


def _check_escalation_no_products(reply: Reply, _: Any, __: set[str]) -> CheckResult:
    if reply.mode != "escalate":
        return CheckResult("escalation_no_products", Result.SKIP, f"mode={reply.mode}")
    if reply.recommendations:
        ids = [r.product_id for r in reply.recommendations]
        return CheckResult("escalation_no_products", Result.FAIL, f"found products: {ids}")
    return CheckResult("escalation_no_products", Result.PASS)


def _check_needs_human_flag(reply: Reply, _: Any, __: set[str]) -> CheckResult:
    if reply.needs_human:
        return CheckResult("needs_human_flag", Result.PASS)
    return CheckResult("needs_human_flag", Result.FAIL, "needs_human=False")


def _check_no_oud_cited(reply: Reply, _: Any, __: set[str]) -> CheckResult:
    oud_recs = [r for r in reply.recommendations if _OUD_RE.search(r.title)]
    if oud_recs:
        titles = [r.title for r in oud_recs]
        return CheckResult("no_oud_cited", Result.FAIL, f"oud products cited: {titles}")
    return CheckResult("no_oud_cited", Result.PASS)


def _check_memory_budget_respected(reply: Reply, _: Any, __: set[str]) -> CheckResult:
    """Giulia's profile has budget_max=80; allow ≤85 for rounding tolerance."""
    limit = 85.0
    violations = []
    for rec in reply.recommendations:
        price = _parse_min_price(rec.price_display)
        if price is not None and price > limit:
            violations.append(f"{rec.product_id}={price:.2f}>{limit:.2f}")
    if violations:
        return CheckResult("memory_budget_respected", Result.FAIL, "; ".join(violations))
    return CheckResult("memory_budget_respected", Result.PASS)


def _check_no_products_in_clarify(reply: Reply, _: Any, __: set[str]) -> CheckResult:
    if reply.mode != "clarify_question":
        return CheckResult("no_products_in_clarify", Result.SKIP, f"mode={reply.mode}")
    if reply.recommendations:
        ids = [r.product_id for r in reply.recommendations]
        return CheckResult("no_products_in_clarify", Result.FAIL, f"found products: {ids}")
    return CheckResult("no_products_in_clarify", Result.PASS)


# ── Dispatch table ────────────────────────────────────────────────────────────

_CHECKS: dict[str, Any] = {
    "mode_matches": _check_mode_matches,
    "budget_respected": _check_budget_respected,
    "in_stock_first": _check_in_stock_first,
    "cite_only_catalog": _check_cite_only_catalog,
    "no_markdown": _check_no_markdown,
    "no_medical_claims": _check_no_medical_claims,
    "max_emoji_1": _check_max_emoji_1,
    "language_match": _check_language_match,
    "clarify_question_count_le_2": _check_clarify_question_count_le_2,
    "probe_count_in_range_if_probe": _check_probe_count_in_range_if_probe,
    "escalation_no_products": _check_escalation_no_products,
    "needs_human_flag": _check_needs_human_flag,
    "no_oud_cited": _check_no_oud_cited,
    "memory_budget_respected": _check_memory_budget_respected,
    "no_products_in_clarify": _check_no_products_in_clarify,
}


# ── Public API ────────────────────────────────────────────────────────────────

def run_checks(
    reply: Reply,
    checks: dict[str, Any],
    catalog_ids: set[str],
) -> list[CheckResult]:
    """Run all checks specified in the checks dict. Returns one CheckResult per check."""
    results: list[CheckResult] = []
    for check_name, check_value in checks.items():
        fn = _CHECKS.get(check_name)
        if fn is None:
            results.append(CheckResult(check_name, Result.SKIP, "unknown check"))
            continue
        try:
            result = fn(reply, check_value, catalog_ids)
        except Exception as exc:
            results.append(CheckResult(check_name, Result.FAIL, f"exception: {exc}"))
        else:
            results.append(result)
    return results
