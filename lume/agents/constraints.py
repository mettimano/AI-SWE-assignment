"""Hard constraint filters applied after retrieval, before reranking.

Only structural/numeric fields are filtered here.
Soft preferences (brand, fragrance family, gender, must_avoid keywords) are delegated
to the LLM reranker via natural-language instructions in soft_instructions().
"""

from __future__ import annotations

from lume.agents.intent import Intent
from lume.catalog.models import NormalizedProduct


def apply(
    candidates: list[NormalizedProduct],
    intent: Intent,
    *,
    include_oos_fallback: bool = True,
) -> tuple[list[NormalizedProduct], list[NormalizedProduct]]:
    """Filter candidates by hard constraints derived from Intent.

    Returns:
        (primary, oos_fallback)
        primary      — products passing all constraints; shown to user
        oos_fallback — out-of-stock products that would otherwise match; used only
                       in no_match mode to suggest alternatives
    """
    primary: list[NormalizedProduct] = []
    oos_fallback: list[NormalizedProduct] = []

    for p in candidates:
        # Budget: use cheapest available variant price; skip if over budget
        if intent.budget_max is not None:
            candidate_price = p.cheapest_available_price
            if candidate_price is not None and candidate_price > intent.budget_max:
                continue
            # If no available variants but product has a listed price, check max
            if candidate_price is None and p.min_price_eur > intent.budget_max:
                continue

        # Stock: separate out-of-stock into fallback bucket
        if not p.effective_available:
            if include_oos_fallback:
                oos_fallback.append(p)
            continue

        primary.append(p)

    return primary, oos_fallback


def soft_instructions(intent: Intent) -> str:
    """Build natural-language soft-filter instructions for the reranker.

    These cover preferences that can't be expressed as structured WHERE clauses
    (brand, fragrance family, gender, negations, occasion, niche preference).
    """
    parts: list[str] = []

    if intent.fragrance_family:
        parts.append(f"Preferisce famiglie olfattive: {', '.join(intent.fragrance_family)}.")
    if intent.gender_lean:
        parts.append(f"Orientamento di genere: {intent.gender_lean}.")
    if intent.occasion:
        parts.append(f"Occasione: {intent.occasion}.")
    if intent.must_avoid:
        parts.append(f"Evitare assolutamente: {', '.join(intent.must_avoid)}.")
    if intent.niche_preference is True:
        parts.append("Preferisce prodotti di nicchia.")
    elif intent.niche_preference is False:
        parts.append("Preferisce prodotti mainstream (non di nicchia).")
    if intent.gift_recipient:
        parts.append(f"È un regalo per: {intent.gift_recipient}.")
    if intent.must_include:
        parts.append(
            f"Deve includere o essere simile a: {', '.join(intent.must_include)}."
        )

    return " ".join(parts)
