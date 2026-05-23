"""LLM reranker: gpt-4o-mini re-orders the fused top-N with natural-language instructions."""

from __future__ import annotations

from openai import OpenAI
from pydantic import BaseModel

from lume.catalog.models import NormalizedProduct
from lume.config import CONFIG, require_openai_key

_SYSTEM = """\
Sei un assistente esperto di beauty e profumeria per Lumé, un rivenditore italiano di lusso.
Il tuo compito è riordinare un elenco di prodotti in base alla query del cliente e alle sue preferenze.

Regole:
- Restituisci SOLO gli id dei prodotti nell'ordine di rilevanza decrescente.
- Non includere prodotti non presenti nella lista di input.
- Tieni conto delle istruzioni di filtro morbido (brand preferiti/scartati, famiglia olfattiva, genere, occasione).
- NON escludere prodotti: restituisci sempre tutti gli id ricevuti, solo riordinati.
"""


class _RerankItem(BaseModel):
    product_id: str
    why: str


class _RerankResponse(BaseModel):
    ranked: list[_RerankItem]


def rerank(
    query: str,
    candidates: list[NormalizedProduct],
    *,
    soft_instructions: str = "",
    user_profile_summary: str = "",
    top_k: int = 8,
) -> list[NormalizedProduct]:
    """Re-order candidates with gpt-4o-mini structured output.

    Args:
        query: Original user message.
        candidates: Products after hard-filter constraints (up to 20).
        soft_instructions: Natural-language hints from Intent (brand, family, gender, must_avoid).
        user_profile_summary: Short prose from redact_for_prompt(); empty if no profile.
        top_k: Return at most this many products.

    Returns:
        Re-ordered and truncated list of NormalizedProduct.
    """
    if not candidates:
        return []

    client = OpenAI(api_key=require_openai_key())

    product_lines = "\n".join(
        f"- {p.product_id}: {p.title} | {p.display_price} | "
        f"{'disponibile' if p.effective_available else 'non disponibile'} | "
        f"{'tester' if p.is_tester else ''}"
        for p in candidates
    )

    user_content_parts = [f"Query cliente: {query}"]
    if soft_instructions:
        user_content_parts.append(f"Preferenze/filtri: {soft_instructions}")
    if user_profile_summary:
        user_content_parts.append(f"Profilo utente: {user_profile_summary}")
    user_content_parts.append(f"\nProdotti da riordinare:\n{product_lines}")

    response = client.beta.chat.completions.parse(
        model=CONFIG.models.intent,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": "\n".join(user_content_parts)},
        ],
        response_format=_RerankResponse,
        temperature=0,
    )

    reranked = response.choices[0].message.parsed
    if reranked is None:
        return candidates[:top_k]

    id_to_product = {p.product_id: p for p in candidates}
    ordered = [
        id_to_product[item.product_id]
        for item in reranked.ranked
        if item.product_id in id_to_product
    ]
    # Append any that the LLM dropped (shouldn't happen, but defensive)
    seen = {p.product_id for p in ordered}
    ordered += [p for p in candidates if p.product_id not in seen]
    return ordered[:top_k]
