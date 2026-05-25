"""LLM-based retrieval query generator.

Replaces the deterministic intent_to_query() for node_retrieve.
Uses the full topic_history (user + assistant turns) to build a richer
BM25 + vector query that captures brand names, references to shown
products, and cross-turn context that a deterministic builder would miss.
"""

from __future__ import annotations

from openai import OpenAI
from pydantic import BaseModel

from lume.agents.intent import Intent
from lume.config import CONFIG, require_openai_key


_SYSTEM = """\
Sei un assistente per la ricerca di prodotti nel catalogo Lumé (beauty e profumeria italiana).
Data la conversazione recente e l'intent estratto, genera una query ottimizzata per ricerca ibrida BM25+vettoriale.

Regole:
- Includi: brand specifici, note olfattive, categoria prodotto, occasione, stile, nomi prodotti citati
- Se l'utente fa riferimento a prodotti mostrati in precedenza ("il secondo", "quello floreale", "simile a quello"),
  estrai le caratteristiche di quel prodotto dalla conversazione e includile nella query
- Escludi: budget, disponibilità, negazioni — gestiti da filtri separati
- Lingua output: italiano
- Formato: termini chiave separati da spazio, max 25 parole — NON una frase intera
"""


class _QueryResponse(BaseModel):
    query: str


def generate_retrieval_query(
    topic_history: list[dict],
    intent: Intent,
) -> str:
    """Generate a BM25+vector retrieval query from conversation history + intent.

    Falls back to a deterministic query if the LLM call fails.
    """
    history_text = ""
    if topic_history:
        recent = topic_history[-6:]  # last 3 exchanges
        lines = []
        for msg in recent:
            role = "Cliente" if msg["role"] == "user" else "Lumé"
            lines.append(f"{role}: {msg['content'][:200]}")
        history_text = "\n".join(lines)

    intent_parts: list[str] = []
    if intent.categories:
        intent_parts.append(f"categoria: {', '.join(intent.categories)}")
    if intent.fragrance_family:
        intent_parts.append(f"famiglia olfattiva: {', '.join(intent.fragrance_family)}")
    if intent.gender_lean:
        intent_parts.append(f"genere: {intent.gender_lean}")
    if intent.occasion:
        intent_parts.append(f"occasione: {intent.occasion}")
    if intent.must_include:
        intent_parts.append(f"deve includere: {', '.join(intent.must_include)}")
    if intent.niche_preference:
        intent_parts.append("nicchia")
    if intent.gift_recipient:
        intent_parts.append(f"regalo per {intent.gift_recipient}")

    user_content_parts: list[str] = []
    if history_text:
        user_content_parts.append(f"Conversazione:\n{history_text}")
    if intent_parts:
        user_content_parts.append(f"Intent: {', '.join(intent_parts)}")
    user_content_parts.append("Genera la query di ricerca.")

    try:
        client = OpenAI(api_key=require_openai_key())
        response = client.beta.chat.completions.parse(
            model=CONFIG.models.intent,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": "\n\n".join(user_content_parts)},
            ],
            response_format=_QueryResponse,
            temperature=0,
        )
        parsed = response.choices[0].message.parsed
        if parsed and parsed.query.strip():
            return parsed.query.strip()
    except Exception:
        pass

    # Deterministic fallback
    parts: list[str] = []
    parts.extend(intent.categories)
    parts.extend(intent.fragrance_family)
    if intent.gender_lean:
        parts.append(intent.gender_lean)
    if intent.occasion:
        parts.append(intent.occasion)
    parts.extend(intent.must_include)
    if topic_history:
        last_user = next(
            (m["content"] for m in reversed(topic_history) if m["role"] == "user"), ""
        )
        if last_user:
            parts.append(last_user[:80])
    return " ".join(filter(None, parts)) or "profumo"
