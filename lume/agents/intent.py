"""Intent extraction agents.

Two focused agents:
  extract_intent — fresh extraction from scratch (first turn or new_topic)
  refine_intent  — updates an existing intent with new information
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from lume.config import CONFIG, require_openai_key
from lume.memory.profile import UserProfile, redact_for_prompt

if TYPE_CHECKING:
    from lume.catalog.models import NormalizedProduct


# ── Intent schema ─────────────────────────────────────────────────────────────

class Intent(BaseModel):
    language: Literal["it", "en"] = "it"
    categories: list[str] = Field(default_factory=list)
    budget_max: float | None = None
    fragrance_family: list[str] = Field(default_factory=list)
    gender_lean: Literal["uomo", "donna", "unisex"] | None = None
    occasion: str | None = None
    gift_recipient: str | None = None
    must_avoid: list[str] = Field(default_factory=list)
    must_include: list[str] = Field(default_factory=list)
    niche_preference: bool | None = None
    escalate: bool = False
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    missing_critical_fields: list[str] = Field(default_factory=list)
    bm25_weight: float = Field(1.0, gt=0)
    vector_weight: float = Field(1.0, gt=0)


# ── Retrieval query builder ───────────────────────────────────────────────────

def intent_to_query(intent: Intent, topic_messages: list[str] | None = None) -> str:
    """Build a retrieval query from structured intent fields + raw topic messages.

    Structured fields give precise semantic signal; raw messages preserve brand
    names, specific terms, and nuances the LLM may not have extracted into the
    intent. Together they give the best BM25 + vector coverage.

    budget_max and must_avoid are hard filters handled elsewhere — excluded here.
    """
    parts: list[str] = []
    parts.extend(intent.categories)
    parts.extend(intent.fragrance_family)
    if intent.gender_lean:
        parts.append(intent.gender_lean)
    if intent.occasion:
        parts.append(intent.occasion)
    if intent.gift_recipient:
        parts.append(f"regalo per {intent.gift_recipient}")
    if intent.niche_preference:
        parts.append("nicchia")
    parts.extend(intent.must_include)
    if topic_messages:
        parts.extend(topic_messages)
    return " ".join(filter(None, parts))


# ── Shared extraction rules (injected into both prompts) ─────────────────────

_EXTRACTION_RULES = """\
─── Regole di estrazione ────────────────────────────────────────────────────────
categories: slug italiani: "profumo", "crema-viso", "make-up", "skincare", "corpo", "set-regalo".
budget_max: solo se ESPLICITAMENTE menzionato. None altrimenti.
fragrance_family: floreale, legnoso, orientale, agrumato, muschiato, speziato,
  acquatico, gourmand, cipriato, verde, fougère.
gender_lean: "uomo"/"donna"/"unisex" se esplicito o fortemente implicito.
must_avoid: cumulativo. "no oud" + ["cuoio"] già presenti → ["cuoio","oud"].
escalate: true per reso/rimborso/ordine/B2B/frustrazione forte.
missing_critical_fields: ["budget_max"] se categories=["profumo"] e budget_max mancante.
  Aggiungi "gender_lean" se categories=["profumo"] e gender_lean mancante.
  Aggiungi "budget_max" se gift_recipient != null e budget_max mancante.
  (Dopo 2 clarify_question questi campi vengono ignorati dal sistema — non è un errore.)
bm25_weight / vector_weight:
  Brand o prodotto specifico menzionato → bm25_weight=2.0, vector_weight=1.0
  Solo umore/occasione/sensazione → bm25_weight=1.0, vector_weight=2.0
  Misto → entrambi 1.0
confidence: 1.0 se intent completo. 0.5-0.7 se assi morbidi aperti. 0.3-0.5 se categoria vaga.

─── Cue implicite ────────────────────────────────────────────────────────────────
"più luxury" → niche_preference=true
"per mia madre / le piacciono i fiori" → gift_recipient="madre", fragrance_family=["floreale"]
"qualcosa per l'estate" → occasion="estate", fragrance_family=["agrumato","acquatico"]
"di nicchia" → niche_preference=true
"senza oud" / "no oud" → must_avoid aggiungi "oud"
"""


# ── extract_intent ────────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = f"""\
Sei l'agente di estrazione intent per Lumé, un rivenditore italiano di beauty e profumeria.
Ricevi un messaggio di un cliente (primo turno o nuova ricerca indipendente).
Estrai quante più informazioni possibili dal messaggio e restituisci un Intent strutturato.
NON generare testo per il cliente.

{_EXTRACTION_RULES}"""


def extract_intent(
    message: str,
    user_profile: UserProfile | None = None,
) -> Intent:
    """Extract a fresh Intent from a first-turn or new_topic message."""
    system = _EXTRACT_SYSTEM
    if user_profile:
        summary = redact_for_prompt(user_profile)
        if summary:
            system += f"\n\nProfilo utente noto (usa come prior se il messaggio non specifica):\n{summary}"

    client = OpenAI(api_key=require_openai_key())
    response = client.beta.chat.completions.parse(
        model=CONFIG.models.intent,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": message},
        ],
        response_format=Intent,
        temperature=0,
    )
    result = response.choices[0].message.parsed
    return result if result is not None else Intent()


# ── refine_intent ─────────────────────────────────────────────────────────────

_REFINE_SYSTEM = f"""\
Sei l'agente di raffinamento intent per Lumé.
Il cliente ha già un intent in corso e sta fornendo nuove informazioni o correzioni.
Aggiorna l'intent esistente con le nuove informazioni. Mantieni i campi non menzionati.
must_avoid è SEMPRE cumulativo (aggiungi, non sostituire).
NON generare testo per il cliente.

{_EXTRACTION_RULES}"""


def refine_intent(
    message: str,
    previous_intent: Intent,
    last_shown: list[NormalizedProduct] | None = None,
    user_profile: UserProfile | None = None,
) -> Intent:
    """Update an existing Intent with new information from a refinement message."""
    context_parts = [f"Intent corrente:\n{previous_intent.model_dump_json(indent=2)}"]

    if last_shown:
        lines = "\n".join(
            f"  {p.product_id}: {p.title} | {p.display_price}"
            for p in last_shown
        )
        context_parts.append(f"Prodotti mostrati nell'ultimo turno:\n{lines}")

    if user_profile:
        summary = redact_for_prompt(user_profile)
        if summary:
            context_parts.append(f"Profilo utente: {summary}")

    context = "\n\n".join(context_parts)
    user_content = f"{context}\n\nNuovo messaggio cliente: {message}"

    client = OpenAI(api_key=require_openai_key())
    response = client.beta.chat.completions.parse(
        model=CONFIG.models.intent,
        messages=[
            {"role": "system", "content": _REFINE_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        response_format=Intent,
        temperature=0,
    )
    result = response.choices[0].message.parsed
    return result if result is not None else previous_intent
