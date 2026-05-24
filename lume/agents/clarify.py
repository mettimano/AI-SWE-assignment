"""Clarify agent — decides ask-vs-probe and generates the clarification payload."""

from __future__ import annotations

from openai import OpenAI
from pydantic import BaseModel, Field

from lume.agents.intent import Intent
from lume.catalog.models import NormalizedProduct
from lume.config import CONFIG, require_openai_key

# ── Output schemas ────────────────────────────────────────────────────────────


class ClarifyQuestion(BaseModel):
    text: str


class ProbeProduct(BaseModel):
    product_id: str
    title: str
    axis_value: str  # e.g. "floreale", "legnoso", "agrumato"


class ClarifyPayload(BaseModel):
    mode: str  # "clarify_question" | "clarify_probe"
    questions: list[ClarifyQuestion] = Field(default_factory=list)   # max 2
    probes: list[ProbeProduct] = Field(default_factory=list)         # 2–4
    framing: str = ""  # one-line message to show the user


# ── Decision rule ─────────────────────────────────────────────────────────────

# Hard-filter fields whose absence forces an explicit question.
_CRITICAL_FIELDS = {"budget_max", "categories"}


def _needs_question(intent: Intent) -> bool:
    return bool(set(intent.missing_critical_fields) & _CRITICAL_FIELDS)


def _needs_probe(intent: Intent) -> bool:
    """True when hard filters are known but soft axes (family/occasion) are still open."""
    return (
        not _needs_question(intent)
        and intent.confidence < CONFIG.clarify.soft_ambiguity_threshold
        and bool(intent.categories)
    )


# ── Question writer ────────────────────────────────────────────────────────────

_QUESTION_SYSTEM = """\
Sei un assistente WhatsApp per Lumé, un rivenditore beauty italiano.
Il cliente ha fatto una richiesta vaga. Scrivi uan domanda breve, naturale e professionale nella lingua in cui l'utente ti scrive
per capire meglio cosa cerca. Tono caldo, colloquiale, senza elenchi puntati.
Rispondi con un JSON { "questions": ["domanda 1"] }.
"""


class _QuestionResponse(BaseModel):
    questions: list[str]


def _generate_questions(intent: Intent, language: str) -> list[ClarifyQuestion]:
    missing = intent.missing_critical_fields
    lang_note = " Rispondi in inglese." if language == "en" else ""
    prompt = (
        f"Categoria richiesta: {intent.categories or 'non specificata'}.\n"
        f"Campi mancanti: {missing}.\n"
        f"Scrivi le domande necessarie.{lang_note}"
    )
    client = OpenAI(api_key=require_openai_key())
    response = client.beta.chat.completions.parse(
        model=CONFIG.models.intent,
        messages=[
            {"role": "system", "content": _QUESTION_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        response_format=_QuestionResponse,
        temperature=0.3,
    )
    parsed = response.choices[0].message.parsed
    if not parsed:
        return []
    return [ClarifyQuestion(text=q) for q in parsed.questions[: CONFIG.clarify.max_questions]]


# ── Probe selector ─────────────────────────────────────────────────────────────

_PROBE_SYSTEM = """\
Sei un assistente WhatsApp per Lumé.
Il cliente è interessato a una categoria ma non ha specificato la famiglia olfattiva o lo stile.
Seleziona da 2 a 4 prodotti dall'elenco che coprono ASSI DIVERSI della categoria richiesta
(es. uno floreale, uno legnoso, uno agrumato) così il cliente può scegliere lo stile che preferisce.
Scrivi una riga di presentazione breve e naturale in italiano (max 1 frase).
Rispondi con JSON { "framing": "...", "probes": [{"product_id": "...", "axis_value": "..."}, ...] }.
"""


class _ProbeItem(BaseModel):
    product_id: str
    axis_value: str


class _ProbeResponse(BaseModel):
    framing: str
    probes: list[_ProbeItem]


def _select_probes(
    intent: Intent,
    candidates: list[NormalizedProduct],
) -> tuple[str, list[ProbeProduct]]:
    if not candidates:
        return "", []

    product_lines = "\n".join(
        f"- {p.product_id}: {p.title} | {p.display_price} | "
        f"collezioni: {', '.join(p.raw_collections[:4])}"
        for p in candidates[: CONFIG.clarify.probe_max * 3]  # give LLM some headroom
    )
    prompt = (
        f"Categoria: {intent.categories}.\n"
        f"Famiglia olfattiva nota: {intent.fragrance_family or 'nessuna'}.\n\n"
        f"Prodotti disponibili:\n{product_lines}"
    )
    client = OpenAI(api_key=require_openai_key())
    response = client.beta.chat.completions.parse(
        model=CONFIG.models.intent,
        messages=[
            {"role": "system", "content": _PROBE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        response_format=_ProbeResponse,
        temperature=0.3,
    )
    parsed = response.choices[0].message.parsed
    if not parsed:
        return "", []

    id_to_product = {p.product_id: p for p in candidates}
    probes = [
        ProbeProduct(
            product_id=item.product_id,
            title=id_to_product[item.product_id].title if item.product_id in id_to_product else item.product_id,
            axis_value=item.axis_value,
        )
        for item in parsed.probes
        if item.product_id in id_to_product
    ][: CONFIG.clarify.probe_max]

    return parsed.framing, probes


# ── Public API ─────────────────────────────────────────────────────────────────


def build_clarify_payload(
    intent: Intent,
    candidates: list[NormalizedProduct],
) -> ClarifyPayload:
    """Decide ask-vs-probe and generate the clarification payload.

    Decision rule:
      - Any missing_critical_field in {budget_max, gender_lean, categories} → ask (max 2 Qs)
      - confidence < threshold AND categories known → probe (2–4 spanning products)
    """
    if _needs_question(intent):
        questions = _generate_questions(intent, intent.language)
        framing = (
            "Certo! Per aiutarti al meglio ho bisogno di qualche info in più 😊"
            if intent.language == "it"
            else "Of course! Just a couple of quick questions to help you better 😊"
        )
        return ClarifyPayload(
            mode="clarify_question",
            questions=questions,
            framing=framing,
        )

    if _needs_probe(intent):
        framing, probes = _select_probes(intent, candidates)
        if probes:
            return ClarifyPayload(
                mode="clarify_probe",
                probes=probes,
                framing=framing,
            )

    # Fallback: treat as confident (caller should not reach here, but be defensive)
    return ClarifyPayload(mode="answer")
