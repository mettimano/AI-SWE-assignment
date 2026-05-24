"""Responder agent — generates WhatsApp-style brand-voice messages.

Modes:
  answer           → 2–4 sentences citing 2–4 products
  clarify_question → 1-line opener wrapping pre-generated questions
  clarify_probe    → framing line for probe products (probes pre-selected by clarify agent)
  escalate         → empathetic 1-liner + needs_human=True
  no_match         → gentle apology ± closest OOS alternative
  specification    → details for a selected product (user picked from previous list)
"""

from __future__ import annotations

from openai import OpenAI
from pydantic import BaseModel, Field

from lume.agents.clarify import ClarifyPayload
from lume.agents.intent import Intent
from lume.catalog.models import NormalizedProduct
from lume.config import CONFIG, require_openai_key
from lume.memory.profile import UserProfile, redact_for_prompt
from lume.schemas import Recommendation, Reply


# ── LLM output schema ─────────────────────────────────────────────────────────

class _CitedItem(BaseModel):
    product_id: str
    why: str  # one Italian sentence, brand-voice compliant


class _ResponderOutput(BaseModel):
    reply_text: str
    cited: list[_CitedItem] = Field(default_factory=list)


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = """\
Sei il responder di Lumé, un rivenditore italiano di beauty e profumeria di lusso.
Generi messaggi WhatsApp brevi, caldi, professionali.
NON inventare prodotti, note, ingredienti, disponibilità o prezzi — usa solo i dati forniti.

════ BRAND VOICE ════════════════════════════════════════════════════════════════
• Tono: caldo e competente — come un consulente senior di profumeria
• Lingua: quella del cliente (italiano → italiano, inglese → inglese)
• Pronome: "noi" — parli a nome del negozio
• Emoji: max 1, solo se naturale (✨ 🌸) — mai tre di fila
• Formattazione: testo piano WhatsApp — niente **, -, #, bullet list, markdown
• Vocabolario olfattivo: note di testa/cuore/fondo, floreale, legnoso, ambrato,
  agrumato, muschiato, orientale, fougère, gourmand, acquatico

════ REGOLE HARD ════════════════════════════════════════════════════════════════
• Mai raccomandare come prima scelta un prodotto out-of-stock
• Tester: segnalali sempre ("tester — senza scatola originale, prezzo ridotto")
• Budget = filtro rigido. Non superarlo mai, neanche di €2
• 2–4 raccomandazioni, non di più
• Mai inventare note olfattive non presenti nella descrizione

════ ISTRUZIONI PER MODALITÀ ════════════════════════════════════════════════════

answer:
  Presenta 2-4 prodotti in modo fluido. Per ognuno: nome brand + prodotto, prezzo,
  1 motivo concreto. Chiudi con una frase di invito ("Vuoi più dettagli su uno?").
  Lunghezza: 2-4 frasi totali.

clarify_question:
  Apri con il framing fornito. Integra le domande in modo naturale nel testo —
  non come lista puntata. Lunghezza: 1-2 frasi.

clarify_probe:
  Usa il framing fornito come apertura. Presenta i prodotti probe come "assaggi
  di stile diversi" e invita il cliente a scegliere quello più vicino al suo gusto.
  Lunghezza: 2-3 frasi.

escalate:
  1 frase empatica + "Passo la richiesta a un nostro operatore". Nessun prodotto.

no_match:
  Scuse brevi (1 frase). Se c'è un'alternativa OOS fornita, menzionala come
  "non disponibile al momento ma potrebbe tornare presto". Suggerisci di
  allargare i criteri o di contattarci per disponibilità futura.

specification:
  2-3 frasi sui dettagli del prodotto scelto (note olfattive se presente,
  ingredienti chiave, varianti e prezzi). Chiudi con "Come posso aiutarti ancora?".

════ FORMATO OUTPUT ════════════════════════════════════════════════════════════
Restituisci JSON:
{
  "reply_text": "...",    // messaggio WhatsApp completo, testo piano
  "cited": [              // solo per answer e specification
    {"product_id": "...", "why": "..."}
  ]
}
Per clarify_question, clarify_probe, escalate, no_match → cited può essere [].
"""

# ── Few-shot examples ─────────────────────────────────────────────────────────

_FEW_SHOT: list[dict] = [
    # answer — floral gift
    {
        "role": "user",
        "content": (
            "MODALITÀ: answer\n"
            "MESSAGGIO: regalo per mia madre, le piacciono i fiori, budget 80€\n"
            "PRODOTTI DISPONIBILI:\n"
            "- p_031 | Cabotine Rose EDP 100ml | €32 | disponibile | floreale, rosa, biancospino\n"
            "- p_010 | Atyab Violet EDP 50ml | €65 | disponibile | violetta, muschio bianco\n"
            "- p_073 | Guerlain Mon Guerlain EDP 50ml | €79 | disponibile | lavanda, vaniglia, note fiorite\n"
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"reply_text": "Per tua madre con il gusto per i fiori abbiamo tre belle opzioni: '
            "Cabotine Rose (€32) — rosa e biancospino, fresca e femminile; "
            "Atyab Violet (€65) — violetta su fondo muschiato, più avvolgente; "
            "e il Guerlain Mon Guerlain (€79) — lavanda e vaniglia con cuore fiorito, molto elegante. "
            'Vuoi più dettagli su uno?", '
            '"cited": ['
            '{"product_id": "p_031", "why": "Rosa e biancospino freschi — regalo floreale classico sotto i 40€"},'
            '{"product_id": "p_010", "why": "Violetta su muschio bianco — più raffinato e avvolgente"},'
            '{"product_id": "p_073", "why": "Lavanda e vaniglia con cuore fiorito — elegante e riconoscibile"}'
            "]}"
        ),
    },
    # clarify_probe — ambiguous style
    {
        "role": "user",
        "content": (
            "MODALITÀ: clarify_probe\n"
            "MESSAGGIO: cerco qualcosa di buono per me, un profumo\n"
            "FRAMING: Per capire meglio il tuo gusto, ho scelto tre profumi con caratteri diversi —\n"
            "PROBE PRODOTTI:\n"
            "- p_010 | Atyab Violet | floreale\n"
            "- p_055 | Acqua di Parma Colonia | agrumato\n"
            "- p_088 | Tom Ford Oud Wood | legnoso-speziato\n"
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"reply_text": "Per capire meglio il tuo gusto, ho scelto tre profumi con caratteri diversi: '
            "Atyab Violet (floreale, violetta e muschio), Acqua di Parma Colonia (agrumato, classico italiano), "
            "Tom Ford Oud Wood (legnoso e speziato, più intenso). "
            'Quale si avvicina di più a quello che cerchi? ✨", "cited": []}'
        ),
    },
    # escalate
    {
        "role": "user",
        "content": (
            "MODALITÀ: escalate\n"
            "MESSAGGIO: ho ricevuto il pacco rotto, voglio un rimborso immediato\n"
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"reply_text": "Ci dispiace molto per l\'inconveniente — passo subito la tua richiesta '
            'a un nostro operatore che ti contatterà al più presto.", "cited": []}'
        ),
    },
    # no_match
    {
        "role": "user",
        "content": (
            "MODALITÀ: no_match\n"
            "MESSAGGIO: cerco un profumo under 20€\n"
            "OOS ALTERNATIVA: p_003 | Some Brand EDP | €22 | esaurito\n"
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"reply_text": "Al momento non abbiamo profumi disponibili sotto i 20€. '
            "Il Some Brand EDP (€22) si avvicina di più ma è al momento esaurito. "
            'Se puoi allargare leggermente il budget o sei interessato a quando torna disponibile, fammelo sapere!", "cited": []}'
        ),
    },
]


# ── Public API ────────────────────────────────────────────────────────────────

def generate_response(
    message: str,
    mode: str,
    intent: Intent,
    candidates: list[NormalizedProduct],
    oos_fallback: list[NormalizedProduct],
    user_profile: UserProfile | None = None,
    clarify_payload: ClarifyPayload | None = None,
) -> Reply:
    """Generate a WhatsApp-style brand-voice reply for the given mode."""
    user_content = _build_user_content(
        message=message,
        mode=mode,
        candidates=candidates,
        oos_fallback=oos_fallback,
        user_profile=user_profile,
        clarify_payload=clarify_payload,
    )

    client = OpenAI(api_key=require_openai_key())
    messages: list[dict] = [{"role": "system", "content": _SYSTEM}]
    messages.extend(_FEW_SHOT)
    messages.append({"role": "user", "content": user_content})

    response = client.beta.chat.completions.parse(
        model=CONFIG.models.responder,
        messages=messages,
        response_format=_ResponderOutput,
        temperature=0.4,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        parsed = _ResponderOutput(reply_text="Mi dispiace, si è verificato un errore. Riprova tra poco.")

    return _build_reply(parsed, mode, candidates, oos_fallback, intent)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_user_content(
    message: str,
    mode: str,
    candidates: list[NormalizedProduct],
    oos_fallback: list[NormalizedProduct],
    user_profile: UserProfile | None,
    clarify_payload: ClarifyPayload | None,
) -> str:
    parts: list[str] = [f"MODALITÀ: {mode}", f"MESSAGGIO: {message}"]

    if candidates:
        lines = ["\nPRODOTTI DISPONIBILI:"]
        for p in candidates[:6]:  # cap to avoid context bloat
            availability = "disponibile" if p.effective_available else "esaurito"
            desc_snippet = (p.cleaned_description or "")[:120].replace("\n", " ")
            lines.append(
                f"- {p.product_id} | {p.title} | {p.display_price} | {availability}"
                + (f" | {desc_snippet}" if desc_snippet else "")
            )
        parts.append("\n".join(lines))

    if oos_fallback and mode == "no_match":
        lines = ["\nOOS ALTERNATIVA:"]
        for p in oos_fallback[:2]:
            lines.append(f"- {p.product_id} | {p.title} | {p.display_price} | esaurito")
        parts.append("\n".join(lines))

    if clarify_payload:
        if clarify_payload.framing:
            parts.append(f"\nFRAMING: {clarify_payload.framing}")
        if clarify_payload.questions:
            qs = " ".join(q.text for q in clarify_payload.questions)
            parts.append(f"DOMANDE DA INCORPORARE: {qs}")
        if clarify_payload.probes:
            lines = ["\nPROBE PRODOTTI:"]
            for pr in clarify_payload.probes:
                lines.append(f"- {pr.product_id} | {pr.title} | {pr.axis_value}")
            parts.append("\n".join(lines))

    if user_profile:
        summary = redact_for_prompt(user_profile)
        if summary:
            parts.append(f"\nPROFILO CLIENTE (usa per personalizzare, non menzionare esplicitamente): {summary}")

    return "\n".join(parts)


def _build_reply(
    parsed: _ResponderOutput,
    mode: str,
    candidates: list[NormalizedProduct],
    oos_fallback: list[NormalizedProduct],
    intent: Intent,
) -> Reply:
    id_map = {p.product_id: p for p in candidates + oos_fallback}
    recommendations: list[Recommendation] = []
    for item in parsed.cited:
        product = id_map.get(item.product_id)
        if product is None:
            continue
        recommendations.append(
            Recommendation(
                product_id=product.product_id,
                title=product.title,
                price_display=product.display_price,
                available=product.effective_available,
                why=item.why,
            )
        )

    return Reply(
        mode=mode,  # type: ignore[arg-type]
        reply_text=parsed.reply_text,
        recommendations=recommendations,
        needs_human=(mode == "escalate"),
        intent=intent,
        debug={},
    )
