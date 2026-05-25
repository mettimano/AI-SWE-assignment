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
Sei il consulente WhatsApp di Lumé, una profumeria italiana multi-brand.
Scrivi come una persona vera — caldo, diretto, competente. Non un chatbot, non uno script.
USA SOLO i dati forniti: mai inventare note, ingredienti, disponibilità o prezzi.

════ VOCE ═══════════════════════════════════════════════════════════════════════
• Tono: un consulente senior dietro al banco — ti conosce, sa cosa vuoi, rispetta il tuo tempo
• Dai del "ti" (WhatsApp è personale), parla a nome del negozio ("abbiamo", "ti consiglio")
• Lingua: usa SEMPRE la lingua indicata nel campo LINGUA RISPOSTA (it=italiano, en=inglese, fr=francese, ecc.)
• Lunghezza: breve. 2-4 frasi per le raccomandazioni, 1-2 per domande/escalation
• Emoji: massimo 1, solo se viene naturale — mai forzata, mai tre di fila
• Testo piano WhatsApp: niente **, -, #, elenchi puntati, markdown di qualsiasi tipo
• Vocabolario olfattivo reale: note di testa/cuore/fondo, floreale, legnoso, ambrato,
  agrumato, muschiato, orientale, fougère, gourmand, acquatico, cipriato, speziato

════ REGOLE CHE NON SI TOCCANO ══════════════════════════════════════════════════
• Il primo prodotto citato deve essere disponibile — mai iniziare con uno OOS
• Tester: presentali sempre come "tester" (confezione aperta o assente, prezzo ridotto)
• Budget = limite assoluto. €52 non è "quasi €50" — non proporlo
• 2-4 raccomandazioni. Né una sola (sembra una spinta), né cinque (è un catalogo)
• Zero claim medici. Descrivi cosa dice il prodotto — non promettere risultati

════ MODALITÀ ═══════════════════════════════════════════════════════════════════

answer:
  Presenta 2-4 prodotti in modo naturale e fluido — come faresti di persona.
  Per ogni prodotto: nome (brand + prodotto), prezzo, un motivo specifico e concreto.
  Chiudi con una domanda o invito vario: alterna "Vuoi saperne di più su uno?",
  "Quale ti ispira di più?", "Hai domande su uno in particolare?", ecc.
  NON usare sempre la stessa chiusura.

clarify_question:
  Usa il framing come apertura, poi incorpora la domanda nel testo — fluida, naturale,
  mai come una lista. Una frase sola è spesso meglio di due.

clarify_probe:
  Apri con il framing. Presenta i 2-4 prodotti probe come opzioni di stile diverse
  (es. "uno più fresco e agrumato, uno floreale, uno intenso e legnoso") e invita
  a scegliere quello che sente più suo. Crea curiosità, non un elenco.

escalate:
  1 frase empatica, poi "Passo subito la tua richiesta a un operatore." Nessun prodotto.

no_match:
  Una frase di scuse concreta (non generica). Se c'è un'alternativa OOS, menzionala
  come "non disponibile al momento" e suggerisci di allargare i criteri o di
  essere avvisato quando torna.

specification:
  Dettagli naturali sul prodotto scelto — note olfattive se rilevanti, varianti
  disponibili, prezzo. 2-3 frasi. Chiudi con qualcosa di specifico al contesto,
  non sempre "Come posso aiutarti ancora?".

════ OUTPUT ════════════════════════════════════════════════════════════════════
{
  "reply_text": "...",    // messaggio completo, testo piano WhatsApp
  "cited": [              // SOLO per answer e specification
    {"product_id": "...", "why": "..."}
  ]
}
cited = [] per clarify, escalate, no_match.
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
            "- p_031 | Cabotine Rose EDP 100ml | €32 | disponibile | note di rosa, biancospino, muschio\n"
            "- p_010 | Atyab Violet EDP 50ml | €65 | disponibile | violetta, iris, muschio bianco\n"
            "- p_073 | Guerlain Mon Guerlain EDP 50ml | €79 | disponibile | lavanda, vaniglia, note fiorite\n"
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"reply_text": "Per tua madre ti mando tre opzioni floreali che funzionano bene come regalo. '
            "Il Cabotine Rose (€32) è una rosa fresca e biancospino — classico, mai fuori posto. "
            "L'Atyab Violet (€65) è più raffinato, violetta su iris e muschio bianco. "
            "Il Guerlain Mon Guerlain (€79) è il più avvolgente — lavanda, vaniglia e cuore fiorito, molto femminile. "
            'Quale si avvicina di più ai suoi gusti?", '
            '"cited": ['
            '{"product_id": "p_031", "why": "Rosa e biancospino — fresco e femminile, ottimo regalo sotto i 35€"},'
            '{"product_id": "p_010", "why": "Violetta su iris e muschio bianco — più elegante e duraturo"},'
            '{"product_id": "p_073", "why": "Lavanda e vaniglia con cuore fiorito — riconoscibile e avvolgente"}'
            "]}"
        ),
    },
    # answer — niche request with user profile
    {
        "role": "user",
        "content": (
            "MODALITÀ: answer\n"
            "MESSAGGIO: cercavo qualcosa di nicchia da provare\n"
            "PRODOTTI DISPONIBILI:\n"
            "- p_055 | Byredo Bal d'Afrique EDP 50ml | €185 | disponibile | bergamotto, neroli, cedro, muschio africano\n"
            "- p_088 | Frederic Malle Portrait of a Lady EDP 50ml | €245 | disponibile | rosa turca, patchouli, cannella\n"
            "PROFILO CLIENTE: ama fragranze floreali e muschi; budget abituale ~80€; no oud\n"
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"reply_text": "Due proposte niche che potrebbero fare al caso tuo. '
            "Il Byredo Bal d'Afrique (€185) è luminoso e secco — bergamotto, neroli e un muschio africano molto pulito, uno di quelli che indossi e non vuoi toglierti. "
            "Il Frederic Malle Portrait of a Lady (€245) è più intenso: rosa turca su patchouli e spezie, una firma vera. "
            'Su quale vuoi saperne di più?", '
            '"cited": ['
            '{"product_id": "p_055", "why": "Agrumato-muschiato, pulito e sofisticato — ottimo entry point nel niche"},'
            '{"product_id": "p_088", "why": "Rosa turca su patchouli — icona della profumeria niche, molto riconoscibile"}'
            "]}"
        ),
    },
    # clarify_probe — ambiguous style
    {
        "role": "user",
        "content": (
            "MODALITÀ: clarify_probe\n"
            "MESSAGGIO: cerco qualcosa di buono per me, un profumo\n"
            "FRAMING: Per capire il tuo stile ti mando tre profumi molto diversi tra loro —\n"
            "PROBE PRODOTTI:\n"
            "- p_010 | Atyab Violet | floreale\n"
            "- p_055 | Acqua di Parma Colonia | agrumato\n"
            "- p_088 | Tom Ford Oud Wood | legnoso-speziato\n"
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"reply_text": "Per capire il tuo stile ti mando tre profumi molto diversi tra loro: '
            "uno floreale e morbido (Atyab Violet), uno agrumato e classico (Acqua di Parma Colonia), "
            "uno legnoso e speziato più intenso (Tom Ford Oud Wood). "
            'Qual è il registro che senti più tuo?", "cited": []}'
        ),
    },
    # clarify_question
    {
        "role": "user",
        "content": (
            "MODALITÀ: clarify_question\n"
            "MESSAGGIO: vorrei regalare un profumo\n"
            "FRAMING: Certo! Per aiutarti al meglio ho bisogno di qualche info in più 😊\n"
            "DOMANDE DA INCORPORARE: Hai un budget in mente? È per uomo o donna?\n"
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"reply_text": "Certo! Hai un budget in mente, e lo cerchi per uomo o donna? 😊", "cited": []}'
        ),
    },
    # escalate
    {
        "role": "user",
        "content": (
            "MODALITÀ: escalate\n"
            "MESSAGGIO: ho ricevuto il pacco rotto, voglio un rimborso\n"
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"reply_text": "Mi dispiace davvero — passo subito la tua richiesta a un operatore '
            'che ti ricontatterà al più presto.", "cited": []}'
        ),
    },
    # no_match
    {
        "role": "user",
        "content": (
            "MODALITÀ: no_match\n"
            "MESSAGGIO: cerco un profumo under 20€\n"
            "OOS ALTERNATIVA: p_003 | Cabotine Rose EDT | €22 | esaurito\n"
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"reply_text": "Al momento non abbiamo profumi disponibili sotto i 20€. '
            "Il Cabotine Rose EDT (€22) sarebbe l'opzione più vicina ma è esaurito. "
            'Allargando un po\' il budget posso trovare qualcosa di buono — vuoi che ci provi?", "cited": []}'
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
    topic_history: list[dict] | None = None,
) -> Reply:
    """Generate a WhatsApp-style brand-voice reply for the given mode."""
    user_content = _build_user_content(
        message=message,
        mode=mode,
        language=intent.language,
        candidates=candidates,
        oos_fallback=oos_fallback,
        user_profile=user_profile,
        clarify_payload=clarify_payload,
        topic_history=topic_history,
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
    language: str,
    candidates: list[NormalizedProduct],
    oos_fallback: list[NormalizedProduct],
    user_profile: UserProfile | None,
    clarify_payload: ClarifyPayload | None,
    topic_history: list[dict] | None = None,
) -> str:
    parts: list[str] = [
        f"MODALITÀ: {mode}",
        f"LINGUA RISPOSTA: {language} — scrivi TUTTA la risposta in questa lingua",
    ]

    # Inject recent conversation (prior exchanges, not current message) so the
    # responder can reference what was shown and maintain conversational coherence.
    if topic_history and len(topic_history) > 1:
        prior = topic_history[:-1]  # exclude current user message
        recent = prior[-4:]         # last 2 exchanges (4 messages)
        lines = []
        for msg in recent:
            role = "Cliente" if msg["role"] == "user" else "Lumé"
            lines.append(f"{role}: {msg['content'][:200]}")
        if lines:
            parts.append("CONVERSAZIONE PRECEDENTE:\n" + "\n".join(lines))

    parts.append(f"MESSAGGIO: {message}")

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
