"""LLM judge — 3-axis quality scoring for Lumé replies.

Axes (each 1–5 with rubric anchors):
  relevance    — do the recommended products match what the user asked for?
  brand_voice  — does the reply sound like a knowledgeable Italian beauty consultant?
  whatsapp_feel — does it feel like a natural WhatsApp message (short, personal, no markdown)?

For non-answer modes (clarify, escalate, no_match):
  relevance measures how well the reply addresses the user's immediate need.
  brand_voice and whatsapp_feel are always scored.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from lume.config import CONFIG, require_openai_key
from lume.schemas import Reply


# ── Output schema ─────────────────────────────────────────────────────────────

class JudgeScores(BaseModel):
    relevance: int = Field(ge=1, le=5, description="1=completely off-target, 5=perfectly relevant")
    brand_voice: int = Field(ge=1, le=5, description="1=robotic/generic, 5=warm expert consultant")
    whatsapp_feel: int = Field(ge=1, le=5, description="1=feels like formal email, 5=natural WhatsApp DM")
    reasoning: str = Field(description="2-3 sentence justification for the scores")


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = """\
Sei un valutatore esperto di assistenti conversazionali per il retail beauty.
Il tuo compito è valutare le risposte di un chatbot WhatsApp per una profumeria italiana chiamata Lumé.

Fornisci tre punteggi da 1 a 5 e una breve motivazione in italiano.

━━ RILEVANZA (relevance) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Misura quanto i prodotti consigliati (o la domanda di chiarimento) rispondono alla richiesta dell'utente.
1 = completamente fuori tema — prodotti sbagliati o irrilevanti
2 = parzialmente pertinente — alcuni prodotti giusti ma mancano aspetti chiave
3 = pertinente — prodotti ragionevoli ma non ottimali per il profilo richiesto
4 = molto pertinente — prodotti ben centrati, dettagli corretti
5 = perfetto — selezione ottimale che soddisfa tutti i vincoli espliciti e impliciti

━━ BRAND VOICE (brand_voice) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Misura se la risposta suona come un consulente senior di profumeria, competente e caldo.
1 = robotico, generico, sembra uno script automatico
2 = formale ma piatto, senza personalità
3 = corretto ma neutro — manca calore o voce distintiva
4 = caldo e competente — linguaggio olfattivo appropriato, tono personale
5 = eccellente — si sente il consulente dietro al banco, uso sicuro del lessico olfattivo italiano

━━ WHATSAPP FEEL (whatsapp_feel) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Misura se il messaggio sembra un DM WhatsApp naturale (breve, personale, senza markdown).
1 = sembra una email formale o un report — troppo lungo, liste puntate, intestazioni
2 = parzialmente adatto — alcune sezioni ok ma contiene elenchi o testo troppo formale
3 = accettabile — lunghezza ok ma stile leggermente rigido
4 = buono — breve, personale, nessun markdown visibile, tono giusto
5 = perfetto — 2-4 frasi naturali come quelle di una persona vera, zero markdown

━━ REGOLE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Sii critico ma equo. Usa tutta la scala — non dare 5 se non è davvero eccellente.
• Se la risposta è in modalità escalate o clarify, relevance misura quanto bene affronta il bisogno immediato.
• Rispondi sempre con il JSON richiesto.
"""


# ── Public API ────────────────────────────────────────────────────────────────

def judge_reply(
    message: str,
    reply: Reply,
    case_description: str,
) -> JudgeScores | None:
    """Score a reply on 3 axes using gpt-4o-mini structured output.

    Returns None if the API call fails or returns no parsed output.
    """
    from openai import OpenAI  # noqa: PLC0415

    product_lines = "\n".join(
        f"  - {r.product_id}: {r.title} | {r.price_display} | {'disponibile' if r.available else 'esaurito'}"
        for r in reply.recommendations
    ) or "  (nessun prodotto citato)"

    user_content = (
        f"Caso: {case_description}\n"
        f"Modalità: {reply.mode}\n\n"
        f"Messaggio utente: {message}\n\n"
        f"Risposta chatbot:\n{reply.reply_text}\n\n"
        f"Prodotti raccomandati:\n{product_lines}"
    )

    client = OpenAI(api_key=require_openai_key())
    try:
        response = client.beta.chat.completions.parse(
            model=CONFIG.models.judge,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_content},
            ],
            response_format=JudgeScores,
            temperature=0.0,
        )
        return response.choices[0].message.parsed
    except Exception:
        return None
