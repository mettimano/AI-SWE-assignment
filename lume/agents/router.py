"""Router LLM — decides what to do next each turn.

The router is the "brain": it sees the full conversation state and picks the next action.
LangGraph enforces hard constraints on top of the router's decision:
  - extract_intent cannot follow extract_intent directly
  - clarify_question is blocked after 2 rounds (graph forces "answer")
  - refine_intent requires an existing current_intent
  - selection requires valid last_shown_products
  - save_preferences is triggered automatically (not by the router)
  - guard runs automatically on every output (not by the router)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from lume.agents.intent import Intent
from lume.config import CONFIG, require_openai_key

if TYPE_CHECKING:
    from lume.catalog.models import NormalizedProduct


# ── Router output ─────────────────────────────────────────────────────────────

class RouterDecision(BaseModel):
    action: Literal[
        "extract_intent",
        "refine_intent",
        "clarify_question",
        "answer",
        "selection",
        "new_topic",
        "escalate",
        "chat",
        "compare",
    ]
    """
    extract_intent   → fresh intent extraction (first turn or completely new search)
    refine_intent    → update existing intent with new info (same ongoing search)
    clarify_question → ask user for missing critical info (max 2 times per intent session)
    answer           → run retrieval pipeline and return product recommendations
    selection        → user picked a product: show details + trigger save_preferences
    new_topic        → user started a new independent search (resets intent + clarify_count)
    escalate         → returns/refunds/B2B/frustration → human handoff
    chat             → off-topic / small talk / meta-question (no retrieval, polite redirect)
    compare          → explain or compare specific products (uses context, retrieves only if needed,
                       never proposes new alternatives the user didn't ask about)
    """

    selected_product_id: str | None = Field(
        None,
        description="Populated only for action='selection'. The product_id the user chose.",
    )
    reasoning: str = Field(
        "",
        description="One-line explanation of why this action was chosen (for debug).",
    )


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """\
Sei il router agent di Lumé, un rivenditore italiano di beauty e profumeria.
Ricevi il messaggio del cliente e lo stato della conversazione.
Il tuo compito è decidere quale azione eseguire. NON generi testo per il cliente.

════════ AZIONI DISPONIBILI ════════════════════════════════════════════════════

"extract_intent"
  Estrae un intent completamente nuovo dal messaggio corrente.
  Usa quando:
  • È il primo messaggio dell'utente.
  • L'utente inizia una ricerca COMPLETAMENTE NUOVA e INDIPENDENTE (vedi new_topic).
  • Dopo una clarify_question e l'utente ha fornito informazioni UTILI.
  NOTA: dopo aver chiamato extract_intent nel turno precedente, NON puoi chiamarlo di nuovo.

"refine_intent"
  Aggiorna l'intent esistente con nuove informazioni (stessa ricerca in corso).
  Usa quando l'utente:
  • Restringe o modifica i criteri ("più economico", "no oud", "per donna")
  • Risponde a una clarify_question con informazioni utili (se extract_intent è già stato usato)
  • Aggiunge contesto ("ah, è per un regalo")
  RICHIEDE: un intent esistente. Non usare come primo turno.

"clarify_question"
  Chiedi al cliente informazioni critiche mancanti.
  Usa quando intent.missing_critical_fields non è vuoto E non hai ancora fatto 2 domande.
  ATTENZIONE: il grafo blocca questa azione se clarify_count >= 2 e chiama "answer" automaticamente.

"answer"
  Esegui il retrieval e fornisci raccomandazioni di prodotti.
  Usa quando:
  • L'intent è sufficientemente chiaro (missing_critical_fields vuoto o confidence >= 0.6)
  • Hai già fatto 2 clarify_question (il grafo ti forza qui comunque)
  • L'utente ha risposto a una clarify ma non ha dato info utili

"selection"
  L'utente vuole DETTAGLI su uno specifico prodotto mostrato.
  Esempi puri: "il secondo", "quello floreale", "dimmi di più sul primo", "Chanel", "prendo il primo"
  Popola selected_product_id con il product_id corretto dall'elenco mostrato.
  ATTENZIONE: se il messaggio contiene "simile", "simili", "come questo", "qualcosa di simile"
  o altri segnali di "trovami altri prodotti dello stesso tipo" → usa "refine_intent", NON "selection".

"new_topic"
  L'utente inizia una ricerca NUOVA e INDIPENDENTE.
  Azzera l'intent e il contatore clarify. Poi il sistema chiamerà extract_intent.
  Usa "new_topic" (non "extract_intent") quando:
    a) La categoria cambia (profumo → makeup)
    b) Il messaggio è autonomo: ha senso SENZA conoscere la conversazione precedente
       Test: "questa frase funziona come prima domanda?" → Sì = new_topic
    c) Segnale esplicito: "ho un'altra richiesta", "cerco altro", "passiamo a X"
  NOTA: anche nella stessa categoria può essere new_topic se la richiesta è autonoma.

"escalate"
  Reso, rimborso, stato ordine, B2B, frustrazione intensa.
  Il sistema risponderà con un messaggio empatico e passerà a un operatore umano.
  NON usare escalate per domande off-topic o meta-domande — quelle vanno in "chat".

"chat"
  Domande off-topic, saluti, small talk, meta-domande sul sistema.
  Esempi: "ciao", "come stai?", "che ore sono?", "come si risolve un'equazione differenziale?",
  "chi sei?", "cosa puoi fare?", "grazie".
  Il responder risponde direttamente con un messaggio caldo, ricordando il proprio ruolo
  (assistente di profumeria) e re-indirizzando se è off-topic. NESSUN retrieval.
  Lo stato (intent, last_shown) NON viene modificato — l'utente può riprendere la ricerca dopo.

"compare"
  L'utente vuole confrontare o farsi spiegare prodotti specifici (in contesto o no).
  Esempi: "che differenza c'è tra il primo e il secondo?", "come differiscono X e Y?",
  "sono simili come note?", "spiegami meglio entrambi", "qual è più intenso?".
  Il sistema risponde usando i prodotti già in `last_shown` se sono pertinenti; se l'utente
  nomina prodotti non in contesto, il sistema fa un retrieval mirato per recuperarli e poi
  spiega. NON propone alternative nuove che l'utente non ha chiesto.
  DIVERSO da "selection" (dettagli su UN prodotto) e da "refine_intent" (cerca altri simili).

════════ LOGICA DI ROUTING ════════════════════════════════════════════════════

• Nessun intent precedente + richiesta shopping → "extract_intent"
• Messaggio off-topic / saluto / meta → "chat"
• Intent presente + utente vuole confrontare/spiegare prodotti → "compare"
• Intent presente + utente seleziona UN prodotto → "selection"
• Intent presente + utente cambia ricerca (autonoma) → "new_topic"
• Intent presente + utente raffina criteri → "refine_intent"
• Intent presente + info sufficienti → "answer"
• Info insufficienti + clarify_count < 2 → "clarify_question"
• Reso/rimborso/B2B/frustrazione → "escalate"

Ricorda: "new_topic" = il SEGNALE di reset. Il sistema si occupa poi di chiamare extract_intent.
"""

_FEW_SHOT: list[dict] = [
    # First turn
    {
        "role": "user",
        "content": (
            "STATO: intent=nessuno, clarify_count=0, last_shown=nessuno.\n"
            "MESSAGGIO: vorrei un regalo per mia madre, le piacciono i fiori, budget 80 euro"
        ),
    },
    {
        "role": "assistant",
        "content": '{"action":"extract_intent","selected_product_id":null,"reasoning":"Primo turno, nessun intent precedente."}',
    },
    # After extract, info sufficient → answer
    {
        "role": "user",
        "content": (
            "STATO: intent={categories:[profumo], budget_max:80, fragrance_family:[floreale], gender_lean:donna, missing_critical_fields:[]}, "
            "clarify_count=0, last_shown=nessuno.\n"
            "MESSAGGIO: (l'intent è già stato estratto, ora decidi cosa fare)"
        ),
    },
    {
        "role": "assistant",
        "content": '{"action":"answer","selected_product_id":null,"reasoning":"Intent completo, nessun campo critico mancante."}',
    },
    # After answer, user refines
    {
        "role": "user",
        "content": (
            "STATO: intent={categories:[profumo], budget_max:80, fragrance_family:[floreale]}, "
            "clarify_count=0, last_shown=[p_010, p_031, p_073].\n"
            "MESSAGGIO: qualcosa di più economico"
        ),
    },
    {
        "role": "assistant",
        "content": '{"action":"refine_intent","selected_product_id":null,"reasoning":"Stessa ricerca, nuovo vincolo di prezzo."}',
    },
    # After answer, user wants details on a specific product (pure selection)
    {
        "role": "user",
        "content": (
            "STATO: intent={categories:[profumo], budget_max:80}, "
            "clarify_count=0, last_shown=[p_010 Atyab Violet, p_031 Cabotine Rose, p_073 Patchouli Premier].\n"
            "MESSAGGIO: dimmi di più sul secondo"
        ),
    },
    {
        "role": "assistant",
        "content": '{"action":"selection","selected_product_id":"p_031","reasoning":"L\'utente vuole solo dettagli sul secondo prodotto (Cabotine Rose), nessun segnale di similarità."}',
    },
    # After answer, user selects AND wants similar products → refine_intent
    {
        "role": "user",
        "content": (
            "STATO: intent={categories:[profumo], budget_max:80}, "
            "clarify_count=0, last_shown=[p_010 Atyab Violet, p_031 Cabotine Rose, p_073 Patchouli Premier].\n"
            "MESSAGGIO: il secondo, voglio qualcosa di simile"
        ),
    },
    {
        "role": "assistant",
        "content": '{"action":"refine_intent","selected_product_id":null,"reasoning":"L\'utente indica il secondo prodotto come ancora stilistica e chiede altri simili — refine_intent aggiornerà l\'intent con quelle caratteristiche."}',
    },
    # New topic — same category, self-contained
    {
        "role": "user",
        "content": (
            "STATO: intent={categories:[profumo], budget_max:80, gender_lean:donna}, "
            "clarify_count=0, last_shown=[p_010, p_031, p_073].\n"
            "MESSAGGIO: cerco un profumo da uomo sotto i 50 euro come regalo"
        ),
    },
    {
        "role": "assistant",
        "content": '{"action":"new_topic","selected_product_id":null,"reasoning":"Richiesta autonoma e completa, cambia gender e contesto — nuova ricerca indipendente."}',
    },
    # Missing critical fields → clarify
    {
        "role": "user",
        "content": (
            "STATO: intent={categories:[profumo], missing_critical_fields:[budget_max, gender_lean]}, "
            "clarify_count=0, last_shown=nessuno.\n"
            "MESSAGGIO: (intent appena estratto da 'vorrei regalare un profumo')"
        ),
    },
    {
        "role": "assistant",
        "content": '{"action":"clarify_question","selected_product_id":null,"reasoning":"Mancano budget_max e gender_lean per una risposta precisa."}',
    },
    # Clarify during clarify — user changes topic
    {
        "role": "user",
        "content": (
            "STATO: intent={categories:[profumo], missing_critical_fields:[budget_max]}, "
            "clarify_count=1, last_shown=nessuno.\n"
            "MESSAGGIO: ma in realtà voglio cercare un fondotinta"
        ),
    },
    {
        "role": "assistant",
        "content": '{"action":"new_topic","selected_product_id":null,"reasoning":"Cambia categoria (profumo → make-up), nuova ricerca indipendente."}',
    },
    # Off-topic / meta — chat, no escalation
    {
        "role": "user",
        "content": (
            "STATO: intent={categories:[profumo], budget_max:80}, clarify_count=0, last_shown=[p_010, p_031, p_073].\n"
            "MESSAGGIO: come si risolve un'equazione differenziale?"
        ),
    },
    {
        "role": "assistant",
        "content": '{"action":"chat","selected_product_id":null,"reasoning":"Domanda off-topic non-shopping, non un trigger di escalation — il responder ringrazia e re-indirizza."}',
    },
    # Greeting at any point
    {
        "role": "user",
        "content": (
            "STATO: intent=nessuno, clarify_count=0, last_shown=nessuno.\n"
            "MESSAGGIO: ciao!"
        ),
    },
    {
        "role": "assistant",
        "content": '{"action":"chat","selected_product_id":null,"reasoning":"Saluto, non è ancora una richiesta shopping."}',
    },
    # Compare two products already shown
    {
        "role": "user",
        "content": (
            "STATO: intent={categories:[profumo]}, clarify_count=0, "
            "last_shown=[p_010 Atyab Violet, p_031 Cabotine Rose, p_073 Patchouli Premier].\n"
            "MESSAGGIO: che differenza c'è tra il primo e il secondo?"
        ),
    },
    {
        "role": "assistant",
        "content": '{"action":"compare","selected_product_id":null,"reasoning":"L\'utente vuole confrontare due prodotti già mostrati, nessuna nuova raccomandazione."}',
    },
    # Compare two products NOT in last_shown — needs retrieval
    {
        "role": "user",
        "content": (
            "STATO: intent={categories:[profumo]}, clarify_count=0, last_shown=nessuno.\n"
            "MESSAGGIO: come differiscono il Chanel Coco Mademoiselle e il Dior Sauvage?"
        ),
    },
    {
        "role": "assistant",
        "content": '{"action":"compare","selected_product_id":null,"reasoning":"L\'utente nomina prodotti specifici da confrontare ma non sono in contesto — il sistema farà retrieval mirato."}',
    },
]


# ── Public API ────────────────────────────────────────────────────────────────

def route(
    message: str,
    current_intent: Intent | None,
    last_shown: list[NormalizedProduct] | None,
    clarify_count: int,
    last_action: str | None,
    topic_history: list[dict] | None = None,
) -> RouterDecision:
    """Decide the next action. The graph enforces hard constraints on top of this."""
    # Build state summary for the LLM
    state_parts: list[str] = []

    if current_intent:
        state_parts.append(
            f"intent={{"
            f"categories:{current_intent.categories}, "
            f"budget_max:{current_intent.budget_max}, "
            f"fragrance_family:{current_intent.fragrance_family}, "
            f"gender_lean:{current_intent.gender_lean}, "
            f"missing_critical_fields:{current_intent.missing_critical_fields}, "
            f"confidence:{current_intent.confidence:.2f}"
            f"}}"
        )
    else:
        state_parts.append("intent=nessuno")

    state_parts.append(f"clarify_count={clarify_count}")
    state_parts.append(f"last_action={last_action or 'nessuno'}")

    if last_shown:
        shown_str = ", ".join(f"{p.product_id} {p.title[:20]}" for p in last_shown[:4])
        state_parts.append(f"last_shown=[{shown_str}]")
    else:
        state_parts.append("last_shown=nessuno")

    state_summary = ", ".join(state_parts)

    # Prepend recent conversation so the router can resolve references like
    # "il secondo" or "quello floreale" against the actual LLM replies shown.
    history_prefix = ""
    if topic_history and len(topic_history) > 1:
        prior = topic_history[:-1]  # exclude current message (already in MESSAGGIO)
        recent = prior[-6:]         # last 3 exchanges
        lines = []
        for msg in recent:
            role = "Cliente" if msg["role"] == "user" else "Lumé"
            lines.append(f"{role}: {msg['content'][:200]}")
        if lines:
            history_prefix = "CONVERSAZIONE RECENTE:\n" + "\n".join(lines) + "\n\n"

    user_content = f"{history_prefix}STATO: {state_summary}.\nMESSAGGIO: {message}"

    client = OpenAI(api_key=require_openai_key())
    messages: list[dict] = [{"role": "system", "content": _SYSTEM}]
    messages.extend(_FEW_SHOT)
    messages.append({"role": "user", "content": user_content})

    response = client.beta.chat.completions.parse(
        model=CONFIG.models.intent,
        messages=messages,
        response_format=RouterDecision,
        temperature=0,
    )
    result = response.choices[0].message.parsed
    if result is None:
        return RouterDecision(action="answer", reasoning="fallback")
    return result
