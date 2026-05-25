"""LangGraph orchestrator — wires all agents with hard state-machine constraints.

Turn flow:
  START → load_profile → router_node → [graph enforces constraints] → sub-agent → guard → END

Hard constraints enforced by the graph (override router if violated):
  • extract_intent cannot follow extract_intent (last_action guard)
  • clarify_question blocked when clarify_count >= 2 → forced to answer pipeline
  • refine_intent requires current_intent to exist
  • selection requires last_shown_products to be non-empty
  • new_topic resets clarify_count and clears current_intent before calling extract_intent
  • save_preferences runs automatically after selection (not router-callable)
  • guard runs automatically on every output (not router-callable)

State that persists across turns (managed by caller / REPL):
  current_intent, last_shown_product_ids, last_shown_products, topic_history,
  clarify_count, last_action, last_shown_mode
"""

from __future__ import annotations

import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from lume.agents.clarify import build_clarify_payload
from lume.agents.constraints import apply as apply_constraints
from lume.agents.constraints import soft_instructions
from lume.agents.intent import Intent, extract_intent, refine_intent
from lume.agents.router import RouterDecision, route
from lume.agents.selection import get_selected_product, infer_and_save_preferences
from lume.catalog.loader import load_products
from lume.catalog.models import NormalizedProduct
from lume.catalog.normalize import normalize_all
from lume.config import CACHE_DIR, CATALOG_PATH
from lume.memory.profile import UserProfile
from lume.memory.store import load_or_init
from lume.retrieval.bm25 import BM25Index
from lume.retrieval.hybrid import hybrid_search
from lume.retrieval.rerank import rerank as llm_rerank
from lume.retrieval.vectors import build_index, query_vector
from lume.schemas import Reply

# ── Module-level singletons ───────────────────────────────────────────────────

_catalog: list[NormalizedProduct] | None = None
_bm25: BM25Index | None = None
_BM25_CACHE = CACHE_DIR / "bm25_index.pkl"


def _ensure_resources() -> tuple[list[NormalizedProduct], BM25Index]:
    global _catalog, _bm25
    if _catalog is None:
        raw = load_products(CATALOG_PATH)
        _catalog = normalize_all(raw)
        build_index(_catalog)  # idempotent
    if _bm25 is None:
        if _BM25_CACHE.exists():
            _bm25 = BM25Index.load(_BM25_CACHE)
        else:
            _bm25 = BM25Index(_catalog)
            _bm25.save(_BM25_CACHE)
    return _catalog, _bm25


# ── State ─────────────────────────────────────────────────────────────────────

class ConversationState(TypedDict, total=False):
    # ── Caller provides these on every invoke ──
    user_id: str | None
    message: str

    # ── Cross-turn state (caller passes back from previous turn's output) ──
    current_intent: Intent | None
    last_shown_product_ids: list[str]
    last_shown_products: list[NormalizedProduct]
    topic_history: list[dict]  # full convo since last new_topic: [{"role": "user"|"assistant", "content": str}]
    clarify_count: int         # 0-2; resets on new_topic
    last_action: str | None    # last action taken (prevents consecutive extract_intent)

    # ── Set during graph run ──
    user_profile: UserProfile | None
    router_decision: RouterDecision | None
    effective_action: str      # after constraint enforcement
    candidates: list[NormalizedProduct]
    oos_fallback: list[NormalizedProduct]
    selected_product: NormalizedProduct | None  # set by node_selection, read by node_save_preferences
    last_shown_mode: str | None  # "answer" | "clarify_probe" | "clarify_question" — drives post-selection routing
    retrieval_query: str | None  # cached LLM-generated query; set by node_retrieve, consumed by node_rerank
    reply: Reply | None
    _t_start: float


# ── Helper: constraint enforcement ───────────────────────────────────────────

def _enforce_constraints(state: ConversationState, decision: RouterDecision) -> str:
    """Apply hard constraints to the router's decision. Returns effective action."""
    action = decision.action
    clarify_count = state.get("clarify_count", 0)
    last_action = state.get("last_action")
    current_intent = state.get("current_intent")
    last_shown = state.get("last_shown_products") or []

    # new_topic always allowed — it resets everything
    if action == "new_topic":
        return "new_topic"

    # escalate always allowed
    if action == "escalate":
        return "escalate"

    # extract_intent: blocked if just ran (can't call twice in a row)
    if action == "extract_intent" and last_action == "extract_intent":
        if clarify_count <2:
            return "clarify_question"
        else:
            return "answer"  # fall through to answer with whatever we have

    # refine_intent: requires existing intent
    if action == "refine_intent" and current_intent is None:
        return "extract_intent"  # no intent yet → extract fresh

    # clarify_question: blocked after 2 rounds
    if action == "clarify_question" and clarify_count >= 2:
        return "answer"  # force answer with available info

    # selection: requires last_shown products
    if action == "selection" and not last_shown:
        return "answer"  # nothing was shown → can't select

    return action


# ── Nodes ─────────────────────────────────────────────────────────────────────

def node_load_profile(state: ConversationState) -> dict:
    user_id = state.get("user_id")
    profile = load_or_init(user_id) if user_id else None
    # Append current user message to conversation history (done once per turn here,
    # before any agent node runs). node_new_topic overrides this with a full reset.
    existing = state.get("topic_history") or []
    topic_history = existing + [{"role": "user", "content": state["message"]}]
    return {"user_profile": profile, "_t_start": time.perf_counter(), "topic_history": topic_history}


def node_router(state: ConversationState) -> dict:
    """Call the router LLM, then enforce hard constraints."""
    topic_history = state.get("topic_history") or []
    # Pass history without the current message (it's in the MESSAGGIO field)
    history_ctx = topic_history[:-1] if topic_history else []
    decision = route(
        message=state["message"],
        current_intent=state.get("current_intent"),
        last_shown=state.get("last_shown_products") or [],
        clarify_count=state.get("clarify_count", 0),
        last_action=state.get("last_action"),
        topic_history=history_ctx,
    )
    effective = _enforce_constraints(state, decision)
    return {"router_decision": decision, "effective_action": effective}


def node_new_topic(state: ConversationState) -> dict:
    """Reset all intent state, then extract a fresh intent."""
    profile = state.get("user_profile")
    # New topic: no history context for intent extraction (clean slate)
    intent = extract_intent(state["message"], profile, topic_history=None)
    intent = _apply_profile_defaults(intent, profile)
    return {
        "current_intent": intent,
        "last_shown_product_ids": [],
        "last_shown_products": [],
        # Reset conversation history; node_load_profile already appended the user
        # message, so we reset to just that one entry.
        "topic_history": [{"role": "user", "content": state["message"]}],
        "last_shown_mode": None,   # clear stale probe/answer mode from previous topic
        "clarify_count": 0,
        "last_action": "extract_intent",
        "retrieval_query": None,
    }


def _apply_profile_defaults(intent: Intent, profile: UserProfile | None) -> Intent:
    """Fill missing critical intent fields from the user's stored profile.

    A returning user shouldn't have to repeat their known budget or gender every
    message. The intent LLM is instructed to use profile defaults, but we apply
    them deterministically here as a fallback to keep routing reliable.
    """
    if profile is None:
        return intent
    prefs = profile.known_preferences
    updates: dict = {}
    missing = list(intent.missing_critical_fields)

    if intent.budget_max is None and prefs.budget_max is not None:
        updates["budget_max"] = prefs.budget_max
        missing = [f for f in missing if f != "budget_max"]

    if intent.gender_lean is None and prefs.gender_lean:
        updates["gender_lean"] = prefs.gender_lean
        missing = [f for f in missing if f != "gender_lean"]

    if updates:
        updates["missing_critical_fields"] = missing
        return intent.model_copy(update=updates)
    return intent


def node_extract_intent(state: ConversationState) -> dict:
    profile = state.get("user_profile")
    topic_history = state.get("topic_history") or []
    # Pass history without the current message (already the LLM user turn)
    history_ctx = topic_history[:-1] if topic_history else []
    intent = extract_intent(state["message"], profile, topic_history=history_ctx)
    intent = _apply_profile_defaults(intent, profile)
    return {"current_intent": intent, "last_action": "extract_intent"}


def node_refine_intent(state: ConversationState) -> dict:
    profile = state.get("user_profile")
    topic_history = state.get("topic_history") or []
    history_ctx = topic_history[:-1] if topic_history else []
    intent = refine_intent(
        state["message"],
        state["current_intent"],
        state.get("last_shown_products") or [],
        profile,
        topic_history=history_ctx,
    )
    return {"current_intent": intent, "last_action": "refine_intent"}


def node_route_after_intent(state: ConversationState) -> dict:
    """After extract/refine intent: re-run constraint check for clarify vs answer."""
    # This is a pass-through node; routing is driven by conditional edges
    return {}


def node_clarify(state: ConversationState) -> dict:
    from lume.agents.responder import generate_response  # noqa: PLC0415

    intent: Intent = state["current_intent"]
    candidates = state.get("candidates") or []
    topic_history = state.get("topic_history") or []
    payload = build_clarify_payload(intent, candidates, topic_history=topic_history)
    t_start = state.get("_t_start", time.perf_counter())

    reply = generate_response(
        message=state["message"],
        mode=payload.mode,
        intent=intent,
        candidates=[],
        oos_fallback=[],
        user_profile=state.get("user_profile"),
        clarify_payload=payload,
        topic_history=topic_history,
    )
    reply.questions = payload.questions
    reply.probes = payload.probes
    reply.debug["latency_ms"] = int((time.perf_counter() - t_start) * 1000)
    reply.user_id = state.get("user_id")

    # Resolve probe NormalizedProducts so the router can match selection references
    probe_ids = [p.product_id for p in payload.probes]
    id_to_candidate = {p.product_id: p for p in candidates}
    probe_products = [id_to_candidate[pid] for pid in probe_ids if pid in id_to_candidate]

    updated_history = topic_history + [{"role": "assistant", "content": reply.reply_text}]

    return {
        "reply": reply,
        "last_action": reply.mode,         # "clarify_question" or "clarify_probe"
        "last_shown_mode": reply.mode,     # drives post-selection routing
        "last_shown_products": probe_products,
        "last_shown_product_ids": probe_ids,
        "clarify_count": state.get("clarify_count", 0) + 1,
        "topic_history": updated_history,
    }


def node_retrieve(state: ConversationState) -> dict:
    from lume.retrieval.query import generate_retrieval_query  # noqa: PLC0415

    intent: Intent = state["current_intent"]
    catalog, bm25 = _ensure_resources()

    topic_history = state.get("topic_history") or []
    # After probe selection, the retrieval query was built from the selected
    # product's qualities in node_selection — use it directly instead of asking
    # the LLM to interpret the ambiguous selection text ("il terzo", etc.).
    is_post_probe = (
        state.get("last_action") == "selection"
        and state.get("last_shown_mode") == "clarify_probe"
    )
    if is_post_probe and state.get("retrieval_query"):
        query = state["retrieval_query"]
    else:
        query = generate_retrieval_query(topic_history, intent) or state["message"]
    id_map = {p.product_id: p for p in catalog}

    # When the user explicitly asked for testers, restrict both retrieval paths so
    # only is_tester=True products surface before reranking.
    tester_only = getattr(intent, "tester_requested", False)
    bm25_results = bm25.query(query, top_k=30)
    if tester_only:
        bm25_results = [(pid, s) for pid, s in bm25_results if id_map.get(pid) and id_map[pid].is_tester]

    vector_where = {"is_tester": True} if tester_only else None
    vector_results = query_vector(query, top_k=30, where=vector_where)

    fused_ids = hybrid_search(
        bm25_results,
        vector_results,
        bm25_weight=intent.bm25_weight,
        vector_weight=intent.vector_weight,
        top_n=20,
    )
    candidates = [id_map[pid] for pid in fused_ids if pid in id_map]
    return {"candidates": candidates, "retrieval_query": query}


def node_constrain(state: ConversationState) -> dict:
    primary, oos = apply_constraints(state.get("candidates", []), state["current_intent"])
    return {"candidates": primary, "oos_fallback": oos}


def node_rerank(state: ConversationState) -> dict:
    from lume.memory.profile import redact_for_prompt  # noqa: PLC0415

    intent: Intent = state["current_intent"]
    profile = state.get("user_profile")
    query = state.get("retrieval_query") or state["message"]
    reranked = llm_rerank(
        query,
        state.get("candidates", []),
        soft_instructions=soft_instructions(intent),
        user_profile_summary=redact_for_prompt(profile) if profile else "",
        top_k=4,
    )
    return {"candidates": reranked}


def node_answer(state: ConversationState) -> dict:
    from lume.agents.responder import generate_response  # noqa: PLC0415

    intent: Intent = state["current_intent"]
    candidates: list[NormalizedProduct] = state.get("candidates", [])
    mode = "answer" if candidates else "no_match"
    t_start = state.get("_t_start", time.perf_counter())
    topic_history = state.get("topic_history") or []

    # After probe selection the user's message is a selection reference ("il terzo")
    # which confuses the responder into giving details about one product. Build an
    # explicit "find similar" message from the selected product so the responder
    # presents 2-4 alternatives instead.
    is_post_probe = (
        state.get("last_action") == "selection"
        and state.get("last_shown_mode") == "clarify_probe"
    )
    if is_post_probe:
        selected: NormalizedProduct | None = state.get("selected_product")
        if selected:
            responder_message = (
                f"Trovami prodotti simili a {selected.title}, "
                "stesse caratteristiche e stile."
            )
        else:
            responder_message = state.get("retrieval_query") or state["message"]
    else:
        responder_message = state["message"]

    reply = generate_response(
        message=responder_message,
        mode=mode,
        intent=intent,
        candidates=candidates,
        oos_fallback=state.get("oos_fallback", []),
        user_profile=state.get("user_profile"),
        topic_history=topic_history,
    )
    reply.debug["latency_ms"] = int((time.perf_counter() - t_start) * 1000)
    reply.user_id = state.get("user_id")

    shown_ids = [r.product_id for r in reply.recommendations]
    catalog, _ = _ensure_resources()
    id_map = {p.product_id: p for p in catalog}
    shown_products = [id_map[pid] for pid in shown_ids if pid in id_map]

    updated_history = topic_history + [{"role": "assistant", "content": reply.reply_text}]

    return {
        "reply": reply,
        "last_shown_product_ids": shown_ids,
        "last_shown_products": shown_products,
        "last_shown_mode": "answer",
        "last_action": "answer",
        "topic_history": updated_history,
    }


def node_selection(state: ConversationState) -> dict:
    """Handle product selection.

    Probe selection (last_shown_mode == "clarify_probe"):
      Refine intent using the selected product as a style anchor, then let
      route_after_save_preferences trigger a fresh retrieval pass → answer.
      No intermediate specification reply is generated.

    Regular answer selection:
      Show product details in specification mode and stop.
    """
    from lume.agents.responder import generate_response  # noqa: PLC0415

    decision: RouterDecision = state["router_decision"]
    catalog, _ = _ensure_resources()
    product = get_selected_product(decision.selected_product_id or "", catalog)
    intent: Intent = state["current_intent"]
    t_start = state.get("_t_start", time.perf_counter())

    topic_history = state.get("topic_history") or []
    result: dict = {"selected_product": product, "last_action": "selection"}

    if state.get("last_shown_mode") == "clarify_probe" and not product:
        # Probe selected but product not found in catalog — fall back to no_match
        result["reply"] = generate_response(
            message=state["message"],
            mode="no_match",
            intent=intent,
            candidates=[],
            oos_fallback=[],
            user_profile=state.get("user_profile"),
            topic_history=topic_history,
        )
        result["reply"].debug["latency_ms"] = int((time.perf_counter() - t_start) * 1000)
        result["reply"].user_id = state.get("user_id")
        result["topic_history"] = topic_history + [{"role": "assistant", "content": result["reply"].reply_text}]
        return result

    if state.get("last_shown_mode") == "clarify_probe" and product:
        # Use selected probe as a style anchor: refine intent, then re-run retrieval.
        # The full conversation history gives refine_intent rich context about which
        # probe product was shown and what the user chose.
        profile = state.get("user_profile")
        synthetic_msg = (
            f"Ho scelto '{product.title}'. "
            "Trovami altri prodotti con caratteristiche simili."
        )
        updated_intent = refine_intent(
            synthetic_msg,
            intent,
            last_shown=state.get("last_shown_products") or [],
            user_profile=profile,
            topic_history=topic_history,
        )
        # Ambiguity resolved — set clarify_count to max so route_after_constrain
        # never triggers another probe regardless of the updated intent's confidence.
        result["current_intent"] = updated_intent
        result["clarify_count"] = 2
        # Pre-build the retrieval query from the selected product's qualities so
        # node_retrieve doesn't misinterpret the user's selection text ("il terzo").
        result["retrieval_query"] = product.search_text
        # No reply here — node_answer generates the response after re-retrieval
    else:
        candidates = [product] if product else []
        reply = generate_response(
            message=state["message"],
            mode="specification",
            intent=intent,
            candidates=candidates,
            oos_fallback=[],
            user_profile=state.get("user_profile"),
            topic_history=topic_history,
        )
        reply.debug["latency_ms"] = int((time.perf_counter() - t_start) * 1000)
        reply.user_id = state.get("user_id")
        result["reply"] = reply
        result["topic_history"] = topic_history + [{"role": "assistant", "content": reply.reply_text}]

    return result


def node_save_preferences(state: ConversationState) -> dict:
    """Persist learned preferences after a selection (runs automatically, not router-callable)."""
    profile = state.get("user_profile")
    product: NormalizedProduct | None = state.get("selected_product")
    intent = state.get("current_intent")
    if profile and product and intent and profile.user_id:
        infer_and_save_preferences(intent, product, profile)
    return {}


def node_escalate(state: ConversationState) -> dict:
    from lume.agents.responder import generate_response  # noqa: PLC0415

    intent: Intent = state.get("current_intent") or Intent()
    t_start = state.get("_t_start", time.perf_counter())
    topic_history = state.get("topic_history") or []
    reply = generate_response(
        message=state["message"],
        mode="escalate",
        intent=intent,
        candidates=[],
        oos_fallback=[],
        user_profile=state.get("user_profile"),
        topic_history=topic_history,
    )
    reply.needs_human = True
    reply.debug["latency_ms"] = int((time.perf_counter() - t_start) * 1000)
    reply.user_id = state.get("user_id")
    updated_history = topic_history + [{"role": "assistant", "content": reply.reply_text}]
    return {"reply": reply, "last_action": "escalate", "topic_history": updated_history}


def node_guard(state: ConversationState) -> dict:
    from lume.agents.guard import check  # noqa: PLC0415

    reply = state.get("reply")
    if reply is None:
        return {}
    catalog, _ = _ensure_resources()
    checked = check(
        reply,
        state.get("candidates", []),
        catalog,
        message=state.get("message", ""),
        oos_fallback=state.get("oos_fallback", []),
        user_profile=state.get("user_profile"),
    )
    return {"reply": checked}


# ── Routing functions ─────────────────────────────────────────────────────────

def route_from_router(state: ConversationState) -> str:
    return state.get("effective_action", "answer")


def route_after_intent(state: ConversationState) -> str:
    """After extract/refine: escalate, question-mode clarify, or retrieve.

    Probe-mode clarify (soft-ambiguity) is deferred to route_after_constrain
    so that candidates are available when build_clarify_payload selects probes.
    """
    intent: Intent = state.get("current_intent") or Intent()
    clarify_count = state.get("clarify_count", 0)

    if intent.escalate:
        return "escalate"
    # Only short-circuit to clarify when a hard-filter field is absent (question mode).
    # Probe mode needs candidates → decided after constrain.
    critical_missing = bool(set(intent.missing_critical_fields) & {"budget_max", "categories"})
    # Only auto-clarify on the first pass. If the user already received a question and
    # still hasn't provided the info, we proceed to retrieval with what we have rather
    # than asking the same question again.
    if critical_missing and clarify_count == 0:
        return "clarify"
    return "retrieve"


def route_after_save_preferences(state: ConversationState) -> str:
    """After saving preferences: if selection was from a probe, re-run retrieval.
    Otherwise the reply (specification) is already set — go straight to guard."""
    return "retrieve" if state.get("last_shown_mode") == "clarify_probe" else "guard"


def route_after_constrain(state: ConversationState) -> str:
    """After constrain: probe-mode clarify if soft-ambiguous, else rerank or no_match."""
    from lume.config import CONFIG  # noqa: PLC0415

    intent: Intent = state.get("current_intent") or Intent()
    clarify_count = state.get("clarify_count", 0)

    # Probe mode: candidates available now, soft axes still open
    critical_missing = bool(set(intent.missing_critical_fields) & {"budget_max", "categories"})
    if (
        not critical_missing
        and intent.confidence < CONFIG.clarify.soft_ambiguity_threshold
        and bool(intent.categories)
        and clarify_count < 2
    ):
        return "clarify"

    if not state.get("candidates"):
        return "answer"  # triggers no_match branch inside node_answer
    return "rerank"


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(ConversationState)

    # Nodes
    g.add_node("load_profile", node_load_profile)
    g.add_node("router", node_router)
    g.add_node("new_topic", node_new_topic)
    g.add_node("extract_intent", node_extract_intent)
    g.add_node("refine_intent", node_refine_intent)
    g.add_node("route_after_intent", node_route_after_intent)
    g.add_node("clarify", node_clarify)
    g.add_node("retrieve", node_retrieve)
    g.add_node("constrain", node_constrain)
    g.add_node("rerank", node_rerank)
    g.add_node("answer", node_answer)
    g.add_node("selection", node_selection)
    g.add_node("save_preferences", node_save_preferences)
    g.add_node("escalate", node_escalate)
    g.add_node("guard", node_guard)

    # Entry
    g.add_edge(START, "load_profile")
    g.add_edge("load_profile", "router")

    # Router → dispatch
    g.add_conditional_edges(
        "router",
        route_from_router,
        {
            "extract_intent": "extract_intent",
            "refine_intent": "refine_intent",
            "clarify_question": "clarify",
            "answer": "retrieve",
            "selection": "selection",
            "new_topic": "new_topic",
            "escalate": "escalate",
        },
    )

    # new_topic resets + extracts → then check if we need to clarify or retrieve
    g.add_edge("new_topic", "route_after_intent")

    # extract/refine → check if clarify needed
    g.add_edge("extract_intent", "route_after_intent")
    g.add_edge("refine_intent", "route_after_intent")
    g.add_conditional_edges(
        "route_after_intent",
        route_after_intent,
        {"escalate": "escalate", "clarify": "clarify", "retrieve": "retrieve"},
    )

    # Retrieval pipeline
    g.add_edge("retrieve", "constrain")
    g.add_conditional_edges(
        "constrain",
        route_after_constrain,
        {"clarify": "clarify", "rerank": "rerank", "answer": "answer"},
    )
    g.add_edge("rerank", "answer")

    # Selection → save_preferences → (probe? re-retrieve : guard)
    g.add_edge("selection", "save_preferences")
    g.add_conditional_edges(
        "save_preferences",
        route_after_save_preferences,
        {"retrieve": "retrieve", "guard": "guard"},
    )

    # All other terminal nodes → guard → END
    g.add_edge("answer", "guard")
    g.add_edge("clarify", "guard")
    g.add_edge("escalate", "guard")
    g.add_edge("guard", END)

    return g.compile()


# ── Singleton + public run_turn ───────────────────────────────────────────────

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_turn(
    message: str,
    *,
    user_id: str | None = None,
    current_intent: Intent | None = None,
    last_shown_product_ids: list[str] | None = None,
    last_shown_products: list[NormalizedProduct] | None = None,
    topic_history: list[dict] | None = None,
    clarify_count: int = 0,
    last_action: str | None = None,
    last_shown_mode: str | None = None,
) -> dict[str, Any]:
    """Run one conversation turn. Returns full state dict for caller to persist.

    The caller (CLI / REPL) is responsible for passing back the cross-turn state
    fields from the previous turn's output.
    """
    state: ConversationState = {
        "user_id": user_id,
        "message": message,
        "current_intent": current_intent,
        "last_shown_product_ids": last_shown_product_ids or [],
        "last_shown_products": last_shown_products or [],
        "topic_history": topic_history or [],
        "clarify_count": clarify_count,
        "last_action": last_action,
        "last_shown_mode": last_shown_mode,
    }
    return get_graph().invoke(state)
