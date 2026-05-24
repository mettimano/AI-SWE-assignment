"""Post-hoc guard — validates every reply before it leaves the graph.

Hard violations trigger one regeneration attempt; if the retry also fails the
reply is replaced with a safe fallback template and needs_human is set True.

Hard violations (trigger regeneration):
  • hallucinated_product — a recommended product_id is not in the catalog
  • oos_leading        — first recommendation in answer mode is out-of-stock
  • markdown_detected  — reply_text contains **, ##, bullet dashes, or numbered lists
  • medical_claim      — reply_text contains prohibited medical/therapeutic language

Soft violations (logged in debug only, no regeneration):
  • excess_emoji       — more than 1 emoji (extra emojis stripped in-place)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from lume.catalog.models import NormalizedProduct
from lume.schemas import Reply

if TYPE_CHECKING:
    from lume.memory.profile import UserProfile


# ── Regex patterns ────────────────────────────────────────────────────────────

# Broad emoji match: covers emoticons, symbols, dingbats, and common decorative chars
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"   # misc symbols, emoticons, transport, etc.
    "\U00002702-\U000027B0"   # dingbats
    "\U0000FE00-\U0000FE0F"   # variation selectors (skin tones etc.)
    "\U00002500-\U00002BEF"   # box drawing, geometric shapes
    "✨🌸💋✅⭐❤🎁🎄🛒💄"
    "]",
    flags=re.UNICODE,
)

_MARKDOWN_RE = re.compile(
    r"\*\*[^*]+\*\*"        # **bold**
    r"|__[^_]+__"            # __bold__
    r"|\*[^*]+\*"            # *italic*
    r"|^#{1,6}\s"            # ## heading
    r"|^\s*-\s"              # - bullet
    r"|^\s*\d+\.\s",         # 1. numbered list
    re.MULTILINE,
)

_MEDICAL_RE = re.compile(
    r"\b(cura|guarisce|tratta(?:mento)?|terapeutic[oa]|clinicament[ei]|"
    r"medicinale|farmaco|diagnosi|prescriz|dermatolog|allergen|ipoallergen|"
    r"cures?|heals?|treats?|therapeutic|clinically[ -]proven)\b",
    re.IGNORECASE,
)


# ── Fallback templates ────────────────────────────────────────────────────────

_FALLBACK: dict[str, str] = {
    "answer": (
        "Ho trovato alcune opzioni che potrebbero interessarti — "
        "vuoi che ti dettagli una in particolare?"
    ),
    "clarify_question": "Per aiutarti al meglio ho bisogno di qualche informazione in più.",
    "clarify_probe": "Ecco qualche proposta con stili diversi — dimmi quale si avvicina di più.",
    "escalate": "Capisco, passo la tua richiesta a un nostro operatore.",
    "no_match": "Al momento non ho trovato prodotti che corrispondano esattamente — posso aiutarti con criteri diversi?",
    "specification": "Ecco i dettagli del prodotto che hai scelto. Come posso aiutarti ancora?",
}


# ── Individual checks ─────────────────────────────────────────────────────────

def _check_hallucinated_products(
    reply: Reply, catalog_ids: set[str]
) -> str | None:
    bad = [r.product_id for r in reply.recommendations if r.product_id not in catalog_ids]
    if bad:
        return f"hallucinated_product:{','.join(bad)}"
    return None


def _check_oos_leading(reply: Reply) -> str | None:
    if reply.mode == "answer" and reply.recommendations:
        if not reply.recommendations[0].available:
            return f"oos_leading:{reply.recommendations[0].product_id}"
    return None


def _check_markdown(reply: Reply) -> str | None:
    if _MARKDOWN_RE.search(reply.reply_text):
        return "markdown_detected"
    return None


def _check_medical_claims(reply: Reply) -> str | None:
    if _MEDICAL_RE.search(reply.reply_text):
        return "medical_claim"
    return None


def _count_emoji(text: str) -> int:
    return len(_EMOJI_RE.findall(text))


def _strip_extra_emoji(text: str) -> str:
    """Keep first emoji, remove subsequent ones."""
    found = False

    def _replacer(m: re.Match) -> str:
        nonlocal found
        if not found:
            found = True
            return m.group()
        return ""

    return _EMOJI_RE.sub(_replacer, text)


# ── Public API ────────────────────────────────────────────────────────────────

def check(
    reply: Reply,
    candidates: list[NormalizedProduct],
    catalog: list[NormalizedProduct],
    *,
    message: str = "",
    oos_fallback: list[NormalizedProduct] | None = None,
    user_profile: UserProfile | None = None,
) -> Reply:
    """Validate reply. Regenerate once on hard violations; fallback template if retry fails."""
    catalog_ids = {p.product_id for p in catalog}

    violations = _collect_violations(reply, catalog_ids)

    # ── Soft fix: excess emoji (strip in-place, no regeneration needed) ──
    emoji_count = _count_emoji(reply.reply_text)
    if emoji_count > 1:
        reply.reply_text = _strip_extra_emoji(reply.reply_text)
        reply.debug.setdefault("guard_warnings", []).append(
            f"excess_emoji:{emoji_count}"
        )

    if not violations:
        return reply

    # ── Hard violations → one regeneration attempt ──
    reply.debug.setdefault("guard_violations", []).extend(violations)

    retry = _regenerate(reply, candidates, oos_fallback or [], user_profile, message)
    if retry is not None:
        retry_violations = _collect_violations(retry, catalog_ids)
        if not retry_violations:
            retry.debug["guard_violations"] = violations
            retry.debug["guard_regenerated"] = True
            return retry
        # Retry also failed — record and fall through to template
        retry.debug.setdefault("guard_violations_retry", []).extend(retry_violations)

    # ── Fallback template ──
    reply.reply_text = _FALLBACK.get(reply.mode, _FALLBACK["answer"])
    reply.recommendations = [r for r in reply.recommendations if r.available][:4]
    reply.needs_human = True
    reply.debug["guard_fallback"] = True
    return reply


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_violations(reply: Reply, catalog_ids: set[str]) -> list[str]:
    checks = [
        _check_hallucinated_products(reply, catalog_ids),
        _check_oos_leading(reply),
        _check_markdown(reply),
        _check_medical_claims(reply),
    ]
    return [v for v in checks if v is not None]


def _regenerate(
    reply: Reply,
    candidates: list[NormalizedProduct],
    oos_fallback: list[NormalizedProduct],
    user_profile: UserProfile | None,
    message: str,
) -> Reply | None:
    """One regeneration attempt using the responder with slightly higher temperature."""
    try:
        from lume.agents.responder import generate_response  # noqa: PLC0415

        intent = reply.intent
        if intent is None:
            return None

        return generate_response(
            message=message,
            mode=reply.mode,
            intent=intent,
            candidates=candidates,
            oos_fallback=oos_fallback,
            user_profile=user_profile,
        )
    except Exception:  # noqa: BLE001
        return None
