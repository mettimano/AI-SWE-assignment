"""Public output schema returned by the graph for every turn."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from lume.agents.clarify import ClarifyQuestion, ProbeProduct
from lume.agents.intent import Intent


class Recommendation(BaseModel):
    product_id: str
    title: str
    price_display: str
    available: bool
    why: str  # short, brand-voice compliant reason


class Reply(BaseModel):
    mode: Literal["answer", "clarify_question", "clarify_probe", "escalate", "no_match", "specification"]
    reply_text: str                                    # WhatsApp-style plain text
    recommendations: list[Recommendation] = Field(default_factory=list)   # mode=answer
    questions: list[ClarifyQuestion] = Field(default_factory=list)        # mode=clarify_question
    probes: list[ProbeProduct] = Field(default_factory=list)              # mode=clarify_probe
    intent: Intent | None = None
    needs_human: bool = False
    user_id: str | None = None
    debug: dict[str, Any] = Field(default_factory=dict)
