"""UserProfile schema and prompt-redaction helper."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class KnownPreferences(BaseModel):
    fragrance_families: list[str] = Field(default_factory=list)
    brands_liked: list[str] = Field(default_factory=list)
    brands_disliked: list[str] = Field(default_factory=list)
    budget_max: float | None = None
    niche_lean: bool = False
    gender_lean: str | None = None  # "uomo" | "donna" | "unisex"
    must_avoid: list[str] = Field(default_factory=list)  # ingredients, notes, brands


class UserProfile(BaseModel):
    user_id: str
    language: str | None = None  # "it" | "en"
    known_preferences: KnownPreferences = Field(default_factory=KnownPreferences)
    past_purchases: list[str] = Field(default_factory=list)          # product_ids
    past_recommendations_accepted: list[str] = Field(default_factory=list)
    past_recommendations_rejected: list[str] = Field(default_factory=list)
    notes: str = ""
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def redact_for_prompt(profile: UserProfile) -> str:
    """Compact prose summary of a user profile, safe to inject into a system prompt.

    Covers only what is known — omits empty fields so the LLM isn't confused by
    'no preferences' entries.
    """
    lines: list[str] = []
    prefs = profile.known_preferences

    if prefs.fragrance_families:
        lines.append(f"Famiglie olfattive preferite: {', '.join(prefs.fragrance_families)}.")
    if prefs.brands_liked:
        lines.append(f"Brand apprezzati: {', '.join(prefs.brands_liked)}.")
    if prefs.brands_disliked:
        lines.append(f"Brand da evitare: {', '.join(prefs.brands_disliked)}.")
    if prefs.must_avoid:
        lines.append(f"Note/ingredienti da evitare: {', '.join(prefs.must_avoid)}.")
    if prefs.budget_max is not None:
        lines.append(f"Budget massimo indicativo: €{prefs.budget_max:.0f}.")
    if prefs.niche_lean:
        lines.append("Preferisce prodotti di nicchia.")
    if prefs.gender_lean:
        lines.append(f"Orientamento di genere: {prefs.gender_lean}.")
    if profile.past_purchases:
        count = len(profile.past_purchases)
        lines.append(f"Ha già acquistato {count} prodott{'o' if count == 1 else 'i'} in passato.")
    if profile.notes:
        lines.append(f"Note aggiuntive: {profile.notes}")

    return " ".join(lines) if lines else ""
