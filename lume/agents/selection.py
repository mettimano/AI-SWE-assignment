"""Selection handler and preference learning agent.

When a user selects a product the graph calls:
  1. get_selected_product() — retrieves full product details from catalog
  2. infer_and_save_preferences() — LLM reads the conversation context and
     updates the user profile with learned preferences
     (gift context is weighted separately — not attributed to the user themselves)
"""

from __future__ import annotations

from openai import OpenAI
from pydantic import BaseModel, Field

from lume.agents.intent import Intent
from lume.catalog.models import NormalizedProduct
from lume.config import CONFIG, require_openai_key
from lume.memory.profile import KnownPreferences, UserProfile
from lume.memory.store import save


def get_selected_product(
    product_id: str,
    catalog: list[NormalizedProduct],
) -> NormalizedProduct | None:
    """Return the selected product from the catalog, or None if not found."""
    id_map = {p.product_id: p for p in catalog}
    return id_map.get(product_id)


# ── Preference learning agent ─────────────────────────────────────────────────

_PREFS_SYSTEM = """\
Sei un agente che apprende le preferenze di un cliente da una sessione di conversazione.
Analizza l'intent della conversazione e il prodotto selezionato dall'utente.
Estrai le preferenze PERSONALI dell'utente da salvare nel suo profilo.

REGOLE IMPORTANTI:
- Se il prodotto è un REGALO (gift_recipient != null), NON salvare come preferenze personali
  a meno che non ci siano segnali chiari che l'utente voglia qualcosa del genere anche per sé.
- Salva solo informazioni CERTE, non inferite.
- must_avoid è cumulativo — aggiungi solo nuovi elementi.
- brand_liked: aggiungi il brand del prodotto selezionato se non già presente.
- Se niche_preference = true nell'intent, aggiorna il profilo di conseguenza.
"""


class _LearnedPrefs(BaseModel):
    fragrance_families: list[str] = Field(default_factory=list)
    brands_liked: list[str] = Field(default_factory=list)
    must_avoid: list[str] = Field(default_factory=list)
    budget_max: float | None = None
    niche_lean: bool | None = None
    gender_lean: str | None = None
    is_gift_purchase: bool = False
    notes: str = ""


def infer_and_save_preferences(
    intent: Intent,
    selected_product: NormalizedProduct,
    user_profile: UserProfile,
) -> UserProfile:
    """Use LLM to infer preferences from the selection context and persist them."""
    if user_profile.user_id is None:
        return user_profile

    context = (
        f"Intent della sessione:\n{intent.model_dump_json(indent=2)}\n\n"
        f"Prodotto selezionato dall'utente:\n"
        f"  ID: {selected_product.product_id}\n"
        f"  Titolo: {selected_product.title}\n"
        f"  Prezzo: {selected_product.display_price}\n"
        f"  Collezioni: {', '.join(selected_product.raw_collections[:5])}\n"
        f"  Ingredienti: {selected_product.ingredienti or 'non disponibili'}\n"
    )

    client = OpenAI(api_key=require_openai_key())
    response = client.beta.chat.completions.parse(
        model=CONFIG.models.intent,
        messages=[
            {"role": "system", "content": _PREFS_SYSTEM},
            {"role": "user", "content": context},
        ],
        response_format=_LearnedPrefs,
        temperature=0,
    )
    learned = response.choices[0].message.parsed
    if learned is None:
        return user_profile

    # Don't update personal preferences for gift purchases
    if learned.is_gift_purchase:
        if learned.notes:
            user_profile.notes = (user_profile.notes + " " + learned.notes).strip()
        save(user_profile)
        return user_profile

    prefs: KnownPreferences = user_profile.known_preferences

    # Union lists
    def _union(existing: list, new: list) -> list:
        seen = set(existing)
        return existing + [x for x in new if x not in seen]

    if learned.fragrance_families:
        prefs.fragrance_families = _union(prefs.fragrance_families, learned.fragrance_families)
    if learned.brands_liked:
        prefs.brands_liked = _union(prefs.brands_liked, learned.brands_liked)
    if learned.must_avoid:
        prefs.must_avoid = _union(prefs.must_avoid, learned.must_avoid)
    if learned.budget_max is not None:
        prefs.budget_max = learned.budget_max
    if learned.niche_lean is not None:
        prefs.niche_lean = learned.niche_lean
    if learned.gender_lean:
        prefs.gender_lean = learned.gender_lean
    if learned.notes:
        user_profile.notes = (user_profile.notes + " " + learned.notes).strip()

    # Record the accepted product
    if selected_product.product_id not in user_profile.past_recommendations_accepted:
        user_profile.past_recommendations_accepted.append(selected_product.product_id)

    save(user_profile)
    return user_profile
