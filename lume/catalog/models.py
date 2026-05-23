"""Pydantic models for the product catalog (raw JSON → normalized)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PriceBand = Literal["entry", "mid", "premium", "luxury"]


class Variant(BaseModel):
    model_config = ConfigDict(extra="ignore")

    variant_id: str
    title: str
    price_eur: float
    available: bool


class CustomField(BaseModel):
    model_config = ConfigDict(extra="ignore")

    key: str
    value: str


class ProductCategory(BaseModel):
    model_config = ConfigDict(extra="ignore")

    l1: str | None = None
    l2: str | None = None
    l3: str | None = None
    l4: str | None = None
    fullPath: str | None = None


class Product(BaseModel):
    """Raw product as stored in catalog.json. Extra fields are silently ignored."""

    model_config = ConfigDict(extra="ignore")

    product_id: str
    title: str
    description: str = ""
    collections: list[str] = Field(default_factory=list)
    min_price_eur: float = 0.0
    max_price_eur: float = 0.0
    available: bool = False
    variants: list[Variant] = Field(default_factory=list)
    custom_fields: list[CustomField] = Field(default_factory=list)
    productCategory: ProductCategory | None = None


class NormalizedProduct(BaseModel):
    """Product after HTML stripping and structural derivation.

    Only 5 hard-constraint fields are stored as Chroma metadata for WHERE filtering
    (available, min/max price, is_tester, is_niche). Everything else — brand, fragrance
    family, gender, occasion — is handled by BM25 + embeddings + LLM rerank at query time.
    """

    model_config = ConfigDict(extra="ignore")

    # Core identity
    product_id: str
    title: str
    cleaned_description: str
    raw_collections: list[str]

    # Hard-constraint metadata (5 Chroma WHERE-filter fields)
    effective_available: bool   # product.available AND any(v.available)
    is_tester: bool             # title starts "T." OR "tester" in collections
    is_niche: bool              # "nicchia" in any collection slug

    # Pricing
    min_price_eur: float
    max_price_eur: float
    price_band: PriceBand

    # Retrieval document: fed to both BM25 and the vector index
    search_text: str

    # Variant list for display and price filtering
    variants: list[Variant] = Field(default_factory=list)

    # From custom_fields — injected into responder context
    tipologia_prodotto: str | None = None
    ingredienti: str | None = None
    category_path: str | None = None

    @property
    def display_price(self) -> str:
        if self.min_price_eur == self.max_price_eur:
            return f"€{self.min_price_eur:.2f}"
        return f"€{self.min_price_eur:.2f}–€{self.max_price_eur:.2f}"

    @property
    def available_variants(self) -> list[Variant]:
        return [v for v in self.variants if v.available]

    @property
    def cheapest_available_price(self) -> float | None:
        avail = self.available_variants
        return min(v.price_eur for v in avail) if avail else None
