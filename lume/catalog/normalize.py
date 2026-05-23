"""Deterministic normalization: Product → NormalizedProduct.

No LLM calls. No external lookups. Only structural derivations from raw fields.
"""

from __future__ import annotations

from .loader import strip_html
from .models import NormalizedProduct, PriceBand, Product


def _price_band(max_price: float) -> PriceBand:
    if max_price < 40:
        return "entry"
    if max_price < 100:
        return "mid"
    if max_price < 200:
        return "premium"
    return "luxury"


def normalize(product: Product) -> NormalizedProduct:
    cleaned_desc = strip_html(product.description)
    colls = product.collections
    cf = {f.key: f.value for f in product.custom_fields}
    tipologia = cf.get("tipologia_prodotto")
    ingredienti = cf.get("ingredienti")

    effective_available = product.available and any(v.available for v in product.variants)
    is_tester = "tester" in colls or product.title.lstrip().startswith("T.")
    is_niche = any("nicchia" in s for s in colls)

    search_text = " ".join(filter(None, [
        product.title,
        cleaned_desc,
        " ".join(colls),
        tipologia,
        ingredienti,
    ]))

    return NormalizedProduct(
        product_id=product.product_id,
        title=product.title,
        cleaned_description=cleaned_desc,
        raw_collections=colls,
        effective_available=effective_available,
        is_tester=is_tester,
        is_niche=is_niche,
        min_price_eur=product.min_price_eur,
        max_price_eur=product.max_price_eur,
        price_band=_price_band(product.max_price_eur),
        search_text=search_text,
        variants=product.variants,
        tipologia_prodotto=tipologia,
        ingredienti=ingredienti,
        category_path=product.productCategory.fullPath if product.productCategory else None,
    )


def normalize_all(products: list[Product]) -> list[NormalizedProduct]:
    return [normalize(p) for p in products]
