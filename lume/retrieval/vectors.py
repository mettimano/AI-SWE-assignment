"""Chroma vector store: embed search_text with text-embedding-3-small, persist locally."""

from __future__ import annotations

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from lume.catalog.models import NormalizedProduct
from lume.config import CONFIG, CHROMA_DIR, require_openai_key

COLLECTION_NAME = "lume_products"


def _embedding_fn() -> OpenAIEmbeddingFunction:
    return OpenAIEmbeddingFunction(
        api_key=require_openai_key(),
        model_name=CONFIG.models.embed,
    )


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def build_index(products: list[NormalizedProduct], *, reset: bool = False) -> None:
    """Embed all products and persist to Chroma. Idempotent unless reset=True."""
    client = get_client()
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_fn(),
        metadata={"hnsw:space": "cosine"},
    )

    existing = set(collection.get(include=[])["ids"])
    new_products = [p for p in products if p.product_id not in existing]
    if not new_products:
        return

    # Chroma batch limit is 5461 — not an issue at 300 products
    collection.add(
        ids=[p.product_id for p in new_products],
        documents=[p.search_text for p in new_products],
        metadatas=[
            {
                "available": p.effective_available,
                "min_price_eur": p.min_price_eur,
                "max_price_eur": p.max_price_eur,
                "is_tester": p.is_tester,
                "is_niche": p.is_niche,
            }
            for p in new_products
        ],
    )


def query_vector(
    text: str,
    top_k: int = 30,
    *,
    where: dict | None = None,
) -> list[tuple[str, float]]:
    """Return [(product_id, distance), ...] sorted ascending (lower = closer)."""
    client = get_client()
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_fn(),
    )
    kwargs: dict = dict(query_texts=[text], n_results=min(top_k, collection.count()))
    if where:
        kwargs["where"] = where
    results = collection.query(**kwargs)
    ids = results["ids"][0]
    distances = results["distances"][0]
    return list(zip(ids, distances))
