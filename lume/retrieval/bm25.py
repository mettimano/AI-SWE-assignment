"""BM25 index over normalized product search_text."""

from __future__ import annotations

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from lume.catalog.models import NormalizedProduct


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Index:
    def __init__(self, products: list[NormalizedProduct]) -> None:
        self._products = products
        self._id_to_idx = {p.product_id: i for i, p in enumerate(products)}
        corpus = [_tokenize(p.search_text) for p in products]
        self._bm25 = BM25Okapi(corpus)

    def query(self, text: str, top_k: int = 30) -> list[tuple[str, float]]:
        """Return [(product_id, score), ...] sorted descending."""
        tokens = _tokenize(text)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self._products[i].product_id, float(s)) for i, s in ranked[:top_k]]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with open(path, "rb") as fh:
            return pickle.load(fh)
