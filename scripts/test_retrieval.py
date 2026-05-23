"""Smoke test: ingest → BM25 + Chroma build → hybrid search → rerank.

Usage:
    uv run python scripts/test_retrieval.py
    uv run python scripts/test_retrieval.py --query "profumo floreale donna budget 80"
    uv run python scripts/test_retrieval.py --reset   # wipe and rebuild Chroma
"""

from __future__ import annotations

import argparse
import time

from rich.console import Console
from rich.table import Table

from lume.catalog.loader import load_products
from lume.catalog.normalize import normalize_all
from lume.config import CACHE_DIR, CATALOG_PATH
from lume.retrieval.bm25 import BM25Index
from lume.retrieval.hybrid import hybrid_search
from lume.retrieval.rerank import rerank
from lume.retrieval.vectors import build_index, query_vector

console = Console()
BM25_CACHE = CACHE_DIR / "bm25_index.pkl"


def build_or_load_bm25(products, *, reset: bool = False) -> BM25Index:
    if not reset and BM25_CACHE.exists():
        console.print("[dim]Loading BM25 index from cache...[/dim]")
        return BM25Index.load(BM25_CACHE)
    console.print("[yellow]Building BM25 index...[/yellow]")
    idx = BM25Index(products)
    idx.save(BM25_CACHE)
    console.print(f"[green]BM25 index saved → {BM25_CACHE}[/green]")
    return idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="profumo floreale donna regalo 80 euro")
    parser.add_argument("--reset", action="store_true", help="Wipe and rebuild Chroma collection")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-rerank", action="store_true", help="Skip LLM rerank step")
    args = parser.parse_args()

    # ── 1. Load + normalize ──────────────────────────────────────────────────
    console.rule("[bold]1. Catalog load + normalize")
    t0 = time.perf_counter()
    products_raw = load_products(CATALOG_PATH)
    products = normalize_all(products_raw)
    console.print(f"  Loaded {len(products)} products in {time.perf_counter()-t0:.2f}s")

    id_to_product = {p.product_id: p for p in products}

    # ── 2. Build indexes ─────────────────────────────────────────────────────
    console.rule("[bold]2. Build / load BM25 + Chroma")
    t0 = time.perf_counter()
    bm25 = build_or_load_bm25(products, reset=args.reset)
    console.print(f"  BM25 ready in {time.perf_counter()-t0:.2f}s")

    console.print("[yellow]Building/updating Chroma index (embeddings)...[/yellow]")
    t0 = time.perf_counter()
    build_index(products, reset=args.reset)
    console.print(f"  Chroma ready in {time.perf_counter()-t0:.2f}s")

    # ── 3. Retrieval ─────────────────────────────────────────────────────────
    console.rule(f"[bold]3. Hybrid search — query: [cyan]{args.query}[/cyan]")
    t0 = time.perf_counter()
    bm25_results = bm25.query(args.query, top_k=30)
    vector_results = query_vector(args.query, top_k=30)
    fused_ids = hybrid_search(bm25_results, vector_results, top_n=20)
    elapsed = time.perf_counter() - t0

    console.print(f"  BM25 top-3 : {[pid for pid, _ in bm25_results[:3]]}")
    console.print(f"  Vector top-3: {[pid for pid, _ in vector_results[:3]]}")
    console.print(f"  Fused top-5: {fused_ids[:5]}  ({elapsed:.2f}s)")

    candidates = [id_to_product[pid] for pid in fused_ids if pid in id_to_product]

    # ── 4. Rerank ────────────────────────────────────────────────────────────
    if not args.no_rerank:
        console.rule("[bold]4. LLM rerank (gpt-4o-mini)")
        t0 = time.perf_counter()
        reranked = rerank(
            args.query,
            candidates,
            soft_instructions="preferisce prodotti disponibili, floreali, femminili",
            top_k=args.top_k,
        )
        elapsed = time.perf_counter() - t0
        console.print(f"  Rerank done in {elapsed:.2f}s")
    else:
        reranked = candidates[: args.top_k]

    # ── 5. Results table ─────────────────────────────────────────────────────
    console.rule("[bold]5. Final results")
    table = Table(show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("ID", style="cyan", width=12)
    table.add_column("Title", width=40)
    table.add_column("Price", justify="right", width=14)
    table.add_column("Avail", width=6)
    table.add_column("Tester", width=7)
    table.add_column("Niche", width=6)

    for i, p in enumerate(reranked, 1):
        table.add_row(
            str(i),
            p.product_id,
            p.title[:40],
            p.display_price,
            "✓" if p.effective_available else "✗",
            "T" if p.is_tester else "",
            "N" if p.is_niche else "",
        )

    console.print(table)


if __name__ == "__main__":
    main()
