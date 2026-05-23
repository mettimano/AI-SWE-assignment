"""Print availability/tester/niche stats across the catalog."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lume.catalog.loader import load_products
from lume.catalog.normalize import normalize_all
from lume.config import CATALOG_PATH


def main() -> None:
    products = load_products(CATALOG_PATH)
    normed = normalize_all(products)

    total = len(normed)
    available = sum(1 for p in normed if p.effective_available)
    testers = sum(1 for p in normed if p.is_tester)
    niche = sum(1 for p in normed if p.is_niche)
    bands = Counter(p.price_band for p in normed)

    print(f"Total products  : {total}")
    print(f"Available       : {available}  ({available/total:.0%})")
    print(f"Testers         : {testers}  ({testers/total:.0%})")
    print(f"Niche           : {niche}  ({niche/total:.0%})")
    print()
    print("Price bands:")
    for band in ("entry", "mid", "premium", "luxury"):
        n = bands[band]
        print(f"  {band:<8} : {n}  ({n/total:.0%})")
    print()

    no_variants = sum(1 for p in normed if not p.variants)
    print(f"No variants     : {no_variants}")
    no_search_text = sum(1 for p in normed if not p.search_text.strip())
    print(f"Empty search_text: {no_search_text}")


if __name__ == "__main__":
    main()
