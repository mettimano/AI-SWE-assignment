"""Load catalog.json → list[Product], stripping HTML and decoding entities."""

from __future__ import annotations

import json
from pathlib import Path

from bs4 import BeautifulSoup

from .models import Product


def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities. Returns plain UTF-8 text."""
    if not text:
        return text
    if "<" not in text and "&" not in text:
        return text
    raw = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    return " ".join(raw.split())


def load_products(catalog_path: Path) -> list[Product]:
    """Parse catalog.json into validated Product objects."""
    with open(catalog_path, encoding="utf-8") as fh:
        raw: list[dict] = json.load(fh)
    return [Product.model_validate(item) for item in raw]
