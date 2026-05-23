"""Tests for catalog normalization."""

from __future__ import annotations

import pytest

from lume.catalog.models import CustomField, Product, Variant
from lume.catalog.normalize import normalize


def _product(**kwargs) -> Product:
    defaults = dict(
        product_id="p_001",
        title="Test Product",
        description="",
        collections=[],
        min_price_eur=0.0,
        max_price_eur=50.0,
        available=True,
        variants=[Variant(variant_id="v1", title="50ml", price_eur=50.0, available=True)],
        custom_fields=[],
        productCategory=None,
    )
    defaults.update(kwargs)
    return Product.model_validate(defaults)


class TestEffectiveAvailable:
    def test_product_available_variant_available(self):
        p = _product(available=True, variants=[
            Variant(variant_id="v1", title="50ml", price_eur=50.0, available=True)
        ])
        assert normalize(p).effective_available is True

    def test_product_available_no_available_variant(self):
        p = _product(available=True, variants=[
            Variant(variant_id="v1", title="50ml", price_eur=50.0, available=False)
        ])
        assert normalize(p).effective_available is False

    def test_product_unavailable_variant_available(self):
        p = _product(available=False, variants=[
            Variant(variant_id="v1", title="50ml", price_eur=50.0, available=True)
        ])
        assert normalize(p).effective_available is False

    def test_no_variants(self):
        p = _product(available=True, variants=[])
        assert normalize(p).effective_available is False

    def test_mixed_variants(self):
        p = _product(available=True, variants=[
            Variant(variant_id="v1", title="30ml", price_eur=40.0, available=False),
            Variant(variant_id="v2", title="50ml", price_eur=60.0, available=True),
        ])
        assert normalize(p).effective_available is True


class TestIsTester:
    def test_tester_in_collections(self):
        p = _product(collections=["parfum", "tester"])
        assert normalize(p).is_tester is True

    def test_title_prefix_T(self):
        p = _product(title="T. Chanel N°5 EDP 50ml")
        assert normalize(p).is_tester is True

    def test_normal_product(self):
        p = _product(title="Chanel N°5 EDP 50ml", collections=["parfum"])
        assert normalize(p).is_tester is False

    def test_title_starts_with_T_no_dot(self):
        # "T " without dot should not trigger
        p = _product(title="The Fragrance 50ml", collections=[])
        assert normalize(p).is_tester is False


class TestIsNiche:
    def test_nicchia_in_collection(self):
        p = _product(collections=["profumi-nicchia", "unisex"])
        assert normalize(p).is_niche is True

    def test_no_nicchia(self):
        p = _product(collections=["parfum", "donna"])
        assert normalize(p).is_niche is False

    def test_empty_collections(self):
        p = _product(collections=[])
        assert normalize(p).is_niche is False


class TestHTMLStripping:
    def test_strips_html_tags(self):
        p = _product(description="<p>A <strong>bold</strong> scent.</p>")
        assert normalize(p).cleaned_description == "A bold scent."

    def test_decodes_html_entities(self):
        p = _product(description="Notes of oud &amp; rose.")
        assert normalize(p).cleaned_description == "Notes of oud & rose."

    def test_plain_text_unchanged(self):
        p = _product(description="Plain description.")
        assert normalize(p).cleaned_description == "Plain description."


class TestSearchText:
    def test_contains_title(self):
        p = _product(title="Byredo Bal d'Afrique")
        assert "Byredo Bal d'Afrique" in normalize(p).search_text

    def test_contains_collections(self):
        p = _product(collections=["profumi-nicchia", "donna"])
        st = normalize(p).search_text
        assert "profumi-nicchia" in st
        assert "donna" in st

    def test_contains_custom_fields(self):
        p = _product(custom_fields=[
            CustomField(key="tipologia_prodotto", value="Eau de Parfum"),
            CustomField(key="ingredienti", value="iris, musk, cedar"),
        ])
        st = normalize(p).search_text
        assert "Eau de Parfum" in st
        assert "iris, musk, cedar" in st

    def test_no_none_in_search_text(self):
        p = _product(description="", collections=[], custom_fields=[])
        st = normalize(p).search_text
        assert "None" not in st


class TestPriceBand:
    @pytest.mark.parametrize("price, expected", [
        (20.0, "entry"),
        (39.99, "entry"),
        (40.0, "mid"),
        (99.99, "mid"),
        (100.0, "premium"),
        (199.99, "premium"),
        (200.0, "luxury"),
        (500.0, "luxury"),
    ])
    def test_price_band(self, price, expected):
        p = _product(max_price_eur=price)
        assert normalize(p).price_band == expected
