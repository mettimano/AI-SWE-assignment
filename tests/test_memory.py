"""Tests for user memory store."""

from __future__ import annotations

import json
import time

import pytest

from lume.memory.profile import UserProfile, redact_for_prompt
from lume.memory.store import (
    load_or_init,
    merge_preferences,
    record_acceptance,
    record_rejection,
    save,
)


class TestLoadOrInit:
    def test_init_blank_profile(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lume.memory.store.USERS_DIR", tmp_path)
        profile = load_or_init("new_user")
        assert profile.user_id == "new_user"
        assert profile.past_purchases == []

    def test_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lume.memory.store.USERS_DIR", tmp_path)
        profile = load_or_init("giulia")
        profile.known_preferences.budget_max = 90.0
        profile.past_purchases = ["p_001", "p_002"]
        save(profile)

        loaded = load_or_init("giulia")
        assert loaded.known_preferences.budget_max == 90.0
        assert loaded.past_purchases == ["p_001", "p_002"]


class TestAtomicWrite:
    def test_file_valid_json_after_save(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lume.memory.store.USERS_DIR", tmp_path)
        profile = UserProfile(user_id="test")
        save(profile)
        content = (tmp_path / "test.json").read_text(encoding="utf-8")
        parsed = json.loads(content)
        assert parsed["user_id"] == "test"

    def test_no_tmp_files_left_after_save(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lume.memory.store.USERS_DIR", tmp_path)
        profile = UserProfile(user_id="clean")
        save(profile)
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []


class TestMergePreferences:
    def test_union_fragrance_families(self):
        profile = UserProfile(user_id="u")
        profile.known_preferences.fragrance_families = ["floreale"]
        merge_preferences(profile, {"fragrance_families": ["muschiato", "floreale"]})
        assert set(profile.known_preferences.fragrance_families) == {"floreale", "muschiato"}

    def test_budget_overwrites(self):
        profile = UserProfile(user_id="u")
        profile.known_preferences.budget_max = 50.0
        merge_preferences(profile, {"budget_max": 80.0})
        assert profile.known_preferences.budget_max == 80.0

    def test_none_budget_does_not_overwrite(self):
        profile = UserProfile(user_id="u")
        profile.known_preferences.budget_max = 50.0
        merge_preferences(profile, {"budget_max": None})
        assert profile.known_preferences.budget_max == 50.0

    def test_must_avoid_union(self):
        profile = UserProfile(user_id="u")
        profile.known_preferences.must_avoid = ["oud"]
        merge_preferences(profile, {"must_avoid": ["cuoio", "oud"]})
        assert set(profile.known_preferences.must_avoid) == {"oud", "cuoio"}

    def test_niche_lean_sticky(self):
        profile = UserProfile(user_id="u")
        merge_preferences(profile, {"niche_lean": True})
        assert profile.known_preferences.niche_lean is True
        merge_preferences(profile, {"niche_lean": False})
        assert profile.known_preferences.niche_lean is True


class TestRecordAcceptanceRejection:
    def test_record_acceptance_dedup(self):
        profile = UserProfile(user_id="u")
        record_acceptance(profile, ["p_001", "p_002"])
        record_acceptance(profile, ["p_002", "p_003"])
        assert profile.past_recommendations_accepted == ["p_001", "p_002", "p_003"]

    def test_record_rejection_dedup(self):
        profile = UserProfile(user_id="u")
        record_rejection(profile, ["p_010"])
        record_rejection(profile, ["p_010", "p_011"])
        assert profile.past_recommendations_rejected == ["p_010", "p_011"]


class TestRedactForPrompt:
    def test_empty_profile_returns_empty(self):
        profile = UserProfile(user_id="u")
        assert redact_for_prompt(profile) == ""

    def test_contains_budget(self):
        profile = UserProfile(user_id="u")
        profile.known_preferences.budget_max = 80.0
        text = redact_for_prompt(profile)
        assert "80" in text

    def test_contains_must_avoid(self):
        profile = UserProfile(user_id="u")
        profile.known_preferences.must_avoid = ["oud", "cuoio"]
        text = redact_for_prompt(profile)
        assert "oud" in text
        assert "cuoio" in text

    def test_contains_fragrance_families(self):
        profile = UserProfile(user_id="u")
        profile.known_preferences.fragrance_families = ["floreale", "muschiato"]
        text = redact_for_prompt(profile)
        assert "floreale" in text

    def test_full_profile_format(self):
        profile = UserProfile(user_id="giulia")
        profile.known_preferences.fragrance_families = ["floreale"]
        profile.known_preferences.budget_max = 80.0
        profile.known_preferences.niche_lean = True
        profile.known_preferences.must_avoid = ["oud"]
        profile.past_purchases = ["p_006", "p_073"]
        text = redact_for_prompt(profile)
        assert len(text) > 0
        assert "None" not in text
        assert "[]" not in text
