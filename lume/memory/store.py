"""Atomic JSON read/write for per-user profiles."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from lume.config import USERS_DIR
from lume.memory.profile import KnownPreferences, UserProfile


def _profile_path(user_id: str) -> Path:
    return USERS_DIR / f"{user_id}.json"


def load_or_init(user_id: str) -> UserProfile:
    """Load existing profile or return a blank one. Never raises on missing file."""
    path = _profile_path(user_id)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return UserProfile.model_validate(data)
    return UserProfile(user_id=user_id)


def save(profile: UserProfile) -> None:
    """Atomically write profile to disk (temp file + rename — safe on crash)."""
    path = _profile_path(profile.user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile.updated_at = datetime.now(timezone.utc).isoformat()
    payload = profile.model_dump_json(indent=2)

    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.write(fd, payload.encode("utf-8"))
        os.close(fd)
        os.replace(tmp_path, path)
    except Exception:
        os.close(fd)
        os.unlink(tmp_path)
        raise


def merge_preferences(profile: UserProfile, updates: dict) -> UserProfile:
    """Additively merge preference updates into profile.

    `updates` is a flat dict with any subset of KnownPreferences fields.
    Lists are unioned (deduplicated); scalars overwrite only if truthy.
    Never destructive — existing preferences are kept unless explicitly overridden.
    """
    prefs = profile.known_preferences

    def _union(existing: list, new: list) -> list:
        seen = set(existing)
        return existing + [x for x in new if x not in seen]

    if "fragrance_families" in updates:
        prefs.fragrance_families = _union(prefs.fragrance_families, updates["fragrance_families"])
    if "brands_liked" in updates:
        prefs.brands_liked = _union(prefs.brands_liked, updates["brands_liked"])
    if "brands_disliked" in updates:
        prefs.brands_disliked = _union(prefs.brands_disliked, updates["brands_disliked"])
    if "must_avoid" in updates:
        prefs.must_avoid = _union(prefs.must_avoid, updates["must_avoid"])
    if updates.get("budget_max") is not None:
        prefs.budget_max = updates["budget_max"]
    if updates.get("niche_lean"):
        prefs.niche_lean = True
    if updates.get("gender_lean"):
        prefs.gender_lean = updates["gender_lean"]

    return profile


def record_acceptance(profile: UserProfile, product_ids: list[str]) -> UserProfile:
    """Add accepted recommendation ids; avoid duplicates."""
    seen = set(profile.past_recommendations_accepted)
    profile.past_recommendations_accepted += [p for p in product_ids if p not in seen]
    return profile


def record_rejection(profile: UserProfile, product_ids: list[str]) -> UserProfile:
    """Add rejected recommendation ids; avoid duplicates."""
    seen = set(profile.past_recommendations_rejected)
    profile.past_recommendations_rejected += [p for p in product_ids if p not in seen]
    return profile
