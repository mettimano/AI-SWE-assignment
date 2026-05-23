"""Central configuration: model names, paths, thresholds, clarify policy."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"
BRAND_PATH = DATA_DIR / "brand.md"
USERS_DIR = DATA_DIR / "users"
CHROMA_DIR = REPO_ROOT / ".chroma"
CACHE_DIR = DATA_DIR / "cache"


@dataclass(frozen=True)
class ModelConfig:
    intent: str = os.getenv("LUME_MODEL_INTENT", "gpt-4o-mini")
    responder: str = os.getenv("LUME_MODEL_RESPONDER", "gpt-4o")
    judge: str = os.getenv("LUME_MODEL_JUDGE", "gpt-4o-mini")
    rerank: str = os.getenv("LUME_MODEL_RERANK", "gpt-4o-mini")
    embed: str = os.getenv("LUME_MODEL_EMBED", "text-embedding-3-small")


# Approximate USD per 1M tokens (Oct 2025 OpenAI pricing — for cost reporting only).
PRICE_PER_MTOK_IN: dict[str, float] = {
    "gpt-4o": 2.50,
    "gpt-4o-mini": 0.15,
    "text-embedding-3-small": 0.02,
}
PRICE_PER_MTOK_OUT: dict[str, float] = {
    "gpt-4o": 10.00,
    "gpt-4o-mini": 0.60,
    "text-embedding-3-small": 0.0,
}


@dataclass(frozen=True)
class RetrievalConfig:
    bm25_top_k: int = 30
    vector_top_k: int = 30
    fused_top_k: int = 20
    rerank_top_k: int = 8
    final_top_k: int = 4
    rrf_k: int = 60


@dataclass(frozen=True)
class ClarifyPolicy:
    """
    Decision rule:
      - if any missing_critical_field present -> ask (mode=clarify_question), max 2 questions.
      - elif soft axes ambiguous AND at least one hard filter known -> probe (mode=clarify_probe), 2-4 products.
      - else -> answer.
    """

    max_questions: int = 2
    probe_min: int = 2
    probe_max: int = 4
    # Confidence threshold below which we treat soft axes as ambiguous.
    soft_ambiguity_threshold: float = 0.55


@dataclass(frozen=True)
class GuardrailConfig:
    max_emoji: int = 1
    sentence_count_answer: tuple[int, int] = (2, 5)
    sentence_count_clarify_question: tuple[int, int] = (1, 3)
    sentence_count_clarify_probe: tuple[int, int] = (2, 4)
    sentence_count_escalate: tuple[int, int] = (1, 3)
    sentence_count_no_match: tuple[int, int] = (1, 3)


@dataclass(frozen=True)
class AppConfig:
    models: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    clarify: ClarifyPolicy = field(default_factory=ClarifyPolicy)
    guard: GuardrailConfig = field(default_factory=GuardrailConfig)


CONFIG = AppConfig()


def require_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in, "
            "or export the variable in your shell."
        )
    return key
