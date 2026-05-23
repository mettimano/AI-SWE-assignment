# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a take-home assignment to build a **multi-agent AI product recommendation system** for **Lumé**, a fictional Italian multi-brand beauty reseller. The system targets a WhatsApp channel and must handle Italian (and sometimes English) queries with implicit customer intent.

Deliverables required by the assignment (see [README.md](README.md)):
- Working Python 3.11+ code (agent-based recommendation system)
- [DESIGN.md](DESIGN.md) — architecture, RAG tradeoffs, cost analysis, failure modes, scaling
- [EVAL.md](EVAL.md) — evaluation design: metrics, 10+ test cases, calibration, cost/trust
- A short Loom video walkthrough

## Data Files

**[data/catalog.json](data/catalog.json)** — 300 products, ~22k lines. Key fields per product:
- `product_id`, `title` (brand-prefixed), `description` (Italian, HTML markup)
- `min_price_eur` / `max_price_eur`, `available` (boolean)
- `collections` — noisy tags: brand slugs, categories (`profumi`, `make-up`, `trattamenti-viso`), occasions, internal markers (`tester`, `bestseller`, `offerta-30`)
- `variants` — sizes (30ml/50ml/100ml) or shades; `custom_fields` — `tipologia_prodotto`, `ingredienti`, `collezione`
- `productCategory` — Google taxonomy L1–L4

**[data/brand.md](data/brand.md)** — Brand voice and business rules. Must be respected in all LLM responses:
- 2–4 WhatsApp-style sentences, warm/knowledgeable tone, sparse emoji (max 1)
- Italian olfactory vocabulary when relevant (note di testa/cuore/fondo, etc.)
- Never invent product facts; never lead with out-of-stock products
- Budget is a hard filter (€50 means strictly <€50)
- Ideal response: 2–4 recommendations with reasons
- Escalate on: frustration, returns/refunds, order status, B2B requests

## Tech Constraints

- Python 3.11+
- Any LLM provider and framework (OpenAI, Anthropic, LangChain, LlamaIndex, etc.)
- AI coding assistants must be used; reasoning should be visible in commits

## Commands

> These will be defined once the project structure is set up. Add build/run/test commands here as they are created.

## Architecture Guidance

The system should decompose into agents handling distinct responsibilities:
1. **Intent extraction** — parse explicit and implicit signals (budget, occasion, fragrance family, gender, gifting, negation, follow-ups questions)
2. **Retrieval** — search the product catalog; consider hybrid search (keyword + semantic) given noisy `collections` tags
3. **Ranking / filtering** — apply hard constraints (budget, stock), score by relevance
4. **Response generation** — produce brand-voice-compliant WhatsApp messages citing real product data only

Key hard cases to handle: implicit intent ("qualcosa di più luxury?"), gifting queries ("per mia madre, le piacciono i fiori"), negation, ambiguous budget, multilingual input, and escalation triggers.

The 300-product catalog is small enough to fit in context, but retrieval design matters for the evaluation — tradeoffs should be documented in DESIGN.md.
