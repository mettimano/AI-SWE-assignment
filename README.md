# TextYess — AI Engineer Take-Home

Thanks for taking the time. This is designed to look like the actual work you'd do at TextYess, just weekend-sized.

**Deadline**: 3 calendar days from when we sent this  
**Deliverables**: Git repo + `DESIGN.md` + `EVAL.md` + Loom (≤5 min)

---

## The problem

**Lumé** is a fictional multi-brand Italian beauty reseller. Customers message on WhatsApp with things like:

- *"cerco un profumo caldo per l'inverno, qualcosa di speziato o con oud"*
- *"un regalo per mia madre, le piace l'odore dei fiori, budget 50€"*
- *"avete dei tester di profumi da donna a meno di 50 euro?"*
- *"qualcosa di più luxury?"* (follow-up)

**Your job**: build a system that understands what the customer needs — including implicit needs — and recommends the right products. The recommendation should feel like it came from a knowledgeable *profumiere*.

---

## What we give you

| File | What it is |
|---|---|
| `catalog.json` | ~300 anonymized products from a real Italian beauty reseller: fragrances, skincare, ancillaries. Price, variants, stock, collections. Descriptions in Italian. |
| `brand.md` | Brand voice and business rules. |

`starter/` has a minimal `pyproject.toml` + `.env.example`. Use it or ignore it.

---

## What you build

**1. Agent-based recommendation system.** Decompose into specialized agents (intent extractor, retriever, ranker, explainer — your call). Takes a customer message + optional prior turns, returns recommended products with reasons and a response text. Exact output shape is up to you — defend it in `DESIGN.md`.

**2. Retrieval / RAG layer.** Your call on chunking, embeddings, hybrid vs. pure vector, re-ranking. Tell us why.

**3. Evaluation plan (`EVAL.md`, ~1 page).** You don't need to implement the eval — we want to see how you'd *design* one. Cover:
- **Metrics**: what would you measure and why (at least 2)? How would each be computed (deterministic check, LLM judge, human, hybrid)?
- **Test cases**: 10+ example queries you'd include, split across happy paths and hard cases (implicit intent, gifting, budget/stock, ambiguity, negation). For each, the kind of "gold" signal you'd use.
- **What good looks like** per metric — thresholds, calibration, failure-mode analysis plan.
- **Cost + trust**: roughly what would it cost to run, and how would you know the eval itself is trustworthy (e.g. LLM-judge calibration against human)?

**Don't skip this. We care more about how you'd measure than about what you'd build.**

**4. `DESIGN.md`** (~2 pages): architecture and why, RAG tradeoffs, cost per query, known failure modes, how this scales to 100 merchants with 5k–50k products each. Plus a short **"Fine-tuning thought"** paragraph: *if you could fine-tune one model for this project, which model, on what data, for what objective, and why?* No need to actually do it — just one paragraph of thinking.

**5. Loom (≤5 min)**: architecture walkthrough, demo on 2–3 queries (include at least one hard one), one tradeoff you agonized over.

---

## Rules

- Python 3.11+, any LLM provider, any framework.
- **AI coding assistants are mandatory.** We use them every day and expect you to as well. Use them with judgment — we want to see your reasoning in commits and `DESIGN.md`, not just a clean repo.
- Don't over-build. A small, well-measured system beats a sprawling half-built one.
- Report cost honestly.

---

## Stretch (skip if tight on time)

- **"No good match" path** — what does the system do when the catalog can't help?
- **Multi-turn refinement** — "cheaper", "something warmer" threading correctly
- **Adversarial eval set** — failure modes you found + mitigations
- **Implicit memory** — prior purchases mentioned earlier in the conversation

---

## How we grade

- **Agent system design** — is the decomposition clear? does each agent do one thing well? is the routing defensible?
- **Intent understanding** — implicit needs handled, constraints respected, ambiguity dealt with gracefully
- **Retrieval + ranking** — right products in the top 3 on most queries; hard cases attempted honestly
- **Evaluation design** — metrics that would actually measure something meaningful; test cases covering real hard scenarios; honest about how you'd know the eval works
- **Communication** — `DESIGN.md` is scannable with real tradeoffs; Loom is crisp; code is legible

We don't grade on UI, CI/CD, infra, model choice, or framework vs. raw SDK.

---

## Deliverables

1. **Git repo** (public or private-invite to `@valdo99` and `@luisbeqja`)
2. **Loom URL** (unlisted is fine)

`DESIGN.md` and `EVAL.md` at the root.

---

## Questions

Email `luisb@textyess.com`, cc `edvaldo@textyess.com`. We'd rather spend 5 minutes clarifying than have you guess wrong.

Good luck — we're excited to see how you think.