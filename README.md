# Lumé

A multi-agent WhatsApp-style recommender for **Lumé**, a fictional Italian beauty reseller with ~300 products. Built as a take-home assignment for TextYess — see [ASSIGNMENT.md](ASSIGNMENT.md) for the original brief.

For the architectural decisions and tradeoffs, read [DESIGN.md](DESIGN.md). For how I'd measure the system in production, [EVAL.md](EVAL.md).

---

## Quickstart

**Requirements:** Python 3.11+, an OpenAI API key, and [uv](https://docs.astral.sh/uv/) (recommended) or `pip`.

```bash
# 1. Clone and enter
git clone <repo> && cd AI-SWE-assignment

# 2. Set up the environment
uv sync                              # or: pip install -e .

# 3. Configure the API key
cp .env.example .env
# Edit .env and put your OPENAI_API_KEY

# 4. Build the catalog indices (one-time, ~30s)
python -m lume.cli --build-index

# 5. Try it
python -m lume.cli --repl
```

---

## Usage

### Interactive chat (REPL)

```bash
python -m lume.cli --repl
```

Clean WhatsApp-like output by default — only the message and who's sending it. Type your query, hit enter. Commands during the chat:

| Command | What it does |
|---|---|
| `/quit` | Exit |
| `/reset` | Clear conversation state, keep user profile |
| `/verbose` | Toggle verbose display (mode badge, recommendation table, probe products) |
| `/debug` | Toggle debug info (intent fields, latency) |

### Persistent user (memory)

Add `--user <id>` to attach the conversation to a per-user profile saved at `data/users/<id>.json`. Preferences (families liked, brands avoided, budget band) and accepted/rejected recommendations are learned and reused next session.

```bash
python -m lume.cli --repl --user giulia
```

A seed profile for `giulia` is included to demo the returning-customer flow.

### One-shot query

```bash
python -m lume.cli "vorrei un regalo per mia madre, le piacciono i fiori, budget 80€"
```

### Verbose mode

Shows the mode badge, full recommendation table with prices and stock, probe products, guard warnings:

```bash
python -m lume.cli --repl --verbose
python -m lume.cli --repl --verbose --debug    # also intent + latency
```

---

## Example interactions

**Confident answer:**
```
Tu: vorrei un regalo per mia madre, le piacciono i fiori, budget 80€
Lumé: Per tua madre ti mando tre opzioni floreali sotto i 50€. Il
      Cabotine Rose (€32) è una rosa fresca e biancospino — classico,
      mai fuori posto. ...
```

**Clarification (missing budget + gender):**
```
Tu: voglio un profumo
Lumé: Hai un budget in mente, e lo cerchi per uomo o donna?
```

**Probe (vague but with hard filters known):**
```
Tu: qualcosa di buono per me, budget 80€, da donna
Lumé: Per capire il tuo stile ti mando tre opzioni molto diverse: uno
      floreale e morbido, uno agrumato e fresco, uno legnoso. Qual è
      il registro che senti più tuo?
```

**Multi-turn refinement:**
```
Tu: vorrei un profumo legnoso da uomo, budget 100€
Lumé: [tre proposte legnose under 100€]
Tu: il secondo, qualcosa di simile
Lumé: [altre tre proposte simili al prodotto scelto]
```

**Escalation:**
```
Tu: voglio restituire un ordine
Lumé: Mi dispiace, ti passo subito a un operatore che ti
      ricontatterà al più presto.
```

---

## Project layout

```
lume/
├── agents/           # LangGraph nodes: intent, router, clarify, responder, guard
├── catalog/          # JSON loading, HTML cleanup, normalization
├── retrieval/        # BM25 + Chroma vector + RRF + LLM rerank
├── memory/           # Per-user profile JSON store
├── cli.py            # Typer CLI / REPL
├── config.py         # Models, thresholds, paths
└── schemas.py        # Public Reply / Recommendation / ProbeProduct / ClarifyingQuestion
data/
├── catalog.json      # 300 anonymized products (given)
├── brand.md          # Brand voice rules (given)
└── users/            # Per-user memory files
eval/
├── cases.yaml        # 18 test cases (single + multi-turn + clarify + memory)
├── deterministic.py  # Mode / budget / stock / language / no-markdown checks
├── judge.py          # 3-axis LLM judge (relevance, brand_voice, whatsapp_feel)
└── run.py            # Eval harness
```

---

## Evaluation

```bash
python -m eval.run                   # deterministic checks only (~30s)
python -m eval.run --judge           # + 3-axis LLM judge (~5min, ~$0.10)
python -m eval.run --case C03 C15    # only specific cases
```

Reports drop into `eval/runs/report_<timestamp>.{json,md}`.

Baseline on the 18 included cases: 100% deterministic checks passing, judge means 3.95 / 3.95 / 4.0 on relevance / brand_voice / whatsapp_feel.

See [EVAL.md](EVAL.md) for the full eval philosophy and what I'd add for production.



---

## Troubleshooting

**"OPENAI_API_KEY is not set"** — copy `.env.example` to `.env` and fill it in.

**Slow first query** — Chroma builds its index on first run (~30s). Subsequent runs are warm.

**`--build-index` fails on the BeautifulSoup parser** — install the optional `lxml` parser: `uv pip install lxml` (or `pip install lxml`).

**Want to wipe a user's memory** — `rm data/users/<id>.json`.
