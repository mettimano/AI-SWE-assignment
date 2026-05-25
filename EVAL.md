# Eval

## Cosa misurare, e perché

L'eval di un sistema agentico non è "il modello dà la risposta giusta?". È un test contro **quattro superfici indipendenti** che possono rompersi in modo non correlato: retrieval, decisioni del grafo, qualità della generazione, e fedeltà end-to-end della conversazione. Una eval che le tratta tutte come un unico numero (es. "judge medio 4.0") non fornisce abbastanza informazione per iterare: quando lo score scende, non si sa dove guardare.

L'obiettivo dell'eval è duplice:
1. **Bloccare le regressioni** prima che escano. Ogni PR deve far girare almeno la parte deterministica.
2. **Diagnosticare il fallimento** in modo localizzato. Se il responder peggiora, il retrieval non deve essere coinvolto nella diagnosi.

Quello che segue è il framework che adotterei se questo sistema andasse in produzione. La sezione **Stato attuale** in fondo dichiara onestamente cosa è già implementato e cosa è ancora un gap.

---

## Le quattro superfici

### 1. Retrieval

Misura: dato un intent e una query, il sistema piazza i prodotti rilevanti in cima?

Questa superficie è valutabile **senza l'LLM responder**. Si sostituisce il responder con un'asserzione sul ranking restituito dal reranker. Va isolata perché un bug nel retrieval è invisibile dietro un responder che chiacchiera bene.

**Metriche:**

| Metrica | Cosa misura | Soglia target |
|---|---|---|
| **Recall@k** (k=4, 10, 20) | Il prodotto/i gold compare nei top-k? | Recall@4 ≥ 0.85 |
| **MRR** (Mean Reciprocal Rank) | Posizione del primo prodotto rilevante | ≥ 0.65 |
| **nDCG@4** | Ranking con relevance graduata (0/1/2 — irrilevante/accettabile/ideale) | ≥ 0.75 |
| **Hard-filter compliance** | % di candidati top-4 che rispettano budget, stock, must_avoid | 100% (deterministico) |
| **First-result in-stock rate** | Il primo candidato è effettivamente disponibile | 100% |
| **Latenza retrieval (p50/p95)** | BM25 + vector + RRF + rerank | p50 < 1.5s, p95 < 3s |

**Test set:** ~50 query con gold label set (non un singolo prodotto, ma un set accettabile — il catalogo ha spesso 3–5 prodotti ugualmente rilevanti). Labelling per famiglia/fascia di prezzo/brand. Va aggiornato quando il catalogo cambia.

**Decomposizione per superficie interna:**
- BM25 only — recall@30
- Vector only — recall@30
- RRF fused — recall@20
- LLM rerank — nDCG@4
Senza decomposizione, se il rerank peggiora non si sa se sta in input garbage (RRF) o in giudizio scadente.

### 2. Per-agent

Ogni nodo del grafo è valutabile isolatamente perché ha I/O strutturato.

**Intent extraction.** Dato un messaggio, l'intent estratto è corretto?
- **Field-level F1** per campo (`categories`, `budget_max`, `gender_lean`, `fragrance_family`, `must_avoid`, `gift_recipient`, `language`, `escalate`, `tester_requested`). Riportare per campo singolo — un F1 medio nasconde i campi sbalestrati.
- **Calibrazione della confidence**: Expected Calibration Error (ECE). Quando l'intent dice `confidence=0.4`, è davvero ambiguo il 60% delle volte? Una confidence non calibrata rompe la decisione ask-vs-probe.
- **Missing-field accuracy**: il campo `missing_critical_fields` è popolato correttamente? Falsi negativi qui causano answer prematuri; falsi positivi causano chiarificazioni inutili.

**Router.** Dato lo state della conversazione, l'azione scelta è quella giusta?
- **Action accuracy** con matrice di confusione 7×7 (le sette azioni).
- **Invariant violation rate**: con quale frequenza il router propone un'azione che `_enforce_constraints()` deve sovrascrivere? Una violation rate > 5% indica che il prompt del router non riflette le regole.

**Clarify.** La decisione ask-vs-probe e il payload generato sono ragionevoli?
- **Ask-vs-probe agreement** con human label (su un set di ~30 casi ambigui).
- **Question coverage**: la domanda generata copre tutti i `missing_critical_fields`? (binario)
- **Probe diversity**: distanza coseno media tra le coppie di probe products. Bassa diversity = il probe non sta "coprendo lo spazio".
- **Probe family coverage**: i 2–4 probe coprono almeno 2 famiglie olfattive distinte (per richieste fragranze).

**Responder.** Il messaggio rispetta brand voice e WhatsApp style?
- **LLM judge** su 3 assi, 1–5: `relevance`, `brand_voice`, `whatsapp_feel`. Soglia target: media ≥ 4.0 per asse.
- **Hallucination rate**: product ID citati che non sono in catalogo. Soglia: 0%.
- **OOS-in-opening rate**: primo prodotto citato esaurito. Soglia: 0%.
- **Sentence count compliance** per modalità (configurato in `GuardrailConfig`).
- **Emoji count compliance** (max 1).
- **Markdown leak rate**: `**`, `-`, `#` nell'output. Soglia: 0%.
- **Language match**: lingua della reply == lingua del messaggio cliente.
- **Medical claim rate**: 0%.

**Memory.** Le preferenze persistite influenzano le raccomandazioni successive?
- **Pref usage rate**: con un profilo seeded floral-niche-budget-80, una query muta ("vorrei provare qualcosa di nuovo") produce raccomandazioni che riflettono le prefs in ≥ 80% dei casi.
- **Negative pref respect**: 100% dei prodotti raccomandati rispettano `must_avoid` del profilo.
- **No re-recommend**: 0% di sovrapposizione con `past_recommendations_rejected`.

### 3. End-to-end

Il vero test: il cliente è arrivato a qualcosa di utile?

**Metriche:**
- **Task success rate**: il cliente arriva a una raccomandazione che riflette tutti i vincoli espressi (budget, stock, famiglia, gender, must_avoid, gift constraints)? Binary, human-labeled.
- **Mean turns to resolution**: quanti turni servono per arrivare a una raccomandazione accettabile? Turni meno = UX migliore, ma chiarificazioni troppo brevi peggiorano la qualità — c'è un sweet spot intorno a 2–3 turni.
- **Escalation precision/recall**: quando il sistema fa `escalate=True`, è davvero un caso umano? E quando dovrebbe escalare, lo fa?
- **Cost per resolved conversation**: USD per conversazione end-to-end, breakdown per agent.
- **Latenza per turno** (p50/p95): per modalità (`answer` ~8–10s p50, `clarify_*` ~3–4s, `escalate` ~1s).

### 4. Multi-turn

Casi che la valutazione single-turn non cattura:

- **Refinement coherence**: "più economico" abbassa il budget senza cancellare la famiglia? Test su 10+ casi.
- **Selection resolution**: "il secondo" risolve sul prodotto giusto? Test con riferimenti ordinali, brand, sostantivi descrittivi ("quello floreale").
- **Post-probe answer breadth**: dopo una probe-selection, la reply contiene 2–4 prodotti simili (non 1, non un singolo dettaglio)?
- **Topic switching**: passando da profumi a creme con "ora invece vorrei una crema", il sistema resetta correttamente il contesto?
- **Memory persistence across sessions**: chiusa la sessione, riaperta con stesso `user_id`, il profilo è ancora lì e ancora influenza l'intent?

---

## Test set

**Composizione attuale (18 casi):**
- 12 single-turn (categorie esplicite, gift, budget, negazione, niche, OOS, escalation, tester, inglese, occasione)
- 2 clarify trigger (question e probe)
- 2 multi-turn (refinement, selection)
- 2 returning-user (memoria)

**Cosa aggiungerei:**
- **30+ retrieval-only cases** con gold set (separati dai casi end-to-end).
- **20+ casi adversarial** dai log reali: typo, code-switching it↔en, frasi a metà, prompt injection, domande non-shopping ("come stai?"), claim impliciti ("è ipoallergenico?").
- **10+ casi di multi-turn lunghi** (5+ turni con cambi di topic, refinement multipli, selezioni).
- **Casi di OOD** (out of distribution): richieste per categorie non in catalogo (es. "vorrei un trucco per il viso"), nomi di brand inesistenti.

Il principio: **un test set ben fatto è 70% del lavoro di eval**. La maggior parte dei progetti AI fallisce perché ha un test set che coincide con i casi "facili" su cui il sistema è stato debuggato.

---

## Judge calibration

Il judge LLM è uno strumento utile ma **non si può fidare ciecamente**. Va calibrato contro human label.

**Procedura:**
1. Estrarre 30–50 output rappresentativi (per modalità, per esito).
2. Far rateare a 2+ esseri umani sui 3 assi (`relevance`, `brand_voice`, `whatsapp_feel`), in cieco.
3. Calcolare **Cohen's kappa** judge-vs-human e human-vs-human per asse.
4. Asse accettato per la decisione: κ ≥ 0.6 (substantial agreement).
5. Asse con κ < 0.6: si tiene il numero per tracking ma non si usa per gating in CI.

Senza questo step, "il judge dà 4.2 di media" non significa nulla. Potrebbe essere che il judge è inflazionato (dà sempre 4+) o che misura una dimensione diversa da quella che umani considerano qualità.

**Periodicità:** ricalibrazione ogni 3 mesi o quando si cambia modello del judge.

---

## CI e cadenza

**Per ogni PR:**
- Checks deterministici (budget, stock, hallucination, sentence count, markdown, emoji, language) — tutti i 18 casi. Devono passare al 100%.
- No judge. Troppo lento e troppo costoso per gating PR.

**Nightly:**
- Eval completa con judge sui 18 + retrieval set (50). Output salvato in `eval/runs/`.
- Trend dei 3 assi judge nel tempo (alert se il calo settimanale è > 0.3 punti).

**Pre-release (settimanale o on-demand):**
- Adversarial set + multi-turn lunghi + human spot-check su 20 output.
- Costo aggregato e p95 latency reportati.

---

## Stato attuale e gap

**Implementato:**
- Eval harness end-to-end (`eval/run.py`) sui 18 casi.
- Checks deterministici (mode, budget, stock, sentence count, language, no-markdown, ecc.).
- Judge LLM a 3 assi (`relevance`, `brand_voice`, `whatsapp_feel`).
- Report JSON + Markdown con per-case detail.

**Gap noti, in ordine di impatto:**

1. **Nessun retrieval eval isolato.** Recall@k, MRR, nDCG non sono calcolati. Quando i candidati restituiti dal reranker sono cattivi, ce ne accorgiamo solo se il responder produce una risposta scadente, e a quel punto il segnale è già rumoroso.
2. **Nessuna judge calibration.** I numeri del judge non sono validati contro human label. Vengono tracciati ma non sono affidabili per gating decisions.
3. **Test set piccolo.** 18 casi sono sufficienti per uno smoke test ma non per una stima onesta del tasso di fallimento. Servono 50–100 casi minimo, con almeno 30 adversarial.
4. **Nessuna eval per-agent.** L'intent extractor e il router non hanno test isolati — sappiamo se il sistema sbaglia, non quale componente.
5. **Nessuna eval di calibrazione confidence.** `intent.confidence < 0.55 → probe` è un threshold non validato. Se la confidence non è calibrata, la decisione ask-vs-probe è arbitraria.

L'ordine in cui chiuderei questi gap, dato il vincolo di tempo:
1. Retrieval eval (alto impatto, basso costo — il labelling è il vero costo).
2. Test set espanso con adversarial (alto impatto, costo proporzionale al volume).
3. Per-agent eval per intent extractor (medio impatto, basso costo).
4. Judge calibration (medio impatto, alto costo per il labelling umano).
5. Confidence calibration via reliability diagrams (basso impatto a questa scala, alto costo).

---

## Risultati attuali

L'ultima eval baseline (18 casi, judge attivo):

| Metrica | Valore |
|---|---|
| Casi passati | 18/18 |
| Check pass rate | 100% |
| Latenza media | ~11.5s (include overhead judge) |
| Judge relevance | 3.95 / 5 |
| Judge brand_voice | 3.95 / 5 |
| Judge whatsapp_feel | 4.0 / 5 |

Il pass rate al 100% sui check deterministici non significa "il sistema è perfetto" — significa che il test set è dominato da casi canonici. Le metriche judge intorno a 4.0 sono un segnale debolmente positivo che diventerà un segnale affidabile solo dopo la calibrazione kappa.
