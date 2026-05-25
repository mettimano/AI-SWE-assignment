# Design

Lumé è un rivenditore italiano di prodotti beauty con circa 300 articoli a catalogo che riceve ordini tramite WhatsApp. Questo canale impone un tipo di esperienza molto specifica: le risposte devono sembrare scritte da una consulente esperta in negozio, non da un motore di ricerca.

Il catalogo è piccolo ma disordinato: le descrizioni dei prodotti sono HTML in italiano, la tassonomia è nascosta dentro slug di collezioni scritti in modo libero e 21 prodotti sono tester con regole di visualizzazione dedicate.

La vera difficoltà non è soltanto recuperare i prodotti giusti, ma riuscire a combinare:

- comprensione dell'intento implicito in italiano,
- capacità di capire quando fare domande e quando invece "intuire",
- gestione coerente del contesto su più messaggi,
- mantenimento rigoroso del tono del brand senza inventare informazioni.

---

## Architettura: un grafo esplicito con LLM ai nodi

Il sistema è una macchina a stati LangGraph. Ogni nodo è una chiamata LLM (intent, router, clarify, responder, rerank) oppure Python deterministico (retrieval, vincoli, guard, persistenza). Gli archi codificano le regole di business.

Questo split è la decisione architetturale centrale. **Il grafo lascia all'LLM lo spazio per essere umano — caldo, idiomatico, conversazionale — mantenendo però il controllo su cosa accade e quando.** L'LLM compone il messaggio, sceglie il tono, interpreta l'intento implicito. Non decide se ripetere la stessa domanda per la terza volta, se raccomandare un prodotto esaurito o se saltare il guard. Quelli sono invarianti del grafo.

In pratica:

- Il router LLM sceglie un'azione ad ogni turno (`extract_intent`, `refine_intent`, `clarify_question`, `selection`, `answer`, `escalate`, `new_topic`). `_enforce_constraints()` confronta la scelta con lo stato — `clarify_count`, `last_action`, presenza di `current_intent`, presenza di prodotti mostrati in precedenza — e la sovrascrive se sta per rompere un invariante.
- Il responder LLM scrive il testo del messaggio. Il guard gira dopo, in modo deterministico, sulla stringa di output. Product ID inventati, marker markdown, OOS in apertura, eccesso di emoji, claim medici — tutto intercettato prima che la risposta esca dal sistema.
- La persistenza in memoria gira come nodo automatico dopo una selezione, non come qualcosa che il router può decidere di saltare.

Il grafo termina in cinque modalità di output:

| Modalità | Trigger | Forma |
|---|---|---|
| `answer` | intent confidente + ≥1 prodotto disponibile | 2–4 frasi che citano 2–4 prodotti |
| `clarify_question` | manca un hard filter | 1 messaggio, ≤2 domande, nessun prodotto |
| `clarify_probe` | hard filter noti, assi soft aperti | framing + 2–4 prodotti che coprono stili diversi |
| `escalate` | resi / rimborsi / B2B / frustrazione | frase empatica + `needs_human=True` |
| `no_match` | zero candidati dopo i filtri hard | scuse + alternativa OOS più vicina |

Un agente a singolo prompt potrebbe fare la maggior parte di questo, ma gli invarianti imposti via prompt falliscono in produzione circa il 5% delle volte. A scala conversazionale è una violazione ogni venti turni. Il costo strutturale del grafo è più codice; il costo a runtime è praticamente zero.

---

## Retrieval: ibrido con un reranker LLM

La pipeline: **BM25 top-30 ⊕ Chroma vector top-30 → RRF (k=60) → top-20 → LLM rerank → top-4**.

**Perché ibrido, non uno dei due.** I titoli beauty italiani hanno prefissi di brand aggressivi ("Acqua di Parma", "Frederic Malle", "T. Byredo") che gli embedding semantici non preservano in modo affidabile — un cliente che chiede "qualcosa di Byredo" ha bisogno di recall esatto. Al contrario, una query come "per mia madre, le piacciono i fiori" non contiene nessuno dei token che combaccerebbero con le descrizioni dei prodotti — lì serve recall semantico. Ognuno dei due approcci, da solo, lascia scoperta una classe di query.

**Perché il rerank LLM in fondo, non filtri deterministici post-hoc.** Gli slug della colonna `collections` del catalogo sono tag di marketing freeform incoerenti, non una tassonomia vera. Un prodotto può essere femminile senza mai apparire in `profumi-donna`. Filtri hard di set-operation su dati incoerenti creano falsi negativi. Il reranker vede il testo completo del prodotto e applica le preferenze soft (brand, famiglia, gender, must_avoid) come istruzioni in linguaggio naturale. I filtri hard sono riservati a ciò che *è* strutturato davvero: prezzo, stock, flag tester.

**Perché `text-embedding-3-small`.** 300 prodotti, query brevi. Il delta di qualità rispetto a modelli di embedding più grandi è invisibile a questa scala; il delta di costo è 5×. Sarebbe ancora la scelta giusta a 10× il catalogo.

---

## Chiarificazione: quando chiedere, quando mostrare

La chiarificazione ha due modalità, scelte dal grafo in base a cosa manca:

- **Ask** quando manca un hard filter (budget, categoria, gender per profumo). Sono campi non indovinabili. Chiedere una volta costa meno che sbagliare.
- **Probe** quando gli hard filter sono noti ma gli assi soft (famiglia, mood, occasione) sono aperti. Mostra 2–4 prodotti che coprono lo spazio dell'ambiguità ("uno fresco, uno legnoso, uno intenso") e lascia che il cliente scelga di pancia.

Il motivo di avere due modalità: il vocabolario olfattivo è specialistico. Una domanda come "preferisci floreale o legnoso?" dà per scontato che il cliente sappia rispondere in quei termini. Quasi nessuno lo sa. Un prodotto probe è autoesplicativo in un modo che una domanda a scelta multipla non può essere.

Il confine vive in un set:

```python
_CRITICAL_FIELDS = {"budget_max", "categories", "gender_lean"}
```

Se manca uno qualsiasi → ask. Altrimenti, se la confidence è < 0.55 → probe. Altrimenti → answer.

Un'eccezione: **i regali vanno in probe anche ad alta confidence quando lo stile del destinatario è sconosciuto.** Sbagliare lo stile di un profumo per uso personale è recuperabile. Sbagliarlo su un regalo no — chi compra non lo può annusare prima di darlo. Quando `gift_recipient` è valorizzato e `fragrance_family` è vuota, il grafo va in probe indipendentemente dalla confidence.

`clarify_count` ha un tetto di 2 per topic. Dopo, viene forzato `answer`. Niente loop infiniti.

L'apertura di una domanda di chiarificazione la genera il responder, non un template. Framing hardcoded ("Certo! Per aiutarti al meglio…") fanno suonare ogni turno come uno script.

---

## Multi-turn

`topic_history: list[dict]` in formato chat OpenAI — sia i turni utente che quelli assistant — viene portato attraverso le chiamate e resettato solo da `new_topic`. Ogni agente che ha bisogno di contesto (router, intent, clarify, responder) lo legge.

Includere i turni assistant è ciò che fa funzionare i riferimenti ordinali. Quando il bot ha appena elencato tre profumi per nome e il cliente risponde "il secondo", l'LLM di refinement ha bisogno di vedere cosa ha detto il bot, non solo cosa ha detto il cliente. Il costo è un prompt doppio; il costo per turno è dominato dal responder, quindi l'impatto è trascurabile.

L'intent viene aggiornato per diff e non per ri-estrazione completa: `refine_intent` ritorna solo i campi che il cliente ha toccato, mergiati in modo additivo su `current_intent`. Dire "più economico" abbassa il budget; non cancella la famiglia olfattiva stabilita in precedenza.

---

## Risposta post-probe: l'anchor di ricerca è il prodotto, non il messaggio

Un probe si chiude con il cliente che sceglie uno dei prodotti mostrati. Il turno successivo deve ritornare una lista di prodotti *simili* — non dettagli su quello scelto.

Il meccanismo: quando `node_selection` esegue il path della probe-selection, imposta `retrieval_query = product.search_text` direttamente. Quella stringa è titolo + descrizione + collezioni + ingredienti — un segnale di similarità denso. `node_retrieve` controlla `last_action == "selection" and last_shown_mode == "clarify_probe"` e usa la query pre-costruita, bypassando il generatore di query LLM. Il responder riceve un messaggio esplicito "Trovami prodotti simili a {title}" così interpreta il turno come una raccomandazione, non come una richiesta di dettaglio.

La scelta strutturale qui è costruire il segnale di retrieval in Python dal catalogo, invece di chiedere all'LLM di derivarlo dalla conversazione. La conversazione, a quel punto, è un proxy povero di quello che il cliente vuole davvero.

---

## Brand voice: tre layer di enforcement

Gli LLM seguono il system prompt la maggior parte delle volte. Ma a volte derivano in linguaggio da assistente generico, e quello uccide la sensazione del commesso WhatsApp competente.

L'enforcement è a layer:

1. **Prompt** — regole esplicite, mode-aware, distillate da `data/brand.md`. Esempi few-shot in puro stile WhatsApp.
2. **Guard** — regex post-hoc sulla stringa di output. Product ID inventati, marker markdown (`**`, `-`, `#`), conteggio emoji, conteggio frasi per modalità, claim medici, OOS in apertura. In caso di violazione hard: rigenera una volta, poi cade su un template sicuro con `needs_human=True`.
3. **Judge** — un asse `whatsapp_feel` nella eval (1–5), gira su ogni test case per tracciare il drift tra una modifica e l'altra.

Il guard lavora sulla stringa di output, non sulle intenzioni dichiarate dal modello. Questo lo rende model-agnostic — cambiare provider o fare fine-tuning del responder non richiede di toccarlo.

---

## Memoria: il vantaggio dell'AI sul commesso umano

Un commesso umano può essere eccellente con il cliente che ha davanti, ma non si ricorda i cento altri clienti entrati il mese scorso. Un assistente AI sì. Questo è il vantaggio strutturale su cui vale la pena progettare: ogni interazione lascia una traccia che migliora la successiva. Fatto bene, il sistema smette di essere un recommender generico e inizia a comportarsi come un consulente personale che già conosce il cliente — la sua fascia di budget, le sue preferenze olfattive, cosa ha comprato, cosa ha rifiutato, a cosa è allergico.

**Cosa viene ricordato.** Dopo ogni raccomandazione accettata, `infer_and_save_preferences` estrae attributi dal prodotto scelto e li merge nel profilo del cliente: famiglie olfattive, brand, fascia di prezzo, niche lean, qualunque cosa esplicitata come hard avoid ("no oud"). Acquisti passati e storia delle raccomandazioni sono tenuti come product ID, così il sistema non ri-suggerisce mai qualcosa che il cliente possiede già o ha esplicitamente rifiutato.

**Come rientra nelle raccomandazioni.** Il profilo viene letto all'inizio di ogni turno e iniettato (come riassunto in prosa redatta, mai JSON grezzo) in tre punti:

- **Estrazione intent** — dà all'LLM dei prior quando il messaggio è muto. "Vorrei un profumo nuovo" con un cliente noto come floreale-niche-budget-80 produce un intent iniziale molto più utile dello stesso messaggio da cold start.
- **Soft instructions del reranker** — spinge i top-4 verso le famiglie che il cliente ha storicamente apprezzato e via da tutto ciò che è in `must_avoid` o `past_recommendations_rejected`.
- **Contesto del responder** — permette al messaggio di richiamare scelte passate in modo naturale ("come il Byredo che avevi preso"), nel modo in cui un commesso che si ricorda di te lo farebbe.

**Perché conta come feature di prodotto, non solo tecnica.** Il momento più costoso in cui perdere un cliente è la seconda visita. Se è costretto a rispiegare tutto quello che aveva detto al sistema l'ultima volta, il sistema ha bruciato la sua fiducia. La memoria persistente trasforma un recommender stateless in qualcosa di più vicino a una relazione — ed è quella relazione che giustifica WhatsApp come canale rispetto a un webshop generico, in primo luogo.

**Storage.** JSON per utente in `data/users/<id>.json`. Scelto al posto di SQLite/Postgres perché:

- Ispezionabile — apri il file, vedi esattamente cosa il sistema pensa di un cliente.
- Scritture atomiche via temp-file + rename gestiscono la sicurezza in scrittura concorrente a questa scala.
- La migrazione è un singolo swap di backend — sostituisci read/write di `store.py` con query `jsonb` e schema, logica di merge e call site non cambiano.

Il merge è additivo, mai distruttivo. "No oud" appende a `must_avoid`; rimuovere una preferenza richiede una negazione esplicita. La memoria dovrebbe inclinare le raccomandazioni future, non bloccare silenziosamente il cliente dentro un profilo di sei mesi fa.

`redact_for_prompt` converte il profilo in un riassunto in prosa da 2 righe prima dell'iniezione — mai JSON grezzo. Nomi di campo come `niche_lean: true` confonderebbero il modello e potrebbero far filtrare la tassonomia interna nelle risposte al cliente.

---

## Costi e latenza

Turno answer mediano: ~$0.012. Clarify: ~$0.001. Escalation: ~$0.0005.

Il responder (`gpt-4o`) è circa il 70% del costo per answer. Tutto il resto usa `gpt-4o-mini`.

Latenza p50 della answer: 8–10s.

---

## Cosa migliorerei dopo

- **Eval set adversarial costruito da chat log reali.** I 18 casi attuali coprono scenari canonici. Non coprono quello che i clienti fanno davvero — typo, messaggi code-switched, frasi a metà. Un set adversarial sarebbe onesto sul tasso di fallimento reale.
- **Fine-tune del responder.** Un LLM fine-tunato sul tono richiesto dall'azienda e su conversazioni clerk-to-customer reali può portare benefici nella qualità dell'output. Si potrebbero anche eliminare i few-shot, riducendo i costi delle chiamate API.
- **Database di produzione.** Il vector DB Chroma attuale non scala bene. Per un sistema in produzione andrebbe usato un vector database come Qdrant o Weaviate. E per le informazioni utente, al posto di un JSON, un database SQL.
