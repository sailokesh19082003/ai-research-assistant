# DocuMind — AI Research & Knowledge Assistant

An enterprise-style backend that ingests PDF documents, auto-classifies them
into technical domains with a trained TensorFlow model, indexes them for
semantic + keyword hybrid search, and answers natural-language questions
with page-cited, context-grounded RAG — plus multi-document summarization,
comparison, and system analytics.

Built as a complete FastAPI service, not a toy demo: it has been run
end-to-end (upload → classify → chunk → embed → index → search → RAG answer
→ summarize → compare → analytics) and ships with a passing test suite.

---

## 1. What's inside

| Capability | How it's implemented |
|---|---|
| PDF ingestion with page-level metadata | `PyMuPDF` parser, background-task upload pipeline |
| Chunking (800–1000 chars, ~150 overlap) | Custom recursive splitter, preserves page numbers across chunk boundaries |
| Document domain classification | A **real, trained** TensorFlow/Keras model (`TextVectorization` → `Embedding` → `GlobalAveragePooling1D` → `Dense(128, relu)` → `Dropout(0.3)` → `Dense(softmax)`), saved as `.h5` + a separate vocabulary artifact. Runs in an **isolated subprocess** so it never conflicts with the vector DB's native libraries. |
| Semantic / keyword / hybrid search | ChromaDB (cosine similarity) + BM25, combined with a tunable weighting |
| RAG QA with citations & memory | Retrieves top-K chunks, forces the LLM to answer only from context, cites file + page, keeps per-session conversation history |
| Summarization | Executive Summary / Technical Summary / Bullet Breakdown / Key Takeaways |
| Multi-document comparison | Methodologies, advantages/disadvantages, similarities, implementation approaches |
| Analytics | Total docs/chunks/queries, category distribution, top-referenced documents |
| Works with or without an OpenAI key | If `OPENAI_API_KEY` is unset, an offline extractive responder keeps every endpoint functional so you can demo/grade it without paying for API calls |
| Works with or without internet access | If the embedding model can't be downloaded from HuggingFace, it automatically falls back to an offline `HashingVectorizer` embedder so ingestion/search never breaks |

Two deliberate engineering decisions worth mentioning in a viva/demo, since
they're the kind of thing that shows real debugging, not just following a
tutorial:

1. **Classifier runs in an isolated subprocess.** On several platforms,
   loading TensorFlow and ChromaDB's native gRPC/protobuf dependencies in the
   *same* process segfaults depending on import order. Rather than fight
   that, classification is dispatched to a short-lived child process
   (`src/ml/_predict_worker.py`) — cheap, robust, and avoids the whole class
   of bug.
2. **Everything degrades gracefully.** No OpenAI key → offline responder.
   No internet for embeddings → hashing-based embedder. No trained
   classifier yet → "Uncategorized" instead of a crash. This means the app
   is demoable/gradable in any environment, then gets strictly better the
   moment you add a real API key / internet access.

---

## 2. Project structure

```
ai-research-assistant/
├── config/settings.py              # All configuration (env-driven)
├── data/
│   ├── raw_documents/               # Uploaded PDFs land here
│   ├── vector_db/                   # ChromaDB persistence
│   └── dataset/sample_dataset.csv   # Labelled training data for the classifier
├── models/                          # tf_classifier.h5, tokenizer.pickle, labels.json (generated)
├── src/
│   ├── database/                    # SQLAlchemy models + session
│   ├── document_processing/         # PDF parser + chunker
│   ├── ml/                          # dataset_prep, train_classifier, predictor, isolated subprocess client
│   ├── vector_store/                # ChromaDB + embeddings + BM25 hybrid search
│   ├── rag/                         # qa_chain, summarizer, comparator, llm_engine
│   └── analytics/                   # metrics.py
├── routes/                          # document / search / analysis / analytics endpoints
├── tests/                           # pytest suite (8 tests, all passing)
├── main.py                          # FastAPI app entrypoint
├── requirements.txt
└── .env.example
```

---

## 3. Step-by-step: run it locally

### Step 1 — Get the project onto your machine
Unzip the delivered file, or if you push it to GitHub first (see §6), clone it:
```bash
cd ai-research-assistant
```

### Step 2 — Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```
> `tensorflow-cpu` and `sentence-transformers` are the two heaviest installs
> (a few hundred MB). If your machine is constrained, the app still runs —
> it just uses the scikit-learn / hashing-vector fallbacks described above.

### Step 4 — Configure environment variables
```bash
cp .env.example .env
```
Open `.env` and optionally paste in a real `OPENAI_API_KEY` if you want
genuine GPT-4o generated answers instead of the offline responder. Leaving
it blank is fine for grading/demo purposes.

### Step 5 — Train the classifier
This produces `models/tf_classifier.h5` and `models/tokenizer.pickle`:
```bash
python -m src.ml.train_classifier
```
You should see per-epoch accuracy logs, ending with a saved-model message.
(A sample labelled dataset across 6 domains — AI, Cyber Security, Cloud
Computing, Robotics, Blockchain, Networking — is already included at
`data/dataset/sample_dataset.csv`; swap in your own labelled data there if
you want a stronger/broader classifier.)

### Step 6 — Run the API server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Open **http://localhost:8000/docs** — this is the auto-generated Swagger UI
where you can try every endpoint interactively (exactly what the assignment
checklist asks for).

### Step 7 — Run the test suite
```bash
pytest tests/ -v
```
All 8 tests should pass.

---

## 4. Trying it out (example requests)

**Upload a PDF:**
```bash
curl -X POST http://localhost:8000/documents/upload -F "file=@/path/to/some.pdf"
```
Response gives you a `doc_id`. Poll its status:
```bash
curl http://localhost:8000/documents/<doc_id>
```
Wait for `"status": "PROCESSED"` — this confirms parsing, classification,
chunking, and vector indexing all completed.

**Ask a question (RAG with citations):**
```bash
curl -X POST http://localhost:8000/search/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What frameworks does this document mention?"}'
```

**Hybrid search:**
```bash
curl -X POST http://localhost:8000/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "cloud infrastructure", "mode": "hybrid", "k": 3}'
```

**Summarize a document:**
```bash
curl http://localhost:8000/analysis/summarize/<doc_id>
```

**Compare two documents:**
```bash
curl -X POST http://localhost:8000/analysis/compare \
  -H "Content-Type: application/json" \
  -d '{"doc_ids": ["<doc_id_1>", "<doc_id_2>"]}'
```

**System analytics:**
```bash
curl http://localhost:8000/analytics/
```

---

## 5. Making it "yours" for submission (customization ideas)

Since your instructor will likely want to see this isn't a copy-paste, a
few genuine, easy-to-defend customizations you can make and explain:

- **Rename the app** — change `APP_NAME` in `config/settings.py` and the
  title in `main.py`. Already suggested: "DocuMind".
- **Swap/expand the classifier's categories** — edit
  `data/dataset/sample_dataset.csv` with your own domains (e.g. add
  "Data Science", "IoT") and retrain (`python -m src.ml.train_classifier`).
- **Tune the hybrid search weighting** — `alpha` in
  `src/vector_store/manager.py`'s `hybrid_search()` controls semantic vs.
  keyword weighting; try 0.4 or 0.8 and compare results.
- **Add a real LLM key** — set `OPENAI_API_KEY` in `.env` for genuinely
  generated (not extractive) answers/summaries.
- **Add your own test documents** — upload a few PDFs on a topic you know
  well and sanity-check the citations point to the right pages.

---

## 6. Publishing / submitting: step-by-step

You have three realistic options depending on what your instructor wants.
Do at least (A); (B) and (C) are for extra credit / a live demo link.

### (A) Push to GitHub (recommended baseline)
```bash
cd ai-research-assistant
git init
git add .
git commit -m "AI Research & Knowledge Assistant - initial implementation"
```
Create an empty repo on GitHub (github.com → New repository → do **not**
initialize with a README), then:
```bash
git remote add origin https://github.com/<your-username>/ai-research-assistant.git
git branch -M main
git push -u origin main
```
Double-check `.env` is **not** committed (it's already in `.gitignore`) —
only `.env.example` should be. Submit the GitHub repo URL.

### (B) Deploy it live (optional, but impressive)
Easiest free options for a FastAPI + SQLite app:

- **Render.com**: New → Web Service → connect your GitHub repo →
  Build command: `pip install -r requirements.txt` →
  Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`.
  Add `OPENAI_API_KEY` under Environment if you have one.
- **Railway.app**: similar flow — connect repo, it auto-detects Python,
  set the start command the same way.

Note: the free tiers of these platforms have limited RAM, and
`tensorflow-cpu` + `sentence-transformers` are both memory-hungry. If the
deploy fails on memory, that's expected — mention in your README that the
app auto-degrades to lighter-weight fallbacks (already built in), or
downgrade to `scikit-learn`-only mode by simply not installing
`tensorflow-cpu` on the deployed instance.

### (C) Package as a submission zip
If your instructor wants a direct file upload instead of a repo link:
```bash
cd ..
zip -r ai-research-assistant-submission.zip ai-research-assistant \
  -x "ai-research-assistant/venv/*" \
  -x "ai-research-assistant/data/vector_db/*" \
  -x "ai-research-assistant/data/raw_documents/*" \
  -x "*__pycache__*"
```
This is exactly the zip already provided to you — see the file list below.

### Before you submit, walk through the checklist from your assignment brief:
- [x] Multi-PDF upload, non-blocking (background task)
- [x] Page-index-preserving text extraction
- [x] Vector similarity search returns relevant chunks
- [x] RAG answers grounded in retrieved text with document/page citations
- [x] TensorFlow model gives real domain classification predictions
- [x] Summarization & comparison endpoints work without hallucinating
      missing facts (offline mode explicitly says so; real LLM mode is
      instructed to do the same)
- [x] Conversation memory persists across turns via `session_id`
- [x] Swagger UI at `/docs` documents every endpoint

---

## 7. Known limitations (be upfront about these if asked)

- The included training dataset is small (36 rows / 6 categories) — enough
  to prove the full pipeline works end-to-end, but you should expand it
  with more labelled examples for a stronger classifier if your grade
  depends on classification accuracy.
- Without an `OPENAI_API_KEY`, answers/summaries come from a simple
  offline extractive fallback, not genuine LLM generation — clearly
  labelled as `[Offline mode]` in every response so it's never mistaken
  for a hallucination-prone real model output.
- Without internet access to HuggingFace, embeddings fall back to a
  hashing-based method that is weaker semantically than the real
  `all-MiniLM-L6-v2` model, though it still gives correct exact/partial
  keyword matches via the BM25 half of hybrid search.
