# Banking FAQ Chatbot

A fully local Retrieval-Augmented Generation (RAG) chatbot that answers general
banking questions from a CSV dataset of FAQs. Everything runs on your own
machine: no API keys, no cloud services, no data leaves the computer.

- **Embeddings:** `all-MiniLM-L6-v2` via Sentence Transformers
- **Vector database:** ChromaDB (persistent, built once)
- **Language model:** Ollama, default `llama3.2:3b`
- **Interface:** Streamlit

---

## 1. How it works

```
  OFFLINE (only when the dataset changes)
  ------------------------------------------------------------------
  banking_knowledge_base_1000.csv
        |  document_loader.py   one document per row
        v
  documents  ->  chunker.py  ->  chunks  ->  embeddings.py  ->  vector_store.py
                                                          ->  chroma_db/  (persisted)
  Run with:  python src/ingest.py


  RUNTIME (every question)
  ------------------------------------------------------------------
  question
        |  embeddings.py  embed the QUESTION only  (~10 ms)
        v
  chroma_db  ->  top_k=3 FAQ chunks  (~2-30 ms)
        |
        v
  rag.py  ->  llm.py  ->  Ollama  ->  answer
```

The single most important design decision: **document embeddings are computed
offline and stored in ChromaDB.** The chatbot never re-embeds the dataset and
never rebuilds the database. At runtime it embeds one short string — the user's
question.

### Project layout

```
banking_chatbot/
├── app/
│   └── streamlit_app.py     Web UI. Caches the expensive objects.
├── src/
│   ├── config.py            All paths, model names and env-var settings.
│   ├── perf.py              Optional step-by-step timing (BANKING_TIMING=1).
│   ├── document_loader.py   CSV -> documents.
│   ├── chunker.py           Documents -> retrieval chunks.
│   ├── embeddings.py        Cached Sentence Transformers model.
│   ├── vector_store.py      ChromaDB wrapper.
│   ├── retriever.py         Question -> best matching chunks.
│   ├── llm.py               Ollama wrapper + prompts.
│   ├── rag.py               Pipeline orchestration + CLI chat loop.
│   ├── ingest.py            OFFLINE: rebuild the vector database.
│   └── test_retriever.py    Manual retrieval check.
├── tests/
│   ├── test_rag.py          End-to-end answer quality checks.
│   ├── test_error_handling.py
│   └── test_app_smoke.py    Streamlit app smoke test.
├── data/raw/                Source CSV (tracked by git).
├── chroma_db/               Generated vector database (rebuildable).
├── requirements.txt
├── .env.example             Copy to .env to override settings.
├── dataset_preprocessing.ipynb   Original exploration notebook (kept, not used at runtime).
└── chatbot_documentation.docx    Original write-up (kept).
```

> `dataset_preprocessing.ipynb` is kept for reference only. `src/ingest.py`
> replaces it: it does the same job with logging, error handling and a stable
> chunk ID scheme, and it is idempotent so re-running is safe.

---

## 2. Setup (Windows)

```powershell
# 1. Create and activate a virtual environment
python -m venv chatbot_EN
.\chatbot_EN\Scripts\Activate.ps1

# 2. Install the Python packages
pip install -r requirements.txt

# 3. Install Ollama (separate program, not a pip package)
#    Download from https://ollama.com and start it, then:
ollama pull llama3.2:3b
```

Ollama must be running before you start the chatbot.

---

## 3. Running it

### Build the vector database (once, and again only if the CSV changes)

```powershell
python src\ingest.py
```

If the dataset shrank and you want a completely fresh database:

```powershell
python src\ingest.py --reset
```

Re-running without `--reset` is safe: chunk IDs are deterministic
(`banking_0`, `banking_1`, ...), so it updates existing rows instead of
duplicating them.

### Start the chat interface

```powershell
streamlit run app\streamlit_app.py
```

The browser opens at <http://localhost:8501>.

### Chat in the terminal instead

```powershell
python src\rag.py
```

### See where the time goes

```powershell
$env:BANKING_TIMING = "1"
streamlit run app\streamlit_app.py
```

Each step is then logged with its duration, e.g.

```
TIMING  pipeline init          0.812s
TIMING  chroma client           0.780s
TIMING  retrieval               0.004s
TIMING  llm generation          9.104s
```

Timing is off by default, so normal runs stay quiet.

---

## 4. Configuration

Every setting lives in `src/config.py` and can be overridden by an environment
variable. Copy `.env.example` to `.env` to set them permanently:

```powershell
copy .env.example .env
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `BANKING_CHROMA_DIR` | `chroma_db` | Where the vector database lives |
| `BANKING_DATASET` | `data/raw/banking_knowledge_base_1000.csv` | Source CSV |
| `BANKING_COLLECTION` | `banking_faq` | ChromaDB collection name |
| `BANKING_EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformers model |
| `BANKING_LLM_MODEL` | `llama3.2:3b` | Default Ollama model |
| `BANKING_TOP_K` | `3` | Chunks given to the LLM as context |
| `BANKING_OLLAMA_HOST` | `http://localhost:11434` | Ollama server |
| `BANKING_OLLAMA_KEEP_ALIVE` | `30m` | How long Ollama keeps the model in RAM |
| `BANKING_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BANKING_TIMING` | off | `1` logs per-step timings |

`.env` is gitignored. This project needs no secrets, so `.env` only ever holds
tuning values.

---

## 5. Testing

```powershell
python src\test_retriever.py     # prints the top-3 chunks for one question
python tests\test_rag.py         # end-to-end answer checks
python tests\test_error_handling.py
python tests\test_app_smoke.py   # loads the Streamlit app headlessly
```

Each script prints `PASS` / `FAIL` per check and exits non-zero on failure.

---

## 6. Why it is fast

Startup cost is paid **once per process** and then reused:

| Component | How it is kept loaded |
| --- | --- |
| Sentence Transformers model | `@lru_cache` on `get_embedding_model()` |
| ChromaDB handle | `@st.cache_resource` on `get_retriever()` |
| Ollama client + model | `@st.cache_resource` on `get_llm()`, plus `keep_alive` |
| Conversation history | `st.session_state` |

Four details that matter:

1. **`local_files_only=True`** when loading the embedding model skips a
   Hugging Face Hub round-trip on every load. Measured in a cold process:
   import + load took **19.8-21.5 s** online versus **11.2-13.2 s** offline,
   so roughly **8.6 s saved per load**. If the model has never been downloaded,
   `embeddings.py` falls back to downloading it.
2. **`get_embedding_model()` is `@lru_cache`d with a `None` default.** The
   original code used the model name as the default argument, which created two
   cache keys for one model and therefore two loads. Measured cost of building
   the pipeline three more times in one process: **~25.3-26.1 s before,
   0.7-1.7 s now** (~8.5 s per rebuild, down to ~0.4 s).
3. **`embedding_function=None`** is passed to ChromaDB so it never loads its own
   default ONNX embedding model. This is a correctness guarantee (we always
   supply our own vectors); in chromadb 1.5.9 it showed **no measurable speed
   difference**, so no speed claim is made for it.
4. **The LLM is warmed up at startup**, not on the first question. The original
   `warm_up()` sent an empty prompt, which Ollama answers immediately without
   loading any weights, so the cost simply moved to the first question. The
   warm-up now generates one real token with `num_predict=1`.

Most of the remaining startup time is `import sentence_transformers` (torch
plus transformers, ~10-11 s on a CPU-only laptop). That is a one-time cost for
the whole Streamlit process, not per question.

Answer generation is the other large cost. On a machine with no GPU, a 3B
model generates a few tokens per second, so a good answer takes 3-14 seconds.
That is the model doing CPU inference, not the code being slow.

### Measured end-to-end

Medians of 5 alternating runs of the old and new app, with `ollama stop` called
before every run so both versions pay the same cold model load:

| | startup | 1st answer | first answer total | 2nd question |
| --- | --- | --- | --- | --- |
| Original | 13.35 s | 35.57 s | 48.92 s | 11.07 s |
| Optimized | 22.02 s | 20.86 s | **41.71 s** | 9.43 s |
| Change | +8.67 s | -14.71 s | **-7.21 s (-15%)** | -1.64 s |

Startup is intentionally longer: the loading work happens under a spinner before
the user types anything, so the *first answer* is what gets faster. Worst
observed first answer dropped from 55.32 s to 44.12 s. Retrieval itself is
0.002-0.03 s.

---

## 7. Design decisions

- **One CSV row = one chunk.** A Question/Answer pair is already a small,
  self-contained knowledge unit; splitting it further fragments the answer and
  hurts retrieval.
- **`top_k=3`.** Enough context to answer most questions without flooding the
  prompt.
- **Guarded answers.** Questions about personal balances or statements are
  refused before retrieval, because an FAQ dataset can never answer them.
- **No stack traces in the UI.** Failures are logged in full to the terminal
  and summarised in friendly words in the chat window.

---

## 8. Troubleshooting

| Symptom | Fix |
| --- | --- |
| "Vector database not found" | `python src\ingest.py` |
| "Ollama is not available" | Start Ollama, then `ollama pull llama3.2:3b` |
| First answer is slow | Expected on a cold model; the warm-up should make later ones quicker |
| Answers say "I couldn't find information" | The FAQ dataset does not cover that topic |
| `ModuleNotFoundError: sentence_transformers` | `pip install -r requirements.txt` |