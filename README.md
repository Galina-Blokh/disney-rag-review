# Disney RAG Review App

Actionable insights and recommendations from Disney park reviews using Retrieval-Augmented Generation (RAG) and a lightweight sentiment signal.

## Features

- Clean and preprocess Disney reviews (tokenization, date parsing, categorical encodings).
- Optional sentiment signal using VADER for quick descriptive stats.
- Build a vector index of review texts using OpenAI embeddings.
- Retrieve relevant passages by natural-language questions with simple business filters (park/branch, country, months/seasons, years).
- Generate concise, business-ready Markdown answers with sections: Filters, Stats Summary, Insights, Recommendations, Caveats.
- Persist and reload the vector index (FAISS preferred, NumPy fallback) for fast repeated queries.

## Repository layout

- `notebooks/`
  - `disney_reviews_preprocessing.ipynb` — end-to-end walkthrough: load data, preprocess, (optionally) sentiment, build index, ask questions, save/load index.
- `src/`
  - `data_loader.py` — data loading, cleaning, tokenization, date parsing, encodings, and saving/loading processed datasets.
  - `sentiment_analyzer.py` — VADER-based sentiment scoring helpers.
  - `rag_qa.py` — RAG pipeline (embeddings, indexing, retrieval, QA orchestration).
  - `insights.py` — helper utilities for summarization/insights (if needed by notebook).
- `requirements.txt` — pinned dependencies (NumPy 1.x for FAISS compatibility).
- `.gitignore` — excludes `.env`, processed datasets, FAISS artifacts, and other large binaries.

## Project structure

```text
2llm-home-assignment/
├── notebooks/
│   └── disney_reviews_preprocessing.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── rag_qa.py
│   ├── sentiment_analyzer.py
│   └── insights.py
├── data/                # ignored: raw/processed datasets (see .gitignore)
│   └── processed/       # generated locally if you save outputs
├── artifacts_faiss/     # ignored: persisted vector index (optional)
├── docs/                # optional docs (docx ignored)
├── requirements.txt
├── README.md
├── .gitignore
└── .env                 # ignored: holds OPENAI_API_KEY
```

## Prerequisites

- Python 3.10 (recommended)
- pip / venv
- Git (optional for version control)
- OpenAI API key (required for embeddings and chat): set `OPENAI_API_KEY` in a `.env` file
- Optional: `faiss-cpu` for fast similarity search (falls back to NumPy if unavailable)

## Setup

```bash
# In project root
python3 -m venv 3llm
source 3llm/bin/activate  # On macOS/Linux
# .\3llm\Scripts\activate  # On Windows PowerShell

pip install -r requirements.txt
```

Create `.env` in the project root with your OpenAI key:

```env
OPENAI_API_KEY=sk-...your-key...
```

## Data

Place `disney-reviews.zip` in the project root (git-ignored). Expected CSV columns inside the ZIP:

- `Review_ID`
- `Rating`
- `Year_Month` (e.g., `YYYY-MM`)
- `Reviewer_Location`
- `Review_Text`
- `Branch` (park location)

The `src/data_loader.py` module will read from the ZIP and standardize/clean the dataset.

## Quickstart (Notebook)

Use `notebooks/disney_reviews_preprocessing.ipynb` for an end-to-end run:

1) Load and preprocess the data
2) (Optional) Add VADER sentiment scores for descriptive stats
3) Build a vector index (OpenAI embeddings + FAISS/NumPy backend)
4) Ask questions and get Markdown answers with Filters, Stats Summary, Insights, Recommendations, Caveats
5) Save and reload the index for future runs

## Programmatic usage (Python)

```python
from src.data_loader import load_and_preprocess
from src.rag_qa import build_faiss_index, answer_question

# 1) Load + preprocess raw data from the ZIP
#    Returns processed df ready for embedding
#    (Requires disney-reviews.zip present in the project root)
df, mappings, summary = load_and_preprocess(zip_path="disney-reviews.zip")
print("Summary:", summary)

# 2) Build an index (FAISS preferred if available, else NumPy fallback)
rag = build_faiss_index(
    df,
    prefer_backend="auto",   # "faiss" | "numpy" | "auto"
    batch_size=100,
    show_progress=True,
)

# 3) Ask a question
res = answer_question(
    question="Is spring a good time to visit Disneyland?",
    rag=rag,
    df=df,
    top_k=12,
)

# res["answer"] is Markdown with sections
print(res["answer"])  # Filters, Stats Summary, Insights, Recommendations, Caveats
```

## Persist and reload the index

```python
from src.rag_qa import save_rag_index, load_rag_index

save_rag_index(rag, "artifacts_faiss")
rag2 = load_rag_index("artifacts_faiss")

res = answer_question(
    question="Is Disneyland California usually crowded in June?",
    rag=rag2,
    df=df,
    top_k=12,
)
print(res["answer"])
```

## Configuration tips

- **Backend**: `prefer_backend="auto"` tries FAISS and falls back to NumPy.
- **Embedding model**: `text-embedding-3-small` by default (cost-effective 1536-dim).
- **top_k**: retrieved passages (default 12). Increase for broader context; decrease for speed.
- **Batch size**: embeddings request size (default 100). Adjust for rate limits.
- **Filters**: questions like "in June", "from Australia", "Paris" are parsed into months/countries/branches.

## Architecture Overview



- **`src/data_loader.py`**: Reads the ZIP, standardizes columns, cleans text, tokenizes, parses dates, and encodes categoricals.
- **`src/rag_qa.py`**: Orchestrates embeddings, index build (FAISS preferred, NumPy fallback), retrieval with filters, and Markdown answer generation.
- **`src/sentiment_analyzer.py`**: Optional VADER-based sentiment signal to support Stats.
- **Notebook (`notebooks/disney_reviews_preprocessing.ipynb`)**: End-to-end workflow to build and query the index.

## Mandatory local files (not in Git)

The following files/directories are intentionally git-ignored and must be provided locally:

- `.env` in the project root
  - Contains `OPENAI_API_KEY=...`
  - Never commit this file.
- `disney-reviews.zip` in the project root
  - The raw dataset used by `src/data_loader.py`.
  - Large; excluded from version control.
- `artifacts_faiss/` (optional)
  - Persisted index directory created by `save_rag_index()`; can be reloaded via `load_rag_index()`.
  - Not tracked by Git; regenerate as needed.
- `data/processed/` (optional)
  - Any saved processed datasets (CSV/Parquet) you generate locally.
  - Reproducible from the ZIP and preprocessing steps.

## Security & git hygiene

- Never commit secrets: `.env` is git-ignored.
- Processed datasets and vector index artifacts are ignored by default to keep the repo lean.
- If you need to share artifacts, consider releases or a shared object store.

## Troubleshooting

- **OpenAI key not found**: ensure `.env` exists in the project root and includes `OPENAI_API_KEY`.
- **FAISS not available**: install `faiss-cpu`, or use NumPy fallback (set `prefer_backend="numpy"`).
- **Rate limits**: lower `batch_size` or pause between runs; ensure you’re using a project key with quota.

## Roadmap / Extensions

- Add richer slicing and dashboards (e.g., by park, season, rating bins).
- Experiment with alternative embeddings and re-rankers.
- Incorporate a human-labeled evaluation set for systematic QA quality measurement.

## License

This repository is provided for assessment and internal use. Add a `LICENSE` file if you intend to open-source or redistribute.
