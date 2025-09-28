"""
RAG (Retrieval-Augmented Generation) QA system for Disney reviews.

Features:
- Build a FAISS vector index over cleaned review texts (uses OpenAI embeddings), with NumPy fallback.
- Parse natural language questions to detect metadata filters (park/Branch, country, months/seasons, years).
- Retrieve top-k relevant passages, constrained by metadata filters when provided.
- Combine retrieved context + computed stats + metadata to answer with OpenAI gpt-4o-mini.
- Save and load the vector index to disk (FAISS or NumPy embeddings) for faster, lighter runs.
- Optional tqdm progress bars during embedding.

Dependencies:
- faiss-cpu (optional; module gracefully falls back to NumPy-only retrieval)
- openai
- python-dotenv
- numpy, pandas
- tqdm (optional for progress bars)

Expected input DataFrame `df` is the output of preprocessing + sentiment and contains:
- Columns: Review_ID, Branch, Reviewer_Location, Cleaned_Review_Text, Review_Date, Year, Month,
          Rating, Sentiment_Label, Review_Tokens
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

import os
import re

import numpy as np
import pandas as pd

# FAISS is optional. Import lazily and provide a NumPy fallback.
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"  # 1536-dim, cost-effective
DEFAULT_CHAT_MODEL = "gpt-4o-mini"


@dataclass
class RAGIndex:
    # One of the following backends will be populated:
    index: Optional["faiss.IndexFlatIP"]  # type: ignore[name-defined]
    matrix: Optional[np.ndarray]  # normalized embeddings (n, dim) for NumPy fallback
    id2meta: List[Dict[str, object]]  # meta per vector in same order
    dim: int
    backend: Literal["faiss", "numpy"]
    embedding_model: str = EMBEDDING_MODEL


def _get_openai_client(api_key: Optional[str] = None, env_path: Optional[str] = None) -> OpenAI:
    if api_key is None:
        if env_path:
            load_dotenv(env_path, override=False)
        else:
            dotenv_path = find_dotenv(usecwd=True)
            if dotenv_path:
                load_dotenv(dotenv_path=dotenv_path, override=False)
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not available. Set it in environment or .env (OPENAI_API_KEY or OPEN_AI_KEY)."
        )
    return OpenAI(api_key=api_key)


def _tqdm(iterable, **kwargs):
    try:
        from tqdm.auto import tqdm  # type: ignore
        return tqdm(iterable, **kwargs)
    except Exception:
        return iterable


def _embed_texts(
    client: OpenAI,
    texts: List[str],
    model: str = EMBEDDING_MODEL,
    batch_size: int = 100,
    show_progress: bool = False,
) -> np.ndarray:
    """Embed a list of texts using OpenAI embeddings API.
    Returns (n, dim) float32 array normalized for cosine similarity with inner product.
    """
    vecs: List[np.ndarray] = []
    it = range(0, len(texts), batch_size)
    if show_progress:
        it = _tqdm(it, desc="Embedding reviews", unit="batch")
    for i in it:
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        # API returns list in same order
        arr = np.array([item.embedding for item in resp.data], dtype=np.float32)
        # Normalize to unit length for cosine similarity with inner product
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        arr = arr / norms
        vecs.append(arr)
    if not vecs:
        return np.zeros((0, 1536), dtype=np.float32)
    return np.vstack(vecs)


def build_faiss_index(
    df: pd.DataFrame,
    text_col: str = "Cleaned_Review_Text",
    api_key: Optional[str] = None,
    env_path: Optional[str] = None,
    embedding_model: str = EMBEDDING_MODEL,
    prefer_backend: Literal["auto", "faiss", "numpy"] = "auto",
    batch_size: int = 100,
    show_progress: bool = True,
) -> RAGIndex:
    """Build a vector index over the review texts (FAISS preferred, NumPy fallback).

    Returns a RAGIndex with FAISS index and metadata for each vector.
    """
    if text_col not in df.columns:
        raise ValueError(f"{text_col} not found in DataFrame. Ensure preprocessing was run.")

    texts = df[text_col].fillna("").astype(str).tolist()
    client = _get_openai_client(api_key=api_key, env_path=env_path)
    X = _embed_texts(client, texts, model=embedding_model, batch_size=batch_size, show_progress=show_progress)
    dim = X.shape[1]

    # Select backend and build index
    faiss_index = None
    backend: Literal["faiss", "numpy"]

    if prefer_backend == "numpy":
        backend = "numpy"
    elif prefer_backend == "faiss":
        if not _FAISS_AVAILABLE:
            raise RuntimeError("prefer_backend='faiss' requested but faiss is not available.")
        try:
            faiss_index = faiss.IndexFlatIP(dim)  # type: ignore[attr-defined]
            faiss_index.add(X)
            backend = "faiss"
        except Exception as e:
            raise RuntimeError(f"Failed to build FAISS index: {e}")
    else:  # auto
        if _FAISS_AVAILABLE:
            try:
                faiss_index = faiss.IndexFlatIP(dim)  # type: ignore[attr-defined]
                faiss_index.add(X)
                backend = "faiss"
            except Exception:
                faiss_index = None
                backend = "numpy"
        else:
            backend = "numpy"

    # Build metadata aligned to vectors
    id2meta: List[Dict[str, object]] = []
    cols = [
        "Review_ID",
        "Branch",
        "Reviewer_Location",
        "Year",
        "Month",
        text_col,
    ]
    for _, row in df[cols].iterrows():
        id2meta.append({
            "Review_ID": row.get("Review_ID"),
            "Branch": row.get("Branch"),
            "Reviewer_Location": row.get("Reviewer_Location"),
            "Year": int(row.get("Year")) if pd.notna(row.get("Year")) else None,
            "Month": int(row.get("Month")) if pd.notna(row.get("Month")) else None,
            "text": row.get(text_col, ""),
        })

    # For NumPy backend, keep the normalized matrix to score with dot products
    matrix = None if backend == "faiss" else X

    return RAGIndex(
        index=faiss_index,
        matrix=matrix,
        id2meta=id2meta,
        dim=dim,
        backend=backend,
        embedding_model=embedding_model,
    )


# -------- Filter parsing --------
_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_SEASONS = {
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "fall": [9, 10, 11],
    "autumn": [9, 10, 11],
    "winter": [12, 1, 2],
}


def parse_question_filters(question: str, df: pd.DataFrame) -> Dict[str, object]:
    """Extract simple filters from natural language: park Branch, reviewer country, months/seasons, years.

    Returns dict with optional keys: branch, country, months (list[int]), years (list[int]), topics (list[str]).
    """
    q = question.lower()
    out: Dict[str, object] = {"topics": []}

    # Identify topics from keywords
    if any(k in q for k in ["crowded", "busy", "queue", "line", "wait time", "wait times"]):
        out["topics"].append("volume")
    if any(k in q for k in ["friendly", "rude", "staff", "service", "helpful", "rude staff"]):
        out["topics"].append("staff_experience")
    if any(k in q for k in ["good time", "best time", "season"]):
        out["topics"].append("timing")

    # Branch detection (exact match on known branches, case-insensitive substring)
    branches = sorted(set(str(b) for b in df["Branch"].dropna().unique()))
    lower_map_branch = {b.lower(): b for b in branches}
    found_branch = None
    for lb, orig in lower_map_branch.items():
        if lb in q:
            found_branch = orig
            break
    if found_branch:
        out["branch"] = found_branch

    # Country detection
    countries = sorted(set(str(c) for c in df["Reviewer_Location"].dropna().unique()))
    lower_map_country = {c.lower(): c for c in countries}
    found_country = None
    # Prefer phrases like 'from X', 'in X'
    m = re.search(r"\bfrom\s+([a-zA-Z \-]+)", q)
    cand = m.group(1).strip().lower() if m else None
    if cand and cand in lower_map_country:
        found_country = lower_map_country[cand]
    else:
        for lc, orig in lower_map_country.items():
            if lc in q:
                found_country = orig
                break
    if found_country:
        out["country"] = found_country

    # Month names and seasons
    months: List[int] = []
    for name, mnum in _MONTHS.items():
        if re.search(rf"\b{name}\b", q):
            months.append(mnum)
    for sname, mlist in _SEASONS.items():
        if re.search(rf"\b{sname}\b", q):
            months.extend(mlist)
    if months:
        out["months"] = sorted(sorted(set(months)), key=lambda x: (x-1) % 12)

    # Years
    years = [int(y) for y in re.findall(r"\b(20\d{2}|19\d{2})\b", q)]
    if years:
        out["years"] = sorted(set(years))

    return out


def _apply_filters(df: pd.DataFrame, filters: Dict[str, object]) -> pd.DataFrame:
    out = df
    if "branch" in filters:
        out = out[out["Branch"].astype(str) == str(filters["branch"])]
    if "country" in filters:
        out = out[out["Reviewer_Location"].astype(str) == str(filters["country"])]
    if "months" in filters:
        out = out[out["Month"].isin(filters["months"]) ]
    if "years" in filters:
        out = out[out["Year"].isin(filters["years"]) ]
    return out


def _summarize_filtered(df: pd.DataFrame) -> str:
    if len(df) == 0:
        return "No matching reviews for the specified filters."
    parts = []
    parts.append(f"Records: {len(df)}")
    # Sentiment distribution
    if "Sentiment_Label" in df.columns:
        vc = df["Sentiment_Label"].value_counts(dropna=False)
        total = int(vc.sum())
        pos = int(vc.get("positive", 0))
        neg = int(vc.get("negative", 0))
        neu = int(vc.get("neutral", 0))
        parts.append(f"Sentiment: +{pos} / {neu} / -{neg} (total {total})")
    # Rating
    if "Rating" in df.columns:
        parts.append(f"Avg rating: {df['Rating'].mean():.2f}")
    # Volume by month (top 3)
    if {"Year", "Month"}.issubset(df.columns):
        vc_m = (
            df.dropna(subset=["Year", "Month"]).groupby(["Year", "Month"]).size().reset_index(name="n")
              .sort_values("n", ascending=False).head(3)
        )
        parts.append("Top months: " + ", ".join([f"{int(y)}-{int(m):02d} ({int(n)})" for y, m, n in vc_m.values]))
    return " | ".join(parts)


def retrieve(
    question: str,
    rag: RAGIndex,
    df: pd.DataFrame,
    api_key: Optional[str] = None,
    env_path: Optional[str] = None,
    top_k: int = 20,
    filter_first: bool = True,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Retrieve top-k passages with optional metadata filtering.

    Returns (contexts, filters) where contexts is a list of dicts with keys: text, meta, score
    """
    client = _get_openai_client(api_key=api_key, env_path=env_path)
    filters = parse_question_filters(question, df)

    # Embed question
    q_vec = _embed_texts(client, [question], model=rag.embedding_model)[0:1]

    # Similarity search
    if rag.backend == "faiss":
        scores, idxs = rag.index.search(q_vec, top_k * 5)  # type: ignore[union-attr]
        idxs = idxs[0]
        scores = scores[0]
    else:
        # NumPy fallback: matrix dot query (cosine since both are normalized)
        q = q_vec[0]
        sims = rag.matrix @ q  # type: ignore[operator]
        # Top-k*5 indices
        idxs = np.argsort(-sims)[: top_k * 5]
        scores = sims[idxs]

    contexts: List[Dict[str, object]] = []

    def meta_matches(meta: Dict[str, object]) -> bool:
        ok = True
        if "branch" in filters:
            ok = ok and (str(meta.get("Branch")) == str(filters["branch"]))
        if "country" in filters:
            ok = ok and (str(meta.get("Reviewer_Location")) == str(filters["country"]))
        if "months" in filters:
            ok = ok and (meta.get("Month") in set(filters["months"]))
        if "years" in filters:
            ok = ok and (meta.get("Year") in set(filters["years"]))
        return ok

    for i, sc in zip(idxs, scores):
        if i < 0:
            continue
        meta = rag.id2meta[int(i)]
        if filter_first and not meta_matches(meta):
            continue
        contexts.append({"text": meta.get("text", ""), "meta": meta, "score": float(sc)})
        if len(contexts) >= top_k:
            break

    # If nothing matched after filtering, relax
    if not contexts:
        for i, sc in zip(idxs, scores):
            if i < 0:
                continue
            meta = rag.id2meta[int(i)]
            contexts.append({"text": meta.get("text", ""), "meta": meta, "score": float(sc)})
            if len(contexts) >= top_k:
                break

    return contexts, filters


def answer_question(
    question: str,
    rag: RAGIndex,
    df: pd.DataFrame,
    api_key: Optional[str] = None,
    env_path: Optional[str] = None,
    model: str = DEFAULT_CHAT_MODEL,
    top_k: int = 12,
) -> Dict[str, object]:
    """End-to-end QA: parse filters, retrieve contexts, compute summary stats, and generate an answer.

    Returns dict with: answer, filters, stats_summary, contexts (subset shown), model
    """
    client = _get_openai_client(api_key=api_key, env_path=env_path)

    contexts, filters = retrieve(
        question=question, rag=rag, df=df, api_key=api_key, env_path=env_path, top_k=top_k
    )

    # Apply filters to DF for stats
    df_f = _apply_filters(df, filters)
    stats_summary = _summarize_filtered(df_f)

    # Build compact context strings (truncate each to prevent token explosion)
    def fmt_ctx(c):
        meta = c["meta"]
        txt = str(c["text"])[:500]
        rid = meta.get("Review_ID")
        b = meta.get("Branch")
        loc = meta.get("Reviewer_Location")
        ym = f"{meta.get('Year')}-{int(meta.get('Month')):02d}" if meta.get("Year") and meta.get("Month") else ""
        return f"[{rid} | {b} | {loc} | {ym}] {txt}"

    condensed_context = "\n".join(fmt_ctx(c) for c in contexts[:top_k])

    sys_prompt = (
        "You are a senior Customer Service and Guest Experience analyst for Disney parks. "
        "Base your output ONLY on the provided CONTEXT and STATS. "
        "Produce short, actionable recommendations for Customer Service, Operations, and Marketing. "
        "Respect any filters. If evidence is weak or mixed, note it briefly. "
        "Be concise, objective, and business-focused. "
        "Format your entire output in valid Markdown with clear section headings."
    )

    user_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"FILTERS: {filters}\n\n"
        f"STATS: {stats_summary}\n\n"
        f"CONTEXT (top {top_k} passages):\n{condensed_context}\n\n"
        "Write the answer in Markdown with the following sections:\n\n"
        "## Filters\n"
        "Summarize the active filters in one line.\n\n"
        "## Stats Summary\n"
        "Paste the provided STATS line as-is.\n\n"
        "## Insights\n"
        "- Provide 3-6 concise, data-driven insights from CONTEXT/STATS.\n"
        "- Reference timeframes/parks/IDs where possible (e.g., '(2016-05, Disneyland California)').\n"
        "- Prioritize, when supported by evidence, insights such as: 'Long queues are a recurring issue in Disneyland California during summer months' and 'Visitors from Australia often complain about travel times to Hong Kong Disneyland'. If evidence is insufficient, mention them instead under Caveats.\n\n"
        "## Recommendations\n"
        "- Provide 4-6 actionable, one-line recommendations formatted as '- [Category] Recommendation'.\n"
        "- Use categories like [Service], [Operations], [Marketing].\n"
        "- Ground each recommendation in the insights; cite IDs/timeframes/parks when possible.\n"
        "- When supported by evidence, you may include: 'Increase staff availability during peak months to reduce queue times' and 'Offer discounted travel packages for visitors from Australia to Hong Kong Disneyland'.\n\n"
        "## Caveats\n"
        "- If evidence is weak or mixed, explain briefly why (e.g., conflicting reports across months/branches or spring break spikes vs. lighter weekdays).\n"
        "- Suggest what to validate next (e.g., exact spring break weeks, weekday vs. weekend, specific branches).\n"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    answer = resp.choices[0].message.content.strip()
    return {
        "answer": answer,
        "filters": filters,
        "stats_summary": stats_summary,
        "contexts": contexts[:top_k],
        "model": model,
    }


# Persistence helpers

def save_rag_index(rag: RAGIndex, dir_path: str) -> None:
    """Persist a RAGIndex to a directory. Saves:
    - metadata.json: backend, dim, embedding_model
    - id2meta.json: aligned metadata for each vector
    - faiss.index if backend is 'faiss', else embeddings.npy for NumPy backend
    """
    import json
    import os
    os.makedirs(dir_path, exist_ok=True)

    meta = {"backend": rag.backend, "dim": rag.dim, "embedding_model": rag.embedding_model}
    with open(os.path.join(dir_path, "metadata.json"), "w") as f:
        json.dump(meta, f)

    with open(os.path.join(dir_path, "id2meta.json"), "w") as f:
        json.dump(rag.id2meta, f)

    if rag.backend == "faiss" and rag.index is not None and _FAISS_AVAILABLE:
        faiss.write_index(rag.index, os.path.join(dir_path, "faiss.index"))  # type: ignore[union-attr]
    else:
        if rag.matrix is None:
            raise ValueError("No matrix available to save for NumPy backend.")
        np.save(os.path.join(dir_path, "embeddings.npy"), rag.matrix)


def load_rag_index(dir_path: str) -> RAGIndex:
    """Load a RAGIndex previously saved by save_rag_index().
    If faiss.index exists and FAISS is available, loads FAISS backend; otherwise loads NumPy embeddings.
    """
    import json
    import os

    with open(os.path.join(dir_path, "metadata.json")) as f:
        meta = json.load(f)
    with open(os.path.join(dir_path, "id2meta.json")) as f:
        id2meta = json.load(f)

    dim = int(meta.get("dim"))
    embedding_model = str(meta.get("embedding_model", EMBEDDING_MODEL))

    faiss_path = os.path.join(dir_path, "faiss.index")
    emb_path = os.path.join(dir_path, "embeddings.npy")

    if os.path.exists(faiss_path) and _FAISS_AVAILABLE:
        try:
            idx = faiss.read_index(faiss_path)  # type: ignore[attr-defined]
            return RAGIndex(index=idx, matrix=None, id2meta=id2meta, dim=dim, backend="faiss", embedding_model=embedding_model)
        except Exception:
            pass  # fall back to NumPy if FAISS read fails

    if os.path.exists(emb_path):
        mat = np.load(emb_path).astype(np.float32)
        # Ensure rows are normalized for cosine/IP
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        mat = mat / norms
        return RAGIndex(index=None, matrix=mat, id2meta=id2meta, dim=mat.shape[1], backend="numpy", embedding_model=embedding_model)

    raise FileNotFoundError("No FAISS index or embeddings.npy found in directory: " + dir_path)


# Convenience notebook helper

def build_and_answer(
    df: pd.DataFrame,
    question: str,
    api_key: Optional[str] = None,
    env_path: Optional[str] = None,
    embedding_model: str = EMBEDDING_MODEL,
    model: str = DEFAULT_CHAT_MODEL,
    top_k: int = 12,
    prefer_backend: Literal["auto", "faiss", "numpy"] = "auto",
    batch_size: int = 100,
    show_progress: bool = True,
) -> Dict[str, object]:
    """Build a fresh index on the fly and answer a question. Useful for demos.
    For repeated queries, prefer building the index once via build_faiss_index and reusing it.
    """
    rag = build_faiss_index(
        df,
        api_key=api_key,
        env_path=env_path,
        embedding_model=embedding_model,
        prefer_backend=prefer_backend,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    return answer_question(question, rag, df, api_key=api_key, env_path=env_path, model=model, top_k=top_k)
