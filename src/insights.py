"""
Insights generation utilities for Disney reviews metadata.

Functions provided:
- busiest_months_by_park(df): Identify the busiest months (by review counts) for each park.
- sentiment_summary_by_park(df): Full positive/neutral/negative counts and percentages by park.
- sentiment_extremes_by_park(df): Determine which parks receive the most positive and most negative reviews.
- temporal_trends(df): Analyze trends over time (ratings and positive rate) for each park, classifying as improving/declining/stable.
- build_structured_summary(df): Aggregate all insights into a structured dict.
- generate_llm_recommendations(summary, model='gpt-4o-mini'): Use OpenAI to produce actionable recommendations.

Expectations:
- DataFrame is the output of load_and_preprocess + add_sentiment, so it includes:
  - Columns: Branch, Reviewer_Location, Review_Date (month-level Timestamp), Year, Month
  - Sentiment_Label in {'negative','neutral','positive'}
  - Rating as numeric
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _ensure_columns(df: pd.DataFrame, cols: List[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def busiest_months_by_park(df: pd.DataFrame) -> pd.DataFrame:
    """Return the busiest months for each park based on review counts.

    Output columns: Branch, Year, Month, Review_Count, Month_Name, Rank_in_Park
    Includes all months ranked per park (Rank 1 is busiest). Caller may filter Rank_in_Park == 1 to get only top months.
    """
    _ensure_columns(df, ["Branch", "Review_Date"])  # Review_Date is month start

    # In case Review_Date has NaT, drop them for counting
    tmp = df.dropna(subset=["Review_Date"]).copy()
    tmp["Year"] = tmp["Review_Date"].dt.year
    tmp["Month"] = tmp["Review_Date"].dt.month

    grp = (
        tmp.groupby(["Branch", "Year", "Month"]).size().reset_index(name="Review_Count")
    )
    grp["Month_Name"] = pd.to_datetime(
        grp[["Year", "Month"]].assign(DAY=1).rename(columns={"Year": "year", "Month": "month", "DAY": "day"})
    ).dt.strftime("%b")

    grp["Rank_in_Park"] = grp.sort_values(["Branch", "Review_Count"], ascending=[True, False]) \
                           .groupby("Branch")["Review_Count"].rank(method="dense", ascending=False).astype(int)
    return grp.sort_values(["Branch", "Rank_in_Park", "Year", "Month"]).reset_index(drop=True)


def sentiment_summary_by_park(df: pd.DataFrame) -> pd.DataFrame:
    """Return counts and percentages of sentiment labels per park (Branch)."""
    _ensure_columns(df, ["Branch", "Sentiment_Label"])

    counts = (
        df.groupby("Branch")["Sentiment_Label"].value_counts().unstack(fill_value=0)
        .reindex(columns=["negative", "neutral", "positive"], fill_value=0)
    )
    counts["total"] = counts.sum(axis=1)
    pct = (counts[["negative", "neutral", "positive"]]
           .div(counts["total"].replace({0: 1}), axis=0))
    pct = pct.add_suffix("_pct")
    out = pd.concat([counts, pct], axis=1).reset_index()
    return out


def sentiment_extremes_by_park(df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    """Return parks with the highest positive and highest negative review counts.

    Returns dict with keys 'most_positive' and 'most_negative', each containing:
      - park: str
      - positive_count / negative_count: int
      - total: int
      - positive_pct / negative_pct: float
    """
    summary = sentiment_summary_by_park(df)
    # Identify extremes
    pos_idx = summary["positive"].idxmax()
    neg_idx = summary["negative"].idxmax()
    most_positive = summary.loc[pos_idx]
    most_negative = summary.loc[neg_idx]

    return {
        "most_positive": {
            "park": most_positive["Branch"],
            "positive_count": int(most_positive["positive"]),
            "total": int(most_positive["total"]),
            "positive_pct": float(most_positive.get("positive_pct", 0.0)),
        },
        "most_negative": {
            "park": most_negative["Branch"],
            "negative_count": int(most_negative["negative"]),
            "total": int(most_negative["total"]),
            "negative_pct": float(most_negative.get("negative_pct", 0.0)),
        },
        "by_park": summary,
    }


def temporal_trends(df: pd.DataFrame, min_points: int = 6, slope_threshold: float = 1e-3) -> pd.DataFrame:
    """Analyze trends in reviews over time per park.

    For each Branch, compute monthly:
      - mean_rating
      - positive_rate
    Fit a simple linear regression slope vs time index. Classify direction by slope_threshold.

    Output columns:
    Branch, rating_slope, rating_direction, positive_rate_slope, positive_rate_direction, n_months
    """
    _ensure_columns(df, ["Branch", "Review_Date", "Rating", "Sentiment_Label"])

    tmp = df.dropna(subset=["Review_Date"]).copy()
    tmp["Year"] = tmp["Review_Date"].dt.year
    tmp["Month"] = tmp["Review_Date"].dt.month
    tmp["is_positive"] = (tmp["Sentiment_Label"] == "positive").astype(int)

    monthly = (tmp
        .groupby(["Branch", "Year", "Month"])  
        .agg(mean_rating=("Rating", "mean"),
             positive_rate=("is_positive", "mean"),
             n_reviews=("Rating", "size"))
        .reset_index()
    )

    rows: List[Dict[str, object]] = []

    for park, g in monthly.groupby("Branch"):
        g = g.sort_values(["Year", "Month"]).reset_index(drop=True)
        g["t"] = np.arange(len(g))  # time index
        n = len(g)
        if n < min_points:
            rows.append({
                "Branch": park,
                "rating_slope": np.nan,
                "rating_direction": "insufficient_data",
                "positive_rate_slope": np.nan,
                "positive_rate_direction": "insufficient_data",
                "n_months": n,
            })
            continue

        # Simple linear regression slope via polyfit degree 1
        r_slope = float(np.polyfit(g["t"], g["mean_rating"], 1)[0])
        p_slope = float(np.polyfit(g["t"], g["positive_rate"], 1)[0])

        def classify(s: float) -> str:
            if s > slope_threshold:
                return "improving"
            if s < -slope_threshold:
                return "declining"
            return "stable"

        rows.append({
            "Branch": park,
            "rating_slope": r_slope,
            "rating_direction": classify(r_slope),
            "positive_rate_slope": p_slope,
            "positive_rate_direction": classify(p_slope),
            "n_months": n,
        })

    return pd.DataFrame(rows).sort_values(["rating_direction", "positive_rate_direction"], ascending=True)


def build_structured_summary(df: pd.DataFrame) -> Dict[str, object]:
    """Build a structured summary dict from the DataFrame (expects sentiment columns present)."""
    busiest = busiest_months_by_park(df)
    extremes = sentiment_extremes_by_park(df)
    trends = temporal_trends(df)

    return {
        "busiest_months_by_park": busiest,
        "sentiment_extremes": extremes,  # includes 'by_park' table
        "temporal_trends": trends,
    }


# Optional: LLM recommendations via OpenAI
try:
    from dotenv import load_dotenv, find_dotenv
    import os
    from openai import OpenAI

    def generate_llm_recommendations(
        summary: Dict[str, object],
        model: str = "gpt-4o-mini",
        max_tokens: int = 500,
        api_key: str | None = None,
        env_path: str | None = None,
    ) -> str:
        """Generate actionable recommendations from the structured summary using an OpenAI model.

        Reads OPENAI_API_KEY from environment (.env supported). If key is missing, returns a helpful message.
        """
        # Load .env if present (search upwards so it works from notebooks/)
        if api_key is None:
            if env_path:
                # Explicit path provided by caller
                load_dotenv(dotenv_path=env_path, override=False)
            else:
                dotenv_path = find_dotenv(usecwd=True)
                if dotenv_path:
                    load_dotenv(dotenv_path=dotenv_path, override=False)
            # Support both standard and alternate keys
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY")
        if not api_key:
            return (
                "OPENAI_API_KEY not found in environment. Create a .env with OPENAI_API_KEY=... "
                "or export it in your shell to enable LLM-based recommendations."
            )

        # Optionally allow model from env if caller didn't specify differently
        if model is None or model == "auto":
            model = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

        client = OpenAI(api_key=api_key)

        # Prepare a compact textual representation of the summary
        def df_to_text(df: pd.DataFrame, max_rows: int = 10) -> str:
            if isinstance(df, pd.DataFrame):
                return df.head(max_rows).to_string(index=False)
            return str(df)

        busiest_text = df_to_text(summary.get("busiest_months_by_park"))
        extremes = summary.get("sentiment_extremes", {})
        by_park = extremes.get("by_park")
        by_park_text = df_to_text(by_park)
        extremes_text = {
            "most_positive": extremes.get("most_positive"),
            "most_negative": extremes.get("most_negative"),
        }
        trends_text = df_to_text(summary.get("temporal_trends"))

        prompt = (
            "You are a data analyst. Based on the provided summaries of Disney park reviews, "
            "produce concise, actionable recommendations for operations and CX teams. "
            "Focus on: staffing, amenities, communications, and peak-time readiness.\n\n"
            f"Busiest months by park (sample):\n{busiest_text}\n\n"
            f"Sentiment by park (sample):\n{by_park_text}\n\n"
            f"Extremes: {extremes_text}\n\n"
            f"Temporal trends (sample):\n{trends_text}\n\n"
            "Return 5-8 bullet points."
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert CX analyst."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

except Exception:
    # Provide a stub when openai or dotenv isn't available
    def generate_llm_recommendations(summary: Dict[str, object], model: str = "gpt-4o-mini", max_tokens: int = 500) -> str:
        return (
            "OpenAI client not available. Install dependencies and ensure OPENAI_API_KEY is set to enable recommendations."
        )
