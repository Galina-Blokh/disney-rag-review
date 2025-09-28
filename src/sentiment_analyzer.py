"""
Sentiment analysis utilities for Disney reviews.

Uses VADER (vaderSentiment) to compute polarity scores for review text
and assigns sentiment labels: negative, neutral, positive.

Primary functions:
- add_sentiment(df, text_col='Cleaned_Review_Text', label_col='Sentiment_Label', score_col='Sentiment_Score')
- summarize_by_park(df, label_col='Sentiment_Label', park_col='Branch')
- summarize_by_country(df, label_col='Sentiment_Label', country_col='Reviewer_Location')

The module expects the DataFrame produced by src.data_loader.load_and_preprocess,
which includes a cleaned text column. If `Cleaned_Review_Text` is not found,
it will fallback to cleaning `Review_Text` on the fly.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    # Optional import to reuse the exact same cleaner used in preprocessing
    from src.data_loader import clean_text
except Exception:
    def clean_text(x: str) -> str:
        # Very basic fallback cleaner
        import re
        if not isinstance(x, str):
            return ""
        t = x.lower()
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t


def _vader_label_from_compound(compound: float) -> str:
    """Map VADER compound score to a sentiment label.

    Standard thresholds per VADER docs:
    - compound >= 0.05 -> positive
    - compound <= -0.05 -> negative
    - otherwise -> neutral
    """
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def add_sentiment(
    df: pd.DataFrame,
    text_col: str = "Cleaned_Review_Text",
    label_col: str = "Sentiment_Label",
    score_col: str = "Sentiment_Score",
) -> pd.DataFrame:
    """Compute VADER sentiment and append label and score columns.

    If `text_col` is not present, the function will attempt to use `Review_Text`
    and clean it first.
    """
    if text_col not in df.columns:
        if "Review_Text" in df.columns:
            tmp_text = df["Review_Text"].fillna("").map(clean_text)
        else:
            raise ValueError(f"Text column '{text_col}' not found and no 'Review_Text' available.")
    else:
        tmp_text = df[text_col].fillna("")

    analyzer = SentimentIntensityAnalyzer()

    # Compute compound scores
    scores = tmp_text.map(lambda t: analyzer.polarity_scores(t)["compound"] if isinstance(t, str) else 0.0)

    out = df.copy()
    out[score_col] = scores
    out[label_col] = scores.map(_vader_label_from_compound)
    return out


def _summarize_by(
    df: pd.DataFrame,
    group_col: str,
    label_col: str = "Sentiment_Label",
) -> pd.DataFrame:
    """Return counts and percentages of sentiment labels per group.

    Output columns: negative, neutral, positive, total, negative_pct, neutral_pct, positive_pct
    """
    # Ensure label column exists
    if label_col not in df.columns:
        raise ValueError(f"'{label_col}' not found. Run add_sentiment() first.")

    counts = (
        df.groupby(group_col)[label_col]
          .value_counts()
          .unstack(fill_value=0)
          .reindex(columns=["negative", "neutral", "positive"], fill_value=0)
    )
    counts["total"] = counts.sum(axis=1)

    # Avoid division by zero
    pct = counts[["negative", "neutral", "positive"]].div(counts["total"].replace({0: 1}), axis=0)
    pct = pct.add_suffix("_pct")

    summary = pd.concat([counts, pct], axis=1).reset_index()
    return summary


def summarize_by_park(
    df: pd.DataFrame,
    label_col: str = "Sentiment_Label",
    park_col: str = "Branch",
) -> pd.DataFrame:
    """Summarize sentiment distribution per park (Branch)."""
    return _summarize_by(df, group_col=park_col, label_col=label_col)


def summarize_by_country(
    df: pd.DataFrame,
    label_col: str = "Sentiment_Label",
    country_col: str = "Reviewer_Location",
) -> pd.DataFrame:
    """Summarize sentiment distribution per reviewer country/location."""
    return _summarize_by(df, group_col=country_col, label_col=label_col)
