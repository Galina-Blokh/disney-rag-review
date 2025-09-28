"""
Data loader and preprocessing utilities for Disney reviews dataset.

Expected columns in the CSV inside disney-reviews.zip:
- Review_ID
- Rating
- Year_Month (format examples: YYYY-MM, YYYY-M)
- Reviewer_Location
- Review_Text
- Branch (park location)

Main entry point:
- load_and_preprocess(zip_path="disney-reviews.zip", csv_name=None) -> (DataFrame, mappings, summary)

This will:
- Load the CSV located in the ZIP (auto-detects the only CSV if csv_name is not provided)
- Handle missing values (impute rating median; fill unknowns for location/branch; empty text for missing Review_Text)
- Clean review text (lowercase, remove special characters) and tokenize
- Parse Year_Month to a standard datetime column (Review_Date), and add Year, Month columns
- Encode categoricals (Branch, Reviewer_Location) into integer codes and return mappings
- Provide a summary dict (n_reviews, n_unique_parks, n_unique_countries)
"""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "Review_ID",
    "Rating",
    "Year_Month",
    "Reviewer_Location",
    "Review_Text",
    "Branch",
]


@dataclass
class EncodingMappings:
    branch_mapping: Dict[str, int]
    reviewer_location_mapping: Dict[str, int]


def list_zip_csvs(zip_path: str) -> List[str]:
    """List CSV files contained in a ZIP archive."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        return [n for n in zf.namelist() if n.lower().endswith(".csv")]


def _read_csv_from_zip(zip_path: str, csv_name: Optional[str] = None) -> pd.DataFrame:
    """Read a CSV file from within a ZIP archive into a DataFrame.

    If csv_name is None, auto-detect the first/only CSV file in the archive.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        candidates = list_zip_csvs(zip_path)
        if not candidates:
            raise FileNotFoundError("No CSV files found inside the ZIP archive.")

        target = csv_name or candidates[0]
        if target not in zf.namelist():
            raise FileNotFoundError(f"CSV '{target}' not found in ZIP. Available: {candidates}")

        with zf.open(target) as f:
            raw = f.read()
            # Try a set of common encodings, falling back gracefully
            for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
                try:
                    text_buf = io.StringIO(raw.decode(enc, errors="strict"))
                    df = pd.read_csv(text_buf)
                    return df
                except UnicodeDecodeError:
                    continue
                except Exception:
                    # If it's not a decode problem, try next encoding anyway
                    continue
            # Final fallback: decode with replacement to avoid hard failure
            text_buf = io.StringIO(raw.decode("utf-8", errors="replace"))
            df = pd.read_csv(text_buf)
            return df


def _ensure_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}. Found: {list(df.columns)}")


def _to_numeric_rating(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    # Impute missing ratings with median
    median_rating = df["Rating"].median()
    if np.isnan(median_rating):
        # Fallback to 0 if all ratings are NaN, though this is unlikely
        median_rating = 0.0
    df["Rating"] = df["Rating"].fillna(median_rating)
    return df


def clean_text(text: str) -> str:
    """Lowercase, remove special characters (keep alphanumerics and spaces), collapse whitespace."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    # Remove anything not a-z, 0-9, or whitespace
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    # Collapse multiple spaces
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer on already-cleaned text."""
    if not isinstance(text, str) or not text:
        return []
    return text.split()


def _parse_year_month(val: object) -> pd.Timestamp | pd.NaTType:
    """Parse values like 'YYYY-MM' or 'YYYY-M' into a Timestamp (set day to 1).

    Returns pd.NaT when parsing fails.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return pd.NaT
    s = str(val).strip()
    # Normalize month to 2 digits if pattern resembles 'YYYY-M'
    m = re.match(r"^(\d{4})-(\d{1,2})$", s)
    if m:
        year, month = m.group(1), int(m.group(2))
        try:
            return pd.Timestamp(year=int(year), month=month, day=1)
        except Exception:
            return pd.NaT
    # Try generic parsing as fallback
    try:
        dt = pd.to_datetime(s, format="%Y-%m", errors="coerce")
        if pd.isna(dt):
            return pd.NaT
        return pd.Timestamp(year=dt.year, month=dt.month, day=1)
    except Exception:
        return pd.NaT


def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Review_ID as string
    df["Review_ID"] = df["Review_ID"].astype(str)

    # Fill missing text with empty string, then clean/tokenize
    df["Review_Text"] = df["Review_Text"].fillna("")

    # For location and branch, fill unknowns to keep rows usable
    df["Reviewer_Location"] = df["Reviewer_Location"].fillna("Unknown")
    df["Branch"] = df["Branch"].fillna("Unknown")

    return df


def _encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, EncodingMappings]:
    df = df.copy()

    # Use pandas categorical codes for stable integer encoding
    df["Branch"] = df["Branch"].astype("string").fillna("Unknown")
    df["Reviewer_Location"] = df["Reviewer_Location"].astype("string").fillna("Unknown")

    branch_cat = pd.Categorical(df["Branch"])
    loc_cat = pd.Categorical(df["Reviewer_Location"])

    df["Branch_Code"] = branch_cat.codes.astype(int)
    df["Reviewer_Location_Code"] = loc_cat.codes.astype(int)

    branch_mapping = {cat: code for code, cat in enumerate(branch_cat.categories)}
    reviewer_location_mapping = {cat: code for code, cat in enumerate(loc_cat.categories)}

    mappings = EncodingMappings(
        branch_mapping=branch_mapping,
        reviewer_location_mapping=reviewer_location_mapping,
    )
    return df, mappings


def preprocess_reviews(df: pd.DataFrame) -> Tuple[pd.DataFrame, EncodingMappings]:
    """Apply all preprocessing steps and return the processed DataFrame and encoding mappings."""
    _ensure_required_columns(df)

    # Handle missing values for key columns first
    df = _handle_missing_values(df)

    # Ratings to numeric with median imputation
    df = _to_numeric_rating(df)

    # Clean text and tokenize
    df["Cleaned_Review_Text"] = df["Review_Text"].apply(clean_text)
    df["Review_Tokens"] = df["Cleaned_Review_Text"].apply(tokenize)

    # Parse Year_Month into a proper date column, add Year and Month
    df["Review_Date"] = df["Year_Month"].apply(_parse_year_month)
    df["Year"] = df["Review_Date"].dt.year
    df["Month"] = df["Review_Date"].dt.month

    # Encode categoricals
    df, mappings = _encode_categoricals(df)

    return df, mappings


def dataset_summary(df: pd.DataFrame) -> Dict[str, int]:
    """Return basic summary: number of reviews, unique parks, unique countries."""
    return {
        "n_reviews": int(len(df)),
        "n_unique_parks": int(df["Branch"].nunique(dropna=False)),
        "n_unique_countries": int(df["Reviewer_Location"].nunique(dropna=False)),
    }


def save_processed_dataset(
    df: pd.DataFrame,
    dir_path: str = "data/processed",
    base_name: str = "disney_reviews_processed",
    save_csv: bool = True,
    save_parquet: bool = True,
) -> Dict[str, str]:
    """Persist the processed dataset to disk.

    - dir_path: directory to save into (created if missing)
    - base_name: file stem without extension
    - save_csv: save as CSV
    - save_parquet: attempt to save as Parquet (requires pyarrow or fastparquet)

    Returns a dict with keys 'csv' and/or 'parquet' pointing to saved file paths.
    """
    out: Dict[str, str] = {}
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)

    if save_csv:
        csv_path = p / f"{base_name}.csv"
        df.to_csv(csv_path, index=False)
        out["csv"] = str(csv_path)

    if save_parquet:
        try:
            pq_path = p / f"{base_name}.parquet"
            df.to_parquet(pq_path, index=False)
            out["parquet"] = str(pq_path)
        except Exception:
            # Parquet engine not installed; skip silently
            pass

    return out


def load_processed_dataset(path: str) -> pd.DataFrame:
    """Load a previously saved dataset (CSV or Parquet)."""
    suffix = Path(path).suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_and_preprocess(
    zip_path: str = "disney-reviews.zip",
    csv_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, EncodingMappings, Dict[str, int]]:
    """Load the dataset from a ZIP and run preprocessing.

    Parameters
    - zip_path: Path to the disney reviews ZIP archive
    - csv_name: Optional specific CSV name inside the ZIP (auto-detected otherwise)

    Returns
    - processed DataFrame
    - encoding mappings for categoricals
    - summary dictionary
    """
    df = _read_csv_from_zip(zip_path=zip_path, csv_name=csv_name)

    # Standardize column names if they come in unexpected cases/spaces
    # Try to trim whitespace and keep original casing for expected columns
    df.columns = [str(c).strip() for c in df.columns]

    # Ensure all required columns exist
    _ensure_required_columns(df)

    processed, mappings = preprocess_reviews(df)
    summary = dataset_summary(processed)
    return processed, mappings, summary


if __name__ == "__main__":
    # Example CLI usage for quick verification
    try:
        df_proc, maps, summ = load_and_preprocess()
        print("Summary:", summ)
        print("\nSample rows:")
        print(df_proc.head(3).to_string(index=False))
        print("\nBranch mapping (first 10):", dict(list(maps.branch_mapping.items())[:10]))
        print("Reviewer location mapping (first 10):", dict(list(maps.reviewer_location_mapping.items())[:10]))
    except Exception as e:
        print("Failed to load and preprocess:", e)
