from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

JSON_COLS = [
    "genres",
    "keywords",
    "production_companies",
    "production_countries",
    "spoken_languages",
]


def parse_json_names(value) -> list[str]:
    """
    TMDB columns store JSON arrays as strings, like:
    '[{"id": 28, "name": "Action"}, ...]'
    Returns a list of 'name' values.
    """
    if pd.isna(value) or value == "":
        return []
    try:
        items = json.loads(value)
        if isinstance(items, list):
            return [d.get("name") for d in items if isinstance(d, dict) and "name" in d]
    except Exception:
        pass
    return []


def clean_tmdb(df: pd.DataFrame, min_budget_for_roi: int = 10_000) -> pd.DataFrame:
    df = df.copy()

    # --- Basic standardization ---
    df.columns = [c.strip() for c in df.columns]

    # Drop low-value mostly-null columns
    df = df.drop(columns=["homepage"], errors="ignore")

    # release_date -> datetime + year
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year

    # runtime -> numeric (it already is mostly, but make robust)
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")

    # Treat 0 budget/revenue as missing
    for col in ["budget", "revenue"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace(0, np.nan)

    # --- Parse JSON columns into list-of-names columns ---
    for col in JSON_COLS:
        out_col = f"{col}_names"
        df[out_col] = df[col].apply(parse_json_names)

    # Helpful counts
    df["n_genres"] = df["genres_names"].apply(len)
    df["n_keywords"] = df["keywords_names"].apply(len)
    df["n_production_companies"] = df["production_companies_names"].apply(len)

    # Primary genre (useful for grouping / comparisons)
    df["primary_genre"] = df["genres_names"].apply(lambda x: x[0] if x else np.nan)

    # --- Financial features ---
    df["profit"] = df["revenue"] - df["budget"]

    # Flag suspicious budgets (tiny values make ROI misleading)
    df["budget_suspect"] = df["budget"].notna() & (df["budget"] < min_budget_for_roi)

    # ROI: only when budget/revenue are present and budget is not suspicious
    df["roi"] = np.where(
        df["budget"].notna()
        & df["revenue"].notna()
        & (df["budget"] >= min_budget_for_roi),
        df["revenue"] / df["budget"],
        np.nan,
    )

    # --- Optional: tidy text columns ---
    for col in ["overview", "tagline", "homepage"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    # Index by id (keep as a column too for safer CSV interoperability)
    if "id" in df.columns:
        df = df.set_index("id", drop=False)

    def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
        cols = list(df.columns)

        # JSON-derived name columns should be LAST (plus counts if you want them near the end)
        json_name_cols = [c for c in cols if c.endswith("_names")]
        json_raw_cols = [c for c in cols if c in JSON_COLS]  # original json string cols

        # Date metrics together
        date_cols = [c for c in ["release_date", "release_year"] if c in cols]

        # Money/numeric metrics toward the end (adjust as needed)
        money_numeric_tail = [
            c for c in [
                "budget", "revenue", "profit", "roi",
                "runtime", "popularity", "vote_average", "vote_count",
                "budget_suspect",
                "n_genres", "n_keywords", "n_production_companies",
            ]
            if c in cols
        ]

        # Put id first, then a few high-signal identifiers/text columns (optional)
        head_cols = [c for c in ["id", "title", "original_title"] if c in cols]

        # Everything else (core descriptive columns) goes in the middle
        reserved = set(head_cols + date_cols + money_numeric_tail + json_raw_cols + json_name_cols)
        middle_cols = [c for c in cols if c not in reserved]

        # Build final ordering:
        # - head (id/title)
        # - dates
        # - middle (descriptive)
        # - raw json string cols (optional; keeps them out of the way but not last)
        # - numeric/money tail
        # - json name cols last (so they don't fill the screen)
        ordered = (
                head_cols
                + date_cols
                + middle_cols
                + json_raw_cols
                + money_numeric_tail
                + json_name_cols
        )

        return df[ordered]

    # Reorder columns for readability
    df = reorder_columns(df)

    return df


def quality_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> str:
    lines = []
    lines.append(f"Rows: {df_raw.shape[0]} -> {df_clean.shape[0]}")
    lines.append(f"Columns: {df_raw.shape[1]} -> {df_clean.shape[1]}")
    missing = df_clean.isna().sum().sort_values(ascending=False).head(8)
    lines.append("\nTop missing columns (cleaned):")
    for k, v in missing.items():
        lines.append(f"  - {k}: {v}")
    lines.append("\nFinancial completeness (cleaned):")
    lines.append(f"  - budget known:  {df_clean['budget'].notna().mean():.1%}")
    lines.append(f"  - revenue known: {df_clean['revenue'].notna().mean():.1%}")
    lines.append(f"  - roi computed:  {df_clean['roi'].notna().mean():.1%}")
    lines.append(f"  - suspect budgets flagged: {(df_clean['budget_suspect']).mean():.1%}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="Path to raw TMDB CSV")
    parser.add_argument("--out", dest="out_path", required=True, help="Path to save cleaned CSV")
    parser.add_argument("--min_budget_for_roi", type=int, default=10_000)
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(in_path)
    df_clean = clean_tmdb(df_raw, min_budget_for_roi=args.min_budget_for_roi)
    df_clean.to_csv(out_path, index=False)

    print(quality_report(df_raw, df_clean))
    print(f"\nSaved cleaned data -> {out_path}")


if __name__ == "__main__":
    main()