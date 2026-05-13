"""
Load a scraped research-developments Excel file, persist it under data/, and
print a per-column summary so we can see what each column represents.

Usage:
    python -m src.ingestion.load_excel <path-to-excel> [--sheet SHEET] [--header ROW]

Examples:
    python -m src.ingestion.load_excel data/raw/developments.xlsx
    python -m src.ingestion.load_excel data/raw/developments.xlsx --sheet "Sheet1"
    python -m src.ingestion.load_excel data/raw/developments.xlsx --header 2
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import RAW_DIR, PROCESSED_DIR  # noqa: E402


def load_excel(src: Path, sheet_name=0, header: int = 0) -> pd.DataFrame:
    if not src.exists():
        raise FileNotFoundError(f"Excel file not found: {src}")
    df = pd.read_excel(src, sheet_name=sheet_name, header=header)
    df = df.dropna(axis=1, how="all")
    return df


def _sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce mixed-type object columns to strings so pyarrow can serialize them."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype("string")
    return out


def store(src: Path, df: pd.DataFrame) -> tuple[Path, Path]:
    """Copy the original to data/raw/ and write a parquet snapshot to data/processed/."""
    raw_copy = RAW_DIR / src.name
    if raw_copy.resolve() != src.resolve():
        shutil.copy2(src, raw_copy)

    parquet_path = PROCESSED_DIR / (src.stem + ".parquet")
    _sanitize_for_parquet(df).to_parquet(parquet_path, index=False)
    return raw_copy, parquet_path


def describe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column summary: dtype, null counts, uniques, sample values, inferred role."""
    rows = []
    for col in df.columns:
        s = df[col]
        sample = s.dropna().astype(str).unique()[:3].tolist()
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "nulls": int(s.isna().sum()),
            "unique": int(s.nunique(dropna=True)),
            "inferred_role": _infer_role(col, s),
            "sample_values": sample,
        })
    return pd.DataFrame(rows)


def _infer_role(name: str, s: pd.Series) -> str:
    """Heuristic guess at what a column represents — useful when there's no data dictionary."""
    n = name.lower()
    if any(k in n for k in ("title", "name", "headline")):
        return "title / short label"
    if any(k in n for k in ("abstract", "summary", "description", "content", "body", "text")):
        return "long-form text (chunk for RAG)"
    if any(k in n for k in ("url", "link", "source")):
        return "source URL"
    if any(k in n for k in ("date", "published", "created", "updated")):
        return "timestamp"
    if any(k in n for k in ("author", "researcher", "person")):
        return "person entity (KG node)"
    if any(k in n for k in ("org", "institution", "affiliation", "company")):
        return "organization entity (KG node)"
    if any(k in n for k in ("topic", "category", "tag", "domain", "field")):
        return "category / topic (KG edge)"
    if any(k in n for k in ("citation", "ref", "doi")):
        return "citation / reference"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric metric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "timestamp"
    return "unclassified"


def main():
    parser = argparse.ArgumentParser(description="Load a scraped Excel dataset.")
    parser.add_argument("path", help="Path to the Excel file")
    parser.add_argument("--sheet", default=0, help="Sheet name or index (default: first sheet)")
    parser.add_argument("--header", type=int, default=0, help="0-indexed header row (default: 0)")
    args = parser.parse_args()

    src = Path(args.path).expanduser().resolve()
    df = load_excel(src, sheet_name=args.sheet, header=args.header)
    raw_copy, parquet_path = store(src, df)

    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    print(f"Loaded : {src}")
    print(f"Shape  : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Stored : {raw_copy}")
    print(f"         {parquet_path}")
    if unnamed:
        print(f"Note   : {len(unnamed)} 'Unnamed:' column(s) detected — header row may be wrong.")
        print(f"         Try: --header 1  (or 2) to skip banner rows.")
    print()
    print("Column summary")
    print("-" * 100)
    summary = describe_columns(df)
    with pd.option_context("display.max_colwidth", 50, "display.width", 140):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
