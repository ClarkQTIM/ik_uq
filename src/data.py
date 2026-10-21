"""
Corneal Ulcer Dataset — EDA
============================
Loads the corneal ulcer study CSV and gives a quick overview of
its structure, columns, types, missingness, and value distributions.

Run:
    python eda.py
"""

import pandas as pd
import numpy as np

DATA_PATH = "/persist/QTIM/Active/26-0096/Corneal_Ulcer_Image_Sets/Data/corneal_ulcer_study_data_7_24_2025_lambda.csv"


def run_eda(path):

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------
    print(f"Loading: {path}\n")
    df = pd.read_csv(path)

    # -------------------------------------------------------------------------
    # Shape
    # -------------------------------------------------------------------------
    print(f"{'='*60}")
    print(f"SHAPE")
    print(f"{'='*60}")
    print(f"  Rows:    {df.shape[0]:,}")
    print(f"  Columns: {df.shape[1]}")

    # -------------------------------------------------------------------------
    # Columns + dtypes
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"COLUMNS & DTYPES")
    print(f"{'='*60}")
    for col in df.columns:
        print(f"  {col:<45} {str(df[col].dtype):<12}")

    # -------------------------------------------------------------------------
    # Missingness
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"MISSING VALUES")
    print(f"{'='*60}")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    missing_df = pd.DataFrame({"missing": missing, "pct": missing_pct})
    missing_df = missing_df[missing_df["missing"] > 0].sort_values("missing", ascending=False)
    if missing_df.empty:
        print("  No missing values.")
    else:
        for col, row in missing_df.iterrows():
            print(f"  {col:<45} {int(row['missing']):>5} ({row['pct']}%)")

    # -------------------------------------------------------------------------
    # Numeric columns — summary stats
    # -------------------------------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        print(f"\n{'='*60}")
        print(f"NUMERIC COLUMNS — SUMMARY STATS")
        print(f"{'='*60}")
        print(df[numeric_cols].describe().round(4).to_string())

    # -------------------------------------------------------------------------
    # Categorical / object columns — value counts
    # -------------------------------------------------------------------------
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if cat_cols:
        print(f"\n{'='*60}")
        print(f"CATEGORICAL COLUMNS — VALUE COUNTS")
        print(f"{'='*60}")
        for col in cat_cols:
            vc = df[col].value_counts(dropna=False)
            print(f"\n  {col}  (n unique: {df[col].nunique()})")
            for val, count in vc.head(20).items():
                pct = count / len(df) * 100
                print(f"    {str(val):<30} {count:>5}  ({pct:.1f}%)")
            if len(vc) > 20:
                print(f"    ... and {len(vc) - 20} more unique values")

    # -------------------------------------------------------------------------
    # Low-cardinality numeric columns — treat as categorical
    # These are likely binary flags or label columns
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"LOW-CARDINALITY NUMERIC COLUMNS (≤10 unique values)")
    print(f"{'='*60}")
    found = False
    for col in numeric_cols:
        n_unique = df[col].nunique()
        if n_unique <= 10:
            found = True
            vc = df[col].value_counts(dropna=False).sort_index()
            vals = ", ".join([f"{v}:{c}" for v, c in vc.items()])
            print(f"  {col:<45} unique={n_unique}  [{vals}]")
    if not found:
        print("  None found.")

    # -------------------------------------------------------------------------
    # First few rows
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"FIRST 5 ROWS")
    print(f"{'='*60}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df.head().to_string())

    print(f"\n{'='*60}")
    print("EDA complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_eda(DATA_PATH)