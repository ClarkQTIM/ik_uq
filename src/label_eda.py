"""
Hard Label Distribution EDA
============================
Checks bac_LCA_bin and fung_LCA_bin value distributions
by study to confirm which rows have valid hard labels
before training Cells C and D.

Run:
    python label_eda.py
"""

import pandas as pd
import numpy as np

DATA_PATH = "/persist/QTIM/Active/26-0096/Corneal_Ulcer_Image_Sets/Data/corneal_ulcer_study_data_7_24_2025_lambda.csv"

def run(path):
    df = pd.read_csv(path)

    # Apply same filtering as model.py
    df = df.dropna(subset=["file_path", "subject"])
    unlabelable = (df["fung_LCA_bin"] == -1) & (df["bac_LCA_bin"] == -1)
    df = df[~unlabelable].reset_index(drop=True)
    print(f"After standard filtering: {len(df):,} rows\n")

    # ── Per-study breakdown of bac_LCA_bin and fung_LCA_bin ──
    studies = df["study"].unique()

    print(f"{'='*70}")
    print(f"BAC_LCA_BIN by study  (-1=unlabelable, 0=absent, 1=present)")
    print(f"{'='*70}")
    bac_pivot = df.groupby("study")["bac_LCA_bin"].value_counts().unstack(fill_value=0)
    bac_pivot = bac_pivot.reindex(columns=[-1, 0, 1], fill_value=0)
    bac_pivot["total"] = bac_pivot.sum(axis=1)
    bac_pivot["pct_1"] = (bac_pivot[1] / bac_pivot["total"] * 100).round(1)
    bac_pivot["pct_-1"] = (bac_pivot[-1] / bac_pivot["total"] * 100).round(1)
    print(bac_pivot.to_string())

    print(f"\n{'='*70}")
    print(f"FUNG_LCA_BIN by study  (-1=unlabelable, 0=absent, 1=present)")
    print(f"{'='*70}")
    fung_pivot = df.groupby("study")["fung_LCA_bin"].value_counts().unstack(fill_value=0)
    fung_pivot = fung_pivot.reindex(columns=[-1, 0, 1], fill_value=0)
    fung_pivot["total"] = fung_pivot.sum(axis=1)
    fung_pivot["pct_1"] = (fung_pivot[1] / fung_pivot["total"] * 100).round(1)
    fung_pivot["pct_-1"] = (fung_pivot[-1] / fung_pivot["total"] * 100).round(1)
    print(fung_pivot.to_string())

    # ── Rows where EITHER bin is -1 (partial unlabelable) ──
    print(f"\n{'='*70}")
    print(f"ROWS WITH ANY -1 (after initial filter)")
    print(f"{'='*70}")
    has_minus1 = (df["bac_LCA_bin"] == -1) | (df["fung_LCA_bin"] == -1)
    print(f"  Rows with bac_LCA_bin = -1:  {(df['bac_LCA_bin']==-1).sum():,}")
    print(f"  Rows with fung_LCA_bin = -1: {(df['fung_LCA_bin']==-1).sum():,}")
    print(f"  Rows with either = -1:       {has_minus1.sum():,}")
    print(f"\nBreakdown by study:")
    print(df[has_minus1].groupby("study")[["bac_LCA_bin","fung_LCA_bin"]].value_counts().to_string())

    # ── Soft label vs hard label agreement ──
    print(f"\n{'='*70}")
    print(f"SOFT vs HARD LABEL AGREEMENT")
    print(f"{'='*70}")
    # Only on rows with valid hard labels (no -1)
    valid = df[(df["bac_LCA_bin"] != -1) & (df["fung_LCA_bin"] != -1)].copy()
    valid["bac_soft_bin"]  = (valid["bac_post_prob"]  > 0.5).astype(int)
    valid["fung_soft_bin"] = (valid["fung_post_prob"] > 0.5).astype(int)

    bac_agree  = (valid["bac_soft_bin"]  == valid["bac_LCA_bin"]).mean()
    fung_agree = (valid["fung_soft_bin"] == valid["fung_LCA_bin"]).mean()

    print(f"  Rows with valid hard labels: {len(valid):,}")
    print(f"  BAC  soft vs hard agreement: {bac_agree*100:.1f}%")
    print(f"  FUNG soft vs hard agreement: {fung_agree*100:.1f}%")

    # Disagreements by study
    valid["bac_disagree"]  = valid["bac_soft_bin"]  != valid["bac_LCA_bin"]
    valid["fung_disagree"] = valid["fung_soft_bin"] != valid["fung_LCA_bin"]
    print(f"\n  BAC disagreements by study:")
    print(valid.groupby("study")["bac_disagree"].sum().to_string())
    print(f"\n  FUNG disagreements by study:")
    print(valid.groupby("study")["fung_disagree"].sum().to_string())

    # ── What to use for hard label training ──
    print(f"\n{'='*70}")
    print(f"RECOMMENDATION")
    print(f"{'='*70}")
    n_valid = len(valid)
    n_total = len(df)
    n_dropped = n_total - n_valid
    print(f"  If we drop rows where either bin = -1:")
    print(f"    Remaining rows: {n_valid:,} ({n_valid/n_total*100:.1f}% of filtered dataset)")
    print(f"    Dropped rows:   {n_dropped:,} ({n_dropped/n_total*100:.1f}%)")
    print(f"\n  If we keep -1 rows and treat -1 as 0 (absent):")
    print(f"    All {n_total:,} rows usable")
    print(f"    Risk: -1 means unknown, not confirmed absent")

if __name__ == "__main__":
    run(DATA_PATH)