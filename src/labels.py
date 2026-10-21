"""
Shared label handling for the keratitis project.

Every script that needs labels must get them from here, so the rule is written
once (audit finding C1: eleven scripts had their own copy of the loading code,
and only one of them handled -1).

The rule (agreed 2026-10-21; Hady Yazbeck's email of 23 Jun 2026):
  * Rows whose two LCA bins are both -1 have no label and are excluded.
  * bac_post_prob / fung_post_prob == -1 is the "no posterior" code: the image
    has no smear or culture result. It is NOT a probability.
  * Soft target  = the posterior where one exists, otherwise the LCA bin.
  * Hard target  = the LCA bin.
  * Reference    = the LCA bin, for every model. Wherever a posterior exists the
                   bin equals posterior > 0.5, so nothing is lost.
  * The task is binary (bacterial vs fungal): the LCA has two mutually
    exclusive classes, so the bins are always complementary and the posteriors
    always sum to 1. y_bac = 1 means bacterial, 0 means fungal.

Why the bin is the right fallback: the trials enrolled on a positive entry test
(MUTT: smear-positive filamentous fungus; SCUT: culture-positive bacterial), and
trial participants whose results are recorded get posteriors within 0.002 of
the bin (MUTT 0.0016, SCUT 0.9996).

Load-time checks fail loudly instead of passing bad values through.
"""

import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PACS_PREFIX = "/pacs/OSHU/Corneal_Ulcer_Image_Sets"
FOLLOWUP_VISITS = {"3 weeks", "3 months", "12 months", "3E (unclear)"}
STUDY_SHORT = {
    "MADURAI photos": "MADURAI", "MUTT2-Pictures": "MUTT2",
    "MUTT1-Pictures": "MUTT1", "SCUT photos": "SCUT",
    "MUTT Pilot Photos": "MUTT Pilot", "CLAIR II photos": "CLAIR II",
    "CLAIR I photos": "CLAIR I",
}


# -----------------------------------------------------------------------------
# Visit timing, from the image name (MUTT1, MUTT2, SCUT encode it)
# -----------------------------------------------------------------------------

def parse_visit(study, name):
    """Return the visit an image belongs to, parsed from its file name."""
    n = str(name).upper().replace(".JPG", "").replace("JPG", "").strip()
    if study.startswith("CLAIR"):
        return "enrollment (single image)"
    if study == "MADURAI photos":
        return "not encoded"
    if study == "MUTT Pilot Photos":
        return "enrollment" if re.match(r"^P\d+[A-Z]-E\d", n) else "unparsed"
    m = re.match(r"^(\d+)\s*([A-Z])\s*[-_ ]*\s*(.*)$", n)
    if not m:
        return "unparsed"
    rest = m.group(3).replace(" ", "")
    for code, lab in [("12M", "12 months"), ("3M", "3 months"), ("3W", "3 weeks"),
                      ("3E", "3E (unclear)"), ("EN", "enrollment"), ("E", "enrollment")]:
        if rest.startswith(code):
            return lab
    return "unparsed"


# -----------------------------------------------------------------------------
# Loading and checking
# -----------------------------------------------------------------------------

def load_labelled(csv_path, img_root=None, group_col="pID", verbose=True):
    """
    Read the study CSV, apply the label rule and run the load-time checks.

    Returns a DataFrame with the original columns plus:
      csv_row        row number in the CSV as read (for joining back)
      has_posterior  True where a real posterior exists
      soft_bac, soft_fung   soft targets (posterior, else bin)
      hard_bac, hard_fung   hard targets (bin)
      y_bac          reference label, 1 = bacterial, 0 = fungal
      visit, is_followup    visit parsed from the image name
      local_path     image path under img_root (if img_root is given)
    """
    raw = pd.read_csv(csv_path)
    raw["csv_row"] = np.arange(len(raw))
    n_raw = len(raw)

    need = ["file_path", group_col]
    df = raw.dropna(subset=need).copy()
    n_no_photo = n_raw - len(df)

    both_bins_missing = (df["bac_LCA_bin"] == -1) & (df["fung_LCA_bin"] == -1)
    n_unlabelable = int(both_bins_missing.sum())
    df = df[~both_bins_missing].reset_index(drop=True)

    # --- Bins: must be complementary 0/1 on every remaining row -------------
    bb, fb = df["bac_LCA_bin"], df["fung_LCA_bin"]
    bad_bins = ~(bb.isin([0, 1]) & fb.isin([0, 1]) & ((bb + fb) == 1))
    if bad_bins.any():
        raise ValueError(f"{int(bad_bins.sum())} rows have LCA bins that are not a "
                         f"complementary 0/1 pair; the binary label rule does not apply.")

    # --- Posteriors: -1 (missing) or a probability; missing together --------
    bp, fp = df["bac_post_prob"], df["fung_post_prob"]
    valid = lambda s: (s == -1) | s.between(0, 1)
    if not (valid(bp) & valid(fp)).all():
        raise ValueError("Posterior columns contain values that are neither -1 nor in [0, 1].")
    if not ((bp == -1) == (fp == -1)).all():
        raise ValueError("bac_post_prob and fung_post_prob are not missing together.")
    has_post = bp != -1
    off_sum = has_post & ((bp + fp - 1).abs() > 1e-3)
    disagree = has_post & ((bp > 0.5).astype(int) != bb)
    if off_sum.any():
        print(f"WARNING: {int(off_sum.sum())} posterior pairs do not sum to 1 (tolerance 1e-3).")
    if disagree.any():
        print(f"WARNING: {int(disagree.sum())} rows where posterior > 0.5 disagrees with the bin.")

    # --- The rule ------------------------------------------------------------
    df["has_posterior"] = has_post.values
    df["soft_bac"] = np.where(has_post, bp, bb).astype(float)
    df["soft_fung"] = np.where(has_post, fp, fb).astype(float)
    df["hard_bac"] = bb.astype(float)
    df["hard_fung"] = fb.astype(float)
    df["y_bac"] = bb.astype(int)
    for c in ["soft_bac", "soft_fung", "hard_bac", "hard_fung"]:
        if not df[c].between(0, 1).all():
            raise AssertionError(f"{c} has values outside [0, 1] after the label rule.")

    # --- Visit timing ----------------------------------------------------------
    if "image_name" not in df.columns:
        df["image_name"] = df["file_path"].map(os.path.basename)
    df["visit"] = [parse_visit(s, n) for s, n in zip(df["study"], df["image_name"])]
    df["is_followup"] = df["visit"].isin(FOLLOWUP_VISITS)

    if img_root is not None:
        df["local_path"] = df["file_path"].map(lambda p: p.replace(PACS_PREFIX, img_root))

    if verbose:
        print(f"Labels: {n_raw:,} CSV rows; {n_no_photo:,} without a photo or {group_col}; "
              f"{n_unlabelable:,} with both bins -1 (excluded); {len(df):,} kept, "
              f"{df[group_col].nunique():,} participants.")
        print(f"  Posterior present on {int(has_post.sum()):,} rows; "
              f"bin used as soft target on {int((~has_post).sum()):,} rows.")
    return df


def label_summary(df, group_col="pID"):
    """Per-study summary of the labels (no identifiers)."""
    d = df.assign(_differs=(df["soft_bac"] - df["hard_bac"]).abs() > 0.1)
    g = d.groupby("study")
    out = pd.DataFrame({
        "images": g.size(),
        "participants": g[group_col].nunique(),
        "with_posterior": g["has_posterior"].sum(),
        "bacterial_share": g["y_bac"].mean().round(3),
        "soft_differs_from_hard": g["_differs"].sum(),
        "followup_images": g["is_followup"].sum(),
    })
    out.index = [STUDY_SHORT.get(s, s) for s in out.index]
    return out


def targets(df, label_type):
    """Training targets, shape (N, 2): [bacterial, fungal]."""
    if label_type == "soft":
        return df[["soft_bac", "soft_fung"]].to_numpy(dtype=np.float32)
    if label_type == "hard":
        return df[["hard_bac", "hard_fung"]].to_numpy(dtype=np.float32)
    raise ValueError(label_type)


# -----------------------------------------------------------------------------
# Splits, grouped by participant
# -----------------------------------------------------------------------------

def split_by_participant(df, seed=42, group_col="pID"):
    """
    70/15/15 train/val/test split grouped by participant, then the test set
    split 20/80 into calibration/evaluation, also grouped by participant.
    Same procedure as before, but grouped by pID (subject maps 9 pIDs to more
    than one subject, which let 5 participants straddle splits).

    Returns a dict of positional index arrays into df.
    """
    groups = df[group_col].to_numpy()
    idx = np.arange(len(df))
    tr, tmp = next(GroupShuffleSplit(1, test_size=0.30, random_state=seed).split(idx, groups=groups))
    va_rel, te_rel = next(GroupShuffleSplit(1, test_size=0.50, random_state=seed).split(tmp, groups=groups[tmp]))
    va, te = tmp[va_rel], tmp[te_rel]
    ca_rel, ev_rel = next(GroupShuffleSplit(1, test_size=0.80, random_state=seed).split(te, groups=groups[te]))
    ca, ev = te[ca_rel], te[ev_rel]
    parts = {"train": tr, "val": va, "test": te, "cal": ca, "eval": ev}
    # no participant in more than one of train / val / test, or of cal / eval
    for a, b in [("train", "val"), ("train", "test"), ("val", "test"), ("cal", "eval")]:
        shared = set(groups[parts[a]]) & set(groups[parts[b]])
        if shared:
            raise AssertionError(f"{len(shared)} participants are in both {a} and {b}.")
    return parts


def participant_codes(df, group_col="pID"):
    """Integer participant codes for clustered statistics (no identifiers)."""
    return pd.factorize(df[group_col])[0].astype(np.int64)
