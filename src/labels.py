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
# Smear and culture columns: 1 positive, 0 negative, -1 missing.
TESTS = ["bac_smear", "fung_smear", "bac_cult_positive", "fung_cult_positive"]
# Label types read from the redo table (src/paper_labels.py -> paper_labels.csv)
# culture_hard: the cultures themselves as hard labels (blood agar -> bacterial, PDA -> fungal), for the culture
# analysis (images with at least one culture; --subset culture), trained with the same loss as the soft labels.
PAPER_LABEL_TYPES = ["paper_soft", "paper_hard", "culture_hard", "twoclass_soft", "twoclass_hard"]
CULTURES = ["bac_cult_positive", "fung_cult_positive"]
TEST_TARGETS = {"smear": ["bac_smear", "fung_smear"],                    # Gram, KOH
                "culture": ["bac_cult_positive", "fung_cult_positive"]}  # blood agar, PDA
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


def test_results(df, group_col="pID"):
    """
    Smear and culture results per image, carried across each participant's images
    (results never conflict within a participant; some of a participant's images
    lack them). Shape (N, 4) in TESTS order: 1 positive, 0 negative, NaN missing.
    """
    t = df[[group_col] + TESTS].copy()
    for c in TESTS:
        t[c] = t[c].where(t[c] >= 0)
    per_participant = t.groupby(group_col)[TESTS].max()
    return per_participant.reindex(df[group_col]).to_numpy(dtype=np.float32)


def has_culture(df):
    """Images with at least one culture recorded (blood agar or PDA): the culture analysis."""
    return df[CULTURES].apply(pd.to_numeric, errors="coerce").isin([0, 1]).any(axis=1).to_numpy()


def load_paper_table(path, verbose=True, require_image=True):
    """
    The redo label table written by src/paper_labels.py: one row per paper image with its local file,
    the paper's split, the four tests, the paper's per-pathogen posteriors, Travis's hard labels and the
    two-class posterior. Keeps images with a file and labels. Adds the columns the training code expects:
      csv_row        row number in the table (for joining back)
      y_bac          binary reference = two-class posterior > 0.5
      soft_bac/fung  two-class posterior and its complement (the 'label posterior' used in the analysis)
      has_posterior  True
      is_followup    from the visit column
    """
    t = pd.read_csv(path)
    t["csv_row"] = np.arange(len(t))
    need = ["split", "subject", "study", "local_path", "has_image", "paper_soft_bac", "paper_soft_fung",
            "hard_bac", "hard_fung", "twoclass_bac", "y_bac"] + TESTS
    miss = [c for c in need if c not in t.columns]
    if miss:
        raise ValueError(f"{path}: missing columns {miss}")
    n0 = len(t)
    if require_image:
        t = t[t["has_image"].astype(bool)]
    t = t[t[["paper_soft_bac", "paper_soft_fung", "twoclass_bac"]].notna().all(axis=1)].copy()
    t["subject"] = t["subject"].astype(str)
    t["local_path"] = t["local_path"].fillna("").astype(str)
    t["image_name"] = t["local_path"].map(os.path.basename)
    t["y_bac"] = (t["twoclass_bac"] > 0.5).astype(int)
    t["soft_bac"] = t["twoclass_bac"].astype(float)
    t["soft_fung"] = 1.0 - t["soft_bac"]
    t["has_posterior"] = True
    t["visit"] = t.get("visit", pd.Series("not encoded", index=t.index)).fillna("not encoded").astype(str)
    t["is_followup"] = t["visit"].isin(FOLLOWUP_VISITS)
    for c in TESTS:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    if verbose:
        print(f"Paper label table: {n0:,} rows; {len(t):,} with an image file and labels; "
              f"{t['subject'].nunique():,} subjects; splits " +
              ", ".join(f"{k} {v:,}" for k, v in t["split"].value_counts().items()) + ".")
    return t.reset_index(drop=True)


def split_paper(df, seed=42, group_col="subject", val_share=0.0):
    """
    The paper's own split (column 'split'): train / test, plus 'val' rows if the table has them.
    Without them, val_share > 0 carves that share of training subjects off as validation; 0 (default)
    leaves no validation set: the paper recipe never uses one to choose the model.
    The test set is split 20/80 into calibration / evaluation, grouped by subject.
    """
    split = df["split"].astype(str).to_numpy()
    groups = df[group_col].to_numpy()
    tr = np.where(split == "train")[0]
    va = np.where(split == "val")[0]
    te = np.where(split == "test")[0]
    if len(va) == 0 and val_share > 0:
        tr_rel, va_rel = next(GroupShuffleSplit(1, test_size=val_share, random_state=seed)
                              .split(tr, groups=groups[tr]))
        tr, va = tr[tr_rel], tr[va_rel]
    if len(te):
        ca_rel, ev_rel = next(GroupShuffleSplit(1, test_size=0.80, random_state=seed).split(te, groups=groups[te]))
    else:
        ca_rel = ev_rel = np.array([], dtype=int)
    parts = {"train": tr, "val": va, "test": te, "cal": te[ca_rel], "eval": te[ev_rel]}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test"), ("cal", "eval")]:
        shared = set(groups[parts[a]]) & set(groups[parts[b]])
        if shared:
            raise AssertionError(f"{len(shared)} subjects are in both {a} and {b}.")
    return parts


def targets(df, label_type, group_col="pID"):
    """
    Training targets, shape (N, 2): [bacterial, fungal].
      soft / hard      the LCA label rule (complementary).
      smear / culture  the raw tests as two independent targets (Gram, KOH or blood agar,
                       PDA), carried across the participant's images; NaN where missing.
                       Not complementary: [1, 1] and [0, 0] occur.
    """
    if label_type == "soft":
        return df[["soft_bac", "soft_fung"]].to_numpy(dtype=np.float32)
    if label_type == "hard":
        return df[["hard_bac", "hard_fung"]].to_numpy(dtype=np.float32)
    if label_type in TEST_TARGETS:
        cols = [TESTS.index(c) for c in TEST_TARGETS[label_type]]
        return test_results(df, group_col)[:, cols]
    if label_type == "paper_soft":     # the paper's per-pathogen posteriors (need not sum to 1)
        return df[["paper_soft_bac", "paper_soft_fung"]].to_numpy(dtype=np.float32)
    if label_type == "paper_hard":     # Travis: smear or culture positive, per pathogen
        return df[["hard_bac", "hard_fung"]].to_numpy(dtype=np.float32)
    if label_type == "culture_hard":   # blood agar and PDA themselves; NaN where not done
        return test_results(df, group_col)[:, [TESTS.index(c) for c in CULTURES]]
    if label_type == "twoclass_soft":  # P(bacterial | all four tests), complementary
        p = df["twoclass_bac"].to_numpy(dtype=np.float32)
        return np.stack([p, 1 - p], 1)
    if label_type == "twoclass_hard":  # its binary label
        y = df["y_bac"].to_numpy(dtype=np.float32)
        return np.stack([y, 1 - y], 1)
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


def split_cv(df, n_folds=5, fold=0, seed=42, group_col="pID", val_share=0.15):
    """
    Cross-validation split, grouped by participant and stratified by study x label so every
    fold has its share of each study (and of bacterial and fungal MADURAI participants).
      test  = fold `fold` of `n_folds` (StratifiedGroupKFold, shuffled with `seed`)
      val   = `val_share` of all participants, taken from the other folds
      train = the rest of the other folds
      test is then split 20/80 into calibration / evaluation, grouped by participant.
    Same seed for every model, so a fold holds the same participants in every run.
    """
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as e:  # scikit-learn < 1.0
        raise SystemExit("Cross-validation needs scikit-learn >= 1.0 (StratifiedGroupKFold).") from e
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in 0..{n_folds - 1}")
    groups = df[group_col].to_numpy()
    strat = (df["study"].astype(str) + "|" + df["y_bac"].astype(str)).to_numpy()
    idx = np.arange(len(df))
    import warnings
    with warnings.catch_warnings():  # small studies (CLAIR I) can't fill every fold; that's fine
        warnings.filterwarnings("ignore", message="The least populated class")
        folds = list(StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                     .split(idx, strat, groups))
    rest, te = folds[fold]
    va_size = val_share / (1 - 1 / n_folds)  # share of the remaining participants
    tr_rel, va_rel = next(GroupShuffleSplit(1, test_size=va_size, random_state=seed)
                          .split(rest, groups=groups[rest]))
    tr, va = rest[tr_rel], rest[va_rel]
    ca_rel, ev_rel = next(GroupShuffleSplit(1, test_size=0.80, random_state=seed)
                          .split(te, groups=groups[te]))
    ca, ev = te[ca_rel], te[ev_rel]
    parts = {"train": tr, "val": va, "test": te, "cal": ca, "eval": ev}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test"), ("cal", "eval")]:
        shared = set(groups[parts[a]]) & set(groups[parts[b]])
        if shared:
            raise AssertionError(f"{len(shared)} participants are in both {a} and {b}.")
    return parts


def participant_codes(df, group_col="pID"):
    """Integer participant codes for clustered statistics (no identifiers)."""
    return pd.factorize(df[group_col])[0].astype(np.int64)
