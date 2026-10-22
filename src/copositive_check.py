"""
Co-positive check: do the models treat patients whose tests point to BOTH pathogens as uncertain?

The labels cannot express co-infection (the LCA has two mutually exclusive classes), so this is a
descriptive check, not a class. Test results are combined per participant (they never conflict
within a participant) and every test image is put in one group:

  both cultures +       blood agar AND potato dextrose agar positive
  bac+ and fung+        a bacterial-positive and a fungal-positive test, not both cultures
  one pathogen +        positive tests for one pathogen only
  no positive test      tests recorded, none positive
  no test results       no smear or culture recorded (trial images labelled by study design)

For every run given (any head, any KL arm), each image gets an uncertainty percentile within the
test set for each signal (0.5 = typical image, 1 = most uncertain):
  entropy    entropy of the bacterial probability (every head)
  evidence   low BD evidence = uncertain (BD runs only)
  conformal  share of two-label binary LAC sets (calibrated on the run's calibration images)
The fair comparison is within the MADURAI study (same setting): co-positive vs one pathogen +.
Prints aggregates only; no identifiers.

    python src/copositive_check.py --runs RUN_A RUN_B RUN_C RUN_D [more runs] --out OUT_DIR
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import load_labelled  # noqa: E402

DEFAULT_CSV = ("/persist/QTIM/Active/26-0096/Corneal_Ulcer_Image_Sets/Data/"
               "corneal_ulcer_study_data_7_24_2025_lambda.csv")
TESTS = ["bac_smear", "fung_smear", "bac_cult_positive", "fung_cult_positive"]
ORDER = ["both cultures +", "bac+ and fung+", "one pathogen +", "no positive test", "no test results"]


def participant_groups(df, group_col):
    t = df[[group_col] + TESTS].copy()
    for c in TESTS:
        t[c] = t[c].where(t[c] >= 0)  # -1 = missing
    g = t.groupby(group_col)[TESTS].max()
    has = g.notna().any(axis=1)
    bac = (g["bac_smear"] == 1) | (g["bac_cult_positive"] == 1)
    fung = (g["fung_smear"] == 1) | (g["fung_cult_positive"] == 1)
    both_cult = (g["bac_cult_positive"] == 1) & (g["fung_cult_positive"] == 1)
    cat = pd.Series("no test results", index=g.index)
    cat[has & ~bac & ~fung] = "no positive test"
    cat[has & (bac ^ fung)] = "one pathogen +"
    cat[has & bac & fung] = "bac+ and fung+"
    cat[both_cult] = "both cultures +"
    return cat


def pct_rank(x):
    """Percentile of each value within the array (0 = least, 1 = most)."""
    return (rankdata(x) - 1) / max(len(x) - 1, 1)


def lac_two_label(p, y, cal, alpha=0.1):
    P = np.stack([1 - p, p], 1)
    s = np.sort(1 - P[cal, y[cal]])
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    q = 1.0 if k > n else s[k - 1]
    return (P >= 1 - q).sum(1) == 2


def describe(run):
    z = np.load(os.path.join(run, "test_predictions.npz"), allow_pickle=False)
    with open(os.path.join(run, "config.json")) as f:
        cfg = json.load(f)
    name = cfg.get("run_name") or os.path.basename(os.path.normpath(run))
    return z, cfg, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--group_col", default="pID")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = load_labelled(a.csv, img_root=None, group_col=a.group_col, verbose=False)
    cat_by_participant = participant_groups(df, a.group_col)
    row_cat = pd.Series(df[a.group_col].map(cat_by_participant).to_numpy(), index=df["csv_row"].to_numpy())
    row_pid = pd.Series(pd.factorize(df[a.group_col])[0], index=df["csv_row"].to_numpy())

    allp = cat_by_participant.value_counts().reindex(ORDER).fillna(0).astype(int)
    print("Participants per group, whole labelled dataset:", allp.to_dict())

    rows = []
    ref_rows = None
    for run in a.runs:
        z, cfg, name = describe(run)
        csv_row = z["csv_row"]
        if ref_rows is None:
            ref_rows = csv_row
            cat = row_cat.loc[csv_row].to_numpy()
            pid = row_pid.loc[csv_row].to_numpy()
            study = z["study"].astype(str)
            post = z["soft_bac"].astype(float)
            n_part = pd.DataFrame({"cat": cat, "pid": pid, "mad": study == "MADURAI"})
            print("\nTest images per group (participants):")
            for c in ORDER:
                m = n_part["cat"] == c
                print(f"  {c:18s} {int(m.sum()):4d} images ({n_part.loc[m, 'pid'].nunique():3d} participants; "
                      f"MADURAI {n_part.loc[m & n_part.mad, 'pid'].nunique()})")
        elif not np.array_equal(csv_row, ref_rows):
            raise SystemExit(f"{run} has a different test set; analyse runs from the same split together.")

        y = z["y_bac"].astype(int)
        p = z["p_bac_binary"].astype(float)
        cal = np.where(z["part"] == 0)[0]
        pc = np.clip(p, 1e-12, 1 - 1e-12)
        H = -(pc * np.log(pc) + (1 - pc) * np.log(1 - pc))
        sig = {"entropy": pct_rank(H), "conformal_two_label": lac_two_label(p, y, cal).astype(float)}
        if str(z["head"]) != "sigmoid":
            ev = z["output"][:, :4].astype(float).sum(1) / 2
            sig["evidence (low = uncertain)"] = pct_rank(-ev)
        arm = ("" if cfg.get("kl_schedule", "icmla") == "icmla"
               else f", KL {cfg.get('kl_form')} {cfg.get('kl_max')}")
        label = f"{cfg['head']} / {cfg['label_type']}{arm}"
        for scope, m_scope in (("all test images", np.ones(len(y), bool)), ("MADURAI study", study == "MADURAI")):
            for c in ORDER:
                m = m_scope & (cat == c)
                if not m.any():
                    continue
                r = {"run": label, "scope": scope, "group": c, "images": int(m.sum()),
                     "participants": int(len(np.unique(pid[m]))),
                     "mean_p_bacterial": p[m].mean(), "mean_posterior": post[m].mean(),
                     "share_p_between_0.3_0.7": np.mean((p[m] > 0.3) & (p[m] < 0.7))}
                for k, v in sig.items():
                    r[k] = v[m].mean()
                rows.append(r)

    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(a.out, "copositive_check.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print("\nUncertainty percentile within the test set (0.5 = typical image; higher = more uncertain);")
    print("conformal_two_label = share of images whose conformal set holds both labels.\n")
    for run, g in T.groupby("run", sort=False):
        print(f"=== {run}")
        print(g.drop(columns="run").round(3).to_string(index=False), "\n")
    print(f"Written to {a.out}/copositive_check.csv")


if __name__ == "__main__":
    main()
