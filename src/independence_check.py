"""
Are bacterial and fungal infection independent? Three checks, aggregates only.

The old 4-class scoring combined the two per-pathogen probabilities as if independent given the image:
    P(both) = b*f, P(bac only) = b(1-f), P(fung only) = (1-b)f, P(neither) = (1-b)(1-f).
That is only right if the two outcomes are conditionally independent. The models' outputs alone can't
test this (b and f can be anti-correlated across images even when outcomes are independent), so the
checks compare against outcomes:

  1. Raw tests (participant level; only the Madurai site has results). For each pair of tests
     (blood agar vs PDA culture, Gram vs KOH smear, any bacterial+ vs any fungal+), the 2x2 counts,
     the both-positive count expected if positivity were independent, the odds ratio, and the
     both-positive count expected if every ulcer had ONE pathogen and the extra positives were test
     false positives (supplement LCA sensitivity/specificity; fungal prevalence set three ways).
  2. Labels vs the product rule (needs --runs): the LCA bins are complementary, so "both" and
     "neither" never occur; the product rule's expected counts for them, per study.
  3. Model outputs (needs --runs): per study, Spearman(b, f), share of images with |b + f - 1| > 0.1,
     both > 0.5, both < 0.5; and the product-rule P(both) by participant test group (co-positive vs
     one pathogen + vs no positive test), against the observed co-positive share.

    python src/independence_check.py --out OUT_DIR [--runs RUN ...] [--csv CSV]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import STUDY_SHORT, load_labelled  # noqa: E402
from copositive_check import ORDER, TESTS, participant_groups  # noqa: E402

DEFAULT_CSV = ("/persist/QTIM/Active/26-0096/Corneal_Ulcer_Image_Sets/Data/"
               "corneal_ulcer_study_data_7_24_2025_lambda.csv")

# Supplement LCA estimates (sensitivity, specificity) and fungal prevalence.
SENS_SPEC = {"bac_smear": (0.40, 0.97), "fung_smear": (0.94, 0.94),
             "bac_cult_positive": (0.37, 0.96), "fung_cult_positive": (0.72, 0.98)}
PI_FUNGAL_SUPPLEMENT = 0.56
PAIRS = [("blood agar vs PDA culture", "bac_cult_positive", "fung_cult_positive"),
         ("Gram vs KOH smear", "bac_smear", "fung_smear")]


def participant_tests(df, group_col):
    t = df[[group_col] + TESTS].copy()
    for c in TESTS:
        t[c] = t[c].where(t[c] >= 0)  # -1 = missing
    return t.groupby(group_col)[TESTS].max()


def two_by_two(x, y):
    """x, y: 0/1 arrays (bacterial test, fungal test). Returns counts and summary."""
    n11 = int(np.sum((x == 1) & (y == 1)))
    n10 = int(np.sum((x == 1) & (y == 0)))
    n01 = int(np.sum((x == 0) & (y == 1)))
    n00 = int(np.sum((x == 0) & (y == 0)))
    n = n11 + n10 + n01 + n00
    out = {"participants": n, "both +": n11, "bacterial only +": n10, "fungal only +": n01, "neither +": n00}
    if n == 0:
        return out
    pb, pf = (n11 + n10) / n, (n11 + n01) / n
    out.update({"bacterial + rate": pb, "fungal + rate": pf,
                "both + expected if independent": n * pb * pf})
    a, b, c, d = n11 + .5, n10 + .5, n01 + .5, n00 + .5  # Haldane correction
    lor = np.log(a * d / (b * c))
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    out.update({"odds ratio": float(np.exp(lor)), "OR ci_low": float(np.exp(lor - 1.96 * se)),
                "OR ci_high": float(np.exp(lor + 1.96 * se))})
    if min(pb, pf) > 0 and max(pb, pf) < 1:
        out["Fisher p"] = float(fisher_exact([[n11, n10], [n01, n00]])[1])
    return out


def expected_both_one_pathogen(n, pb, pf, test_b, test_f):
    """Both-positive count expected if every ulcer had one pathogen, tests conditionally independent
    given the pathogen, supplement sensitivity/specificity; fungal prevalence pi set three ways."""
    sb, cb = SENS_SPEC[test_b]
    sf, cf = SENS_SPEC[test_f]
    pis = {"supplement": PI_FUNGAL_SUPPLEMENT,
           "matched to fungal + rate": (pf - (1 - cf)) / (sf - (1 - cf)),
           "matched to bacterial + rate": (sb - pb) / (sb - (1 - cb))}
    out = {}
    for k, pi in pis.items():
        pi = float(np.clip(pi, 0, 1))
        out[k] = n * (pi * (1 - cb) * sf + (1 - pi) * sb * (1 - cf))
    return out


def load_run(run):
    z = np.load(os.path.join(run, "test_predictions.npz"), allow_pickle=False)
    with open(os.path.join(run, "config.json")) as f:
        cfg = json.load(f)
    arm = "" if cfg.get("kl_schedule", "icmla") == "icmla" else f", KL {cfg.get('kl_form')} {cfg.get('kl_max')}"
    name = f"{cfg['head']} / {cfg['label_type']}{arm}"
    if cfg.get("fold") is not None:
        name += f" / fold {cfg['fold']}"
    return name, {k: z[k] for k in z.files}


def md(df):
    try:
        return df.to_markdown(index=False, floatfmt=".3f")
    except ImportError:
        return "```\n" + df.to_string(index=False, float_format=lambda v: f"{v:.3f}") + "\n```"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--group_col", default="pID")
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = load_labelled(a.csv, img_root=None, group_col=a.group_col, verbose=False)
    df["study_short"] = [STUDY_SHORT.get(s, s) for s in df["study"]]
    g = participant_tests(df, a.group_col)
    first = df.groupby(a.group_col).first()
    g["study"] = first["study_short"].reindex(g.index)
    g["site"] = first["study_site"].reindex(g.index) if "study_site" in df.columns else "unknown"
    any_b = ((g["bac_smear"] == 1) | (g["bac_cult_positive"] == 1)).astype(float)
    any_f = ((g["fung_smear"] == 1) | (g["fung_cult_positive"] == 1)).astype(float)
    has_any = g[TESTS].notna().any(axis=1)
    any_b[~has_any] = np.nan
    any_f[~has_any] = np.nan
    g["any bacterial +"], g["any fungal +"] = any_b, any_f

    # ---- 1. raw tests -----------------------------------------------------------
    scopes = [("All participants with results", np.ones(len(g), bool))]
    scopes += [(f"study: {s}", (g["study"] == s).to_numpy()) for s in sorted(g["study"].dropna().unique())]
    scopes += [(f"site: {s}", (g["site"] == s).to_numpy()) for s in sorted(g["site"].dropna().unique())]
    rows = []
    for scope, m in scopes:
        for pair, tb, tf in PAIRS + [("any bacterial + vs any fungal +", "any bacterial +", "any fungal +")]:
            sub = g[m & g[tb].notna().to_numpy() & g[tf].notna().to_numpy()]
            if len(sub) == 0:
                continue
            r = {"scope": scope, "tests": pair}
            r.update(two_by_two(sub[tb].to_numpy(), sub[tf].to_numpy()))
            if tb in SENS_SPEC and "bacterial + rate" in r:
                e = expected_both_one_pathogen(r["participants"], r["bacterial + rate"], r["fungal + rate"], tb, tf)
                r["both + expected, one pathogen + test error (min)"] = min(e.values())
                r["both + expected, one pathogen + test error (max)"] = max(e.values())
                r["... with supplement prevalence"] = e["supplement"]
            rows.append(r)
    T1 = pd.DataFrame(rows)
    T1.to_csv(os.path.join(a.out, "tests_2x2.csv"), index=False)

    # ---- 2 and 3. runs ------------------------------------------------------------------
    cat_p = participant_groups(df, a.group_col)
    row_cat = pd.Series(df[a.group_col].map(cat_p).to_numpy(), index=df["csv_row"].to_numpy())
    rows2, rows3 = [], []
    for run in a.runs:
        name, z = load_run(run)
        b = z["p_hat"][:, 0].astype(float)
        f = z["p_hat"][:, 1].astype(float)
        study = z["study"].astype(str)
        cat = row_cat.reindex(z["csv_row"]).to_numpy()
        both, neither = b * f, (1 - b) * (1 - f)
        for s_name, m in [("All", np.ones(len(b), bool)), ("Single-etiology trials", study != "MADURAI")] + \
                         [(s, study == s) for s in sorted(set(study))]:
            if m.sum() < 5:
                continue
            rows2.append({"run": name, "scope": s_name, "images": int(m.sum()),
                          "Spearman(b, f)": spearmanr(b[m], f[m])[0],
                          "share b + f off 1 by > 0.1": np.mean(np.abs(b[m] + f[m] - 1) > 0.1),
                          "share both > 0.5": np.mean((b[m] > .5) & (f[m] > .5)),
                          "share both < 0.5": np.mean((b[m] < .5) & (f[m] < .5)),
                          "product rule: expected 'both' images": both[m].sum(),
                          "product rule: expected 'neither' images": neither[m].sum(),
                          "observed in LCA labels": 0})
        mad = study == "MADURAI"
        for c in ORDER:
            m = mad & (cat == c)
            if not m.any():
                continue
            rows3.append({"run": name, "MADURAI group": c, "images": int(m.sum()),
                          "participants": int(len(np.unique(z["participant"][m]))),
                          "mean P(both) = b*f": both[m].mean(), "mean P(neither)": neither[m].mean(),
                          "mean b + f": (b[m] + f[m]).mean()})
        m = mad & np.isin(cat, ORDER[:-1])  # MADURAI test images with any test recorded
        if m.any():
            pc = pd.Series(cat[m]).groupby(z["participant"][m]).first()
            rows3.append({"run": name, "MADURAI group": "all with results (observed co-positive share)",
                          "images": int(m.sum()), "participants": int(len(pc)),
                          "mean P(both) = b*f": both[m].mean(), "mean P(neither)": neither[m].mean(),
                          "mean b + f": (b[m] + f[m]).mean(),
                          "observed co-positive share (participants)":
                              float(np.isin(pc, ["both cultures +", "bac+ and fung+"]).mean())})
    T2, T3 = pd.DataFrame(rows2), pd.DataFrame(rows3)
    if len(T2):
        T2.to_csv(os.path.join(a.out, "outputs_by_study.csv"), index=False)
        T3.to_csv(os.path.join(a.out, "product_rule_by_test_group.csv"), index=False)

    with open(os.path.join(a.out, "independence.md"), "w") as fh:
        fh.write("# Independence of bacterial and fungal infection\n\n"
                 "## 1. Raw tests, participant level\n\n"
                 "Expected 'both +' two ways: if positivity were independent, and if every ulcer had one "
                 "pathogen with the extra positives coming from test false positives (supplement LCA "
                 "sensitivity/specificity; fungal prevalence from the supplement, or matched to either "
                 "marginal rate: min and max shown).\n\n" + md(T1) + "\n\n")
        if len(T2):
            fh.write("## 2. Model outputs and the product rule, by study\n\n" + md(T2) + "\n\n"
                     "## 3. Product-rule P(both) by participant test group (MADURAI)\n\n" + md(T3) + "\n")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show = ["scope", "tests", "participants", "both +", "both + expected if independent",
            "both + expected, one pathogen + test error (min)", "both + expected, one pathogen + test error (max)",
            "odds ratio", "OR ci_low", "OR ci_high"]
    print(T1[[c for c in show if c in T1.columns]].round(3).to_string(index=False))
    print(f"\nWritten to {a.out}/independence.md")


if __name__ == "__main__":
    main()
