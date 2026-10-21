"""
Story check for the v2 runs: does the abstract's story survive the label fix?

Reads the test_predictions.npz of the four cells (A sigmoid/soft, B BD/soft,
C sigmoid/hard, D BD/hard), scores everything as binary bacterial-vs-fungal
against the LCA bin, and tests the six story claims written down on 2026-10-21:

  1. Soft labels discriminate at least as well as hard labels (overall and MADURAI).
  2. Soft-label uncertainty tracks errors better than hard-label uncertainty.
  3. BD evidence tracks errors at least as well as sigmoid entropy.
  4. BD evidence combined with conformal beats either alone at the same coverage.
  5. A selected subset reaches high accuracy at useful coverage.
  6. Uncertainty is highest in MADURAI.

Uncertainty signal per cell: sigmoid cells use the entropy of the bacterial
probability p = (p_bac + 1 - p_fung) / 2; BD cells use mean evidence
(alpha_bac + beta_bac + alpha_fung + beta_fung) / 2 (low evidence = uncertain).
"Tracks errors" = error-detection AUROC: how well uncertainty ranks wrong
predictions above right ones (0.5 = no better than chance).

Conformal: binary LAC, calibrated on the calibration participants, textbook
rank k = ceil((n + 1)(1 - alpha)). Confidence intervals: 95% bootstrap over
participants (images are not independent). Verdicts: "holds" (claimed
direction, CI excludes 0), "inconclusive" (CI includes 0), "reversed"
(opposite direction, CI excludes 0).

    python src/analyze_v2.py --A RUN_A --B RUN_B --C RUN_C --D RUN_D --out OUT_DIR
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CELLS = {"A": ("sigmoid", "soft"), "B": ("beta_dirichlet", "soft"),
         "C": ("sigmoid", "hard"), "D": ("beta_dirichlet", "hard")}


def load(run):
    z = np.load(os.path.join(run, "test_predictions.npz"), allow_pickle=False)
    return {k: z[k] for k in z.files}


def uncertainty(z):
    """Higher = more uncertain."""
    p = np.clip(z["p_bac_binary"].astype(float), 1e-12, 1 - 1e-12)
    if str(z["head"]) == "sigmoid":
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))
    o = z["output"].astype(float)
    return -((o[:, 0] + o[:, 1] + o[:, 2] + o[:, 3]) / 2)  # negative evidence


def auroc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan


def lac_sets(p, y, cal, alpha=0.1):
    """Binary LAC; returns (set membership [N, 2], qhat). Textbook k-th smallest score."""
    P = np.stack([1 - p, p], 1)
    s = np.sort(1 - P[cal, y[cal]])
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    q = 1.0 if k > n else s[k - 1]  # too few calibration points: full sets
    return P >= 1 - q, q


def boot(stat, groups, idx, reps=1000, seed=0):
    """Participant-level bootstrap of stat(index_array) -> 2.5% and 97.5% percentiles."""
    rng = np.random.default_rng(seed)
    g = groups[idx]
    ug = np.unique(g)
    members = {u: idx[g == u] for u in ug}
    vals = []
    for _ in range(reps):
        pick = rng.choice(ug, size=len(ug), replace=True)
        b = np.concatenate([members[u] for u in pick])
        v = stat(b)
        if np.isfinite(v):
            vals.append(v)
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)


def verdict(diff, ci):
    """Claimed direction is always 'first minus second > 0'."""
    lo, hi = ci
    if np.isnan(diff) or np.isnan(lo):
        return "n/a"
    if lo > 0:
        return "holds"
    if hi < 0:
        return "reversed"
    return "inconclusive"


def md(df):
    try:
        return df.to_markdown(index=False, floatfmt=".3f")
    except ImportError:  # tabulate not installed
        return "```\n" + df.to_string(index=False, float_format=lambda v: f"{v:.3f}") + "\n```"


def main():
    ap = argparse.ArgumentParser()
    for c in CELLS:
        ap.add_argument(f"--{c}", required=True, help=f"run folder for Cell {c}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--exclude_followups_eval", action="store_true")
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    Z = {c: load(getattr(a, c)) for c in CELLS}
    for c, (head, lab) in CELLS.items():
        assert str(Z[c]["head"]) == head and str(Z[c]["label_type"]) == lab, \
            f"Cell {c} folder holds {Z[c]['head']} / {Z[c]['label_type']}"
    ref = Z["A"]
    for c in "BCD":  # same test images, same order
        assert np.array_equal(Z[c]["csv_row"], ref["csv_row"]), f"Cell {c} has a different test set"

    y = ref["y_bac"].astype(int)
    study = ref["study"].astype(str)
    grp = ref["participant"]
    cal = np.where(ref["part"] == 0)[0]
    ev = np.where(ref["part"] == 1)[0]
    if a.exclude_followups_eval:
        ev = ev[~ref["is_followup"][ev].astype(bool)]
    mad = study == "MADURAI"
    subsets = {"eval": ev, "eval_MADURAI": ev[mad[ev]], "eval_other": ev[~mad[ev]]}

    p = {c: Z[c]["p_bac_binary"].astype(float) for c in CELLS}
    u = {c: uncertainty(Z[c]) for c in CELLS}
    correct = {c: (p[c] > 0.5) == (y == 1) for c in CELLS}

    # ---- per-cell metrics ----------------------------------------------------
    rows = []
    sets = {}
    for c in CELLS:
        S, q = lac_sets(p[c], y, cal, a.alpha)
        sets[c] = S
        for name, idx in subsets.items():
            single = S[idx].sum(1) == 1
            cov = S[idx, y[idx]].mean()
            rows.append({
                "cell": c, "subset": name, "n_images": len(idx), "participants": len(np.unique(grp[idx])),
                "auroc": auroc(y[idx], p[c][idx]),
                "accuracy": correct[c][idx].mean(),
                "error_detection_auroc": auroc(~correct[c][idx], u[c][idx]),
                "conformal_coverage": cov, "singleton_rate": single.mean(),
                "singleton_accuracy": correct[c][idx][single].mean() if single.any() else np.nan,
                "qhat": q,
            })
    per_cell = pd.DataFrame(rows)
    per_cell.to_csv(os.path.join(a.out, "per_cell_metrics.csv"), index=False)

    # ---- story claims ----------------------------------------------------------
    story = []

    def paired(c1, c2, metric_fn, name, claim):
        idx = subsets[name]
        d = metric_fn(c1, idx) - metric_fn(c2, idx)
        ci = boot(lambda b: metric_fn(c1, b) - metric_fn(c2, b), grp, idx, a.reps)
        story.append({"claim": claim, "subset": name, "comparison": f"{c1} - {c2}",
                      "difference": d, "ci_low": ci[0], "ci_high": ci[1], "verdict": verdict(d, ci)})

    disc = lambda c, b: auroc(y[b], p[c][b])
    errdet = lambda c, b: auroc(~correct[c][b], u[c][b])
    for name in ("eval", "eval_MADURAI"):
        paired("A", "C", disc, name, "1 soft >= hard discrimination (sigmoid)")
        paired("B", "D", disc, name, "1 soft >= hard discrimination (BD)")
        paired("A", "C", errdet, name, "2 soft uncertainty tracks errors better (sigmoid)")
        paired("B", "D", errdet, name, "2 soft uncertainty tracks errors better (BD)")
        paired("B", "A", errdet, name, "3 BD evidence >= sigmoid entropy (soft)")
        paired("D", "C", errdet, name, "3 BD evidence >= sigmoid entropy (hard)")

    # 4 and 5: combined filter = conformal singleton (sigmoid) AND high BD evidence
    # (above the calibration 67th percentile), compared with each signal alone at the
    # same coverage.
    comb_rows = []
    for sig, bd in (("A", "B"), ("C", "D")):
        evid = -u[bd]
        t67 = np.percentile(evid[cal], 67)
        for name in ("eval", "eval_MADURAI"):
            idx = subsets[name]
            single = sets[sig][idx].sum(1) == 1
            sel = single & (evid[idx] > t67)
            cov = sel.mean()
            n = max(int(round(cov * len(idx))), 1)
            acc_comb = correct[sig][idx][sel].mean() if sel.any() else np.nan
            top_conf = np.argsort(u[sig][idx])[:n]          # most confident sigmoid
            top_evid = np.argsort(-evid[idx])[:n]           # highest evidence
            acc_conf = correct[sig][idx][top_conf].mean()
            acc_evid = correct[sig][idx][top_evid].mean()
            comb_rows.append({"pair": f"{sig}+{bd}", "subset": name, "coverage": cov,
                              "accuracy_all": correct[sig][idx].mean(),
                              "accuracy_combined": acc_comb,
                              "accuracy_confidence_alone_same_coverage": acc_conf,
                              "accuracy_evidence_alone_same_coverage": acc_evid})
            story.append({"claim": f"5 high-accuracy subset ({sig}+{bd})", "subset": name,
                          "comparison": f"accuracy at {cov:.0%} coverage (all: {correct[sig][idx].mean():.0%})",
                          "difference": acc_comb, "ci_low": np.nan, "ci_high": np.nan,
                          "verdict": "judge"})
            better = acc_comb - max(acc_conf, acc_evid)
            story.append({"claim": f"4 combined beats either alone ({sig}+{bd})", "subset": name,
                          "comparison": "combined - best single", "difference": better,
                          "ci_low": np.nan, "ci_high": np.nan,
                          "verdict": ("n/a" if np.isnan(better) else "holds" if better > 0.01
                                      else "reversed" if better < -0.01 else "inconclusive")})
    combined = pd.DataFrame(comb_rows)
    combined.to_csv(os.path.join(a.out, "combined_filter.csv"), index=False)

    # 6: uncertainty by study
    by_study = pd.DataFrame({"study": study[ev], **{f"uncertainty_{c}": u[c][ev] for c in CELLS}})
    by_study = by_study.groupby("study").agg(["mean", "size"])
    by_study.to_csv(os.path.join(a.out, "uncertainty_by_study.csv"))
    for c in CELLS:
        means = pd.Series(u[c][ev]).groupby(study[ev]).mean()
        means = means[pd.Series(study[ev]).value_counts().reindex(means.index) >= 10]  # skip tiny cohorts
        story.append({"claim": f"6 uncertainty highest in MADURAI (Cell {c})", "subset": "eval, cohorts n>=10",
                      "comparison": f"highest: {means.idxmax()}", "difference": np.nan,
                      "ci_low": np.nan, "ci_high": np.nan,
                      "verdict": "holds" if means.idxmax() == "MADURAI" else "fails"})

    story = pd.DataFrame(story)
    story.to_csv(os.path.join(a.out, "story_check.csv"), index=False)

    with open(os.path.join(a.out, "story_check.md"), "w") as f:
        f.write("# Story check (v2 runs)\n\n")
        f.write(f"Reference: LCA bin, binary. Conformal alpha = {a.alpha}. "
                f"Follow-up photos {'excluded from' if a.exclude_followups_eval else 'included in'} evaluation.\n\n")
        f.write("## Claims\n\n" + md(story) + "\n\n")
        f.write("## Per-cell metrics\n\n" + md(per_cell) + "\n\n")
        f.write("## Combined filter (claims 4 and 5)\n\n" + md(combined) + "\n")
    print(story.to_string(index=False))
    print(f"\nWritten to {a.out}")


if __name__ == "__main__":
    main()
