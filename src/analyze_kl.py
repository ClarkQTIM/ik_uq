"""
KL ablation for the Beta-Dirichlet (BD) cells.

Question: with a meaningful KL weight, does BD evidence become an uncertainty signal?
Criteria fixed before the runs (2026-10-21):
  (a) on enrolment photos and within MADURAI, evidence flags errors at least as well as the
      BD model's own entropy (error-detection AUROC, participant bootstrap CI);
  (b) sigmoid conformal singleton + high BD evidence beats sigmoid confidence alone at the
      same coverage (the abstract's combined filter);
  also reported: AUROC, Brier, calibration error, accuracy, so any cost of the KL term shows.
Every arm is reported, whatever its result.

    python src/analyze_kl.py --A RUN_A --C RUN_C --bd RUN_B RUN_D RUN_B_klS1 RUN_D_klS1 ... --out OUT
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_v2 import auroc, boot, lac_sets, load, md  # noqa: E402


def entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (p >= lo) & ((p < hi) if i < bins - 1 else (p <= hi))
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def arm_name(cfg):
    if cfg.get("kl_schedule", "icmla") == "icmla":
        return "ICMLA schedule (published)"
    form = "Sensoy" if cfg.get("kl_form", "sensoy") == "sensoy" else "ICMLA eq. 5"
    return f"{form} form, kl_max {cfg['kl_max']:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", required=True, help="sigmoid / soft run (conformal for soft BD arms)")
    ap.add_argument("--C", required=True, help="sigmoid / hard run (conformal for hard BD arms)")
    ap.add_argument("--bd", nargs="+", required=True, help="BD run folders (any arms, soft or hard)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    sig = {"soft": load(a.A), "hard": load(a.C)}
    ref = sig["soft"]
    y = ref["y_bac"].astype(int)
    study = ref["study"].astype(str)
    grp = ref["participant"]
    fu = ref["is_followup"].astype(bool)
    cal = np.where(ref["part"] == 0)[0]
    ev = np.where(ref["part"] == 1)[0]
    subsets = {"eval": ev, "eval_enrolment_only": ev[~fu[ev]], "eval_MADURAI": ev[study[ev] == "MADURAI"]}
    sig_sets = {k: lac_sets(z["p_bac_binary"].astype(float), y, cal)[0] for k, z in sig.items()}
    sig_conf = {k: np.abs(z["p_bac_binary"].astype(float) - 0.5) for k, z in sig.items()}
    sig_ok = {k: (z["p_bac_binary"] > 0.5) == (y == 1) for k, z in sig.items()}

    rows, comb = [], []
    for run in a.bd:
        z = load(run)
        with open(os.path.join(run, "config.json")) as f:
            cfg = json.load(f)
        assert str(z["head"]) == "beta_dirichlet", f"{run} is not a BD run"
        assert np.array_equal(z["csv_row"], ref["csv_row"]), f"{run} has a different test set"
        lab = str(z["label_type"])
        arm = arm_name(cfg)
        p = z["p_bac_binary"].astype(float)
        ok = (p > 0.5) == (y == 1)
        evid = z["output"][:, :4].astype(float).sum(1) / 2
        H = entropy(p)
        t67 = np.percentile(evid[cal], 67)
        for name, idx in subsets.items():
            d = auroc(~ok[idx], -evid[idx]) - auroc(~ok[idx], H[idx])
            ci = boot(lambda b: auroc(~ok[b], -evid[b]) - auroc(~ok[b], H[b]), grp, idx, a.reps)
            verdict = ("evidence >= entropy" if d >= 0 else
                       "not clearly worse (CI includes 0)" if ci[1] >= 0 else "evidence worse")
            rows.append({
                "arm": arm, "labels": lab, "subset": name, "n": len(idx),
                "auroc": auroc(y[idx], p[idx]), "accuracy": ok[idx].mean(),
                "brier": np.mean((p[idx] - y[idx]) ** 2), "ece": ece(p[idx], y[idx]),
                "errdet_evidence": auroc(~ok[idx], -evid[idx]), "errdet_entropy": auroc(~ok[idx], H[idx]),
                "evidence_minus_entropy": d, "ci_low": ci[0], "ci_high": ci[1], "criterion_a": verdict,
                "evidence_median_right": np.median(evid[idx][ok[idx]]),
                "evidence_median_wrong": np.median(evid[idx][~ok[idx]]),
                "spearman_evidence_confidence": spearmanr(evid[idx], np.abs(p[idx] - 0.5))[0],
            })
            # criterion (b): sigmoid conformal singleton + high BD evidence vs each alone
            single = sig_sets[lab][idx].sum(1) == 1
            sel = single & (evid[idx] > t67)
            cov = sel.mean()
            n = max(int(round(cov * len(idx))), 1)
            s_ok = sig_ok[lab][idx]
            acc_comb = s_ok[sel].mean() if sel.any() else np.nan
            acc_conf = s_ok[np.argsort(-sig_conf[lab][idx])[:n]].mean()
            acc_evid = s_ok[np.argsort(-evid[idx])[:n]].mean()
            comb.append({"arm": arm, "labels": lab, "subset": name, "coverage": cov,
                         "accuracy_all": s_ok.mean(), "accuracy_combined": acc_comb,
                         "accuracy_sigmoid_confidence_alone": acc_conf,
                         "accuracy_evidence_alone": acc_evid,
                         "criterion_b": ("met" if acc_comb > max(acc_conf, acc_evid) + 0.01 else "not met")})
        by_study = pd.Series(evid[ev]).groupby(study[ev]).mean().round(2).to_dict()
        print(f"{arm:32s} {lab}: mean evidence by study {by_study}")

    T = pd.DataFrame(rows).sort_values(["labels", "subset", "arm"])
    Cb = pd.DataFrame(comb).sort_values(["labels", "subset", "arm"])
    T.to_csv(os.path.join(a.out, "kl_ablation.csv"), index=False)
    Cb.to_csv(os.path.join(a.out, "kl_ablation_combined.csv"), index=False)
    with open(os.path.join(a.out, "kl_ablation.md"), "w") as f:
        f.write("# KL ablation (BD cells)\n\nReference: LCA bin, binary. Error detection = AUROC of "
                "the signal for ranking wrong predictions above right ones.\n\n")
        f.write("## Per arm\n\n" + md(T) + "\n\n## Combined filter\n\n" + md(Cb) + "\n")
    pd.set_option("display.width", 250)
    cols = ["arm", "labels", "subset", "auroc", "brier", "ece", "errdet_evidence", "errdet_entropy",
            "evidence_minus_entropy", "ci_low", "ci_high", "criterion_a"]
    print(T[cols].round(3).to_string(index=False))
    print(Cb.round(3).to_string(index=False))
    print(f"\nWritten to {a.out}")


if __name__ == "__main__":
    main()
