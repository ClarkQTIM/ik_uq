"""
Choose BD's KL setting on the validation set, for each analysis and label (soft, hard), from the runs of
scripts/launch_kl_redo.sh. Test images are never used.

Two things are wanted from a BD model, and they can pull apart (a stronger KL term has given better evidence and
lower AUROC), so both are reported and three picks made, each fixed before the sweep:
  discrimination   mean over the two outputs of the validation AUROC against the analysis's reference
                   (LCA analysis: posterior > 0.5; culture analysis: blood agar, PDA)
  evidence         mean over the two outputs of how well low evidence (-(alpha + beta)) flags the validation
                   errors (AUROC, wrong vs right at 0.5); entropy's own figure is shown alongside
  picks            best discrimination; best evidence; and the proposed rule: the best evidence among settings
                   whose discrimination is within --tolerance of the best
Validation loss is not used: it contains the KL term, so it isn't comparable across settings. Intervals are a
patient-level bootstrap on the validation set. Aggregates only.

    python src/select_kl.py --runs results/provisional_kl [--tolerance 0.02] [--boot 500]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eda_paper_data import md  # noqa: E402


def kl_name(c):
    if c["kl_schedule"] == "icmla":
        return "published"
    return ("Sensoy" if c["kl_form"] == "sensoy" else "ICMLA form") + f" {c['kl_max']:g}"


def scores(p, out, ref):
    """(discrimination, evidence, entropy) over the two outputs; NaN where undefined."""
    d, e, h = [], [], []
    for j in range(2):
        m = ~np.isnan(ref[:, j])
        t, pj = ref[m, j], np.clip(p[m, j], 1e-7, 1 - 1e-7)
        if len(np.unique(t)) == 2:
            d.append(roc_auc_score(t, pj))
        err = (pj > 0.5) != (t == 1)
        if len(np.unique(err)) == 2:
            e.append(roc_auc_score(err, -(out[m, 2 * j] + out[m, 2 * j + 1])))
            h.append(roc_auc_score(err, -(pj * np.log(pj) + (1 - pj) * np.log(1 - pj))))
    return tuple(float(np.mean(x)) if x else np.nan for x in (d, e, h))


def boot(p, out, ref, part, n, rng):
    ids = np.unique(part)
    rows = {k: np.where(part == k)[0] for k in ids}
    res = []
    for _ in range(n):
        pick = np.concatenate([rows[k] for k in rng.choice(ids, len(ids))])
        res.append(scores(p[pick], out[pick], ref[pick])[:2])
    return np.nanpercentile(np.array(res), [2.5, 97.5], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="folder holding the sweep's run folders")
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="default: <runs>/kl_selection")
    a = ap.parse_args()
    out_dir = a.out or os.path.join(a.runs, "kl_selection")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    rows = []
    for vm in sorted(glob.glob(os.path.join(a.runs, "*", "val_metrics.json"))):
        c = json.load(open(vm))
        if c["head"] != "beta_dirichlet":
            continue
        z = np.load(os.path.join(os.path.dirname(vm), "val_predictions.npz"), allow_pickle=False)
        analysis = "culture" if c["subset"] == "culture" else "LCA"
        ref = z["culture"] if analysis == "culture" else z["lca_hard_labels"]
        r = {"analysis": analysis, "label": "soft" if c["label_type"] == "paper_soft" else "hard",
             "KL": kl_name(c), "run": os.path.basename(os.path.dirname(vm)),
             "validation images (patients)": f"{c['n_images']:,} ({c['n_patients']:,})"}
        for scope, m in (("all", np.ones(len(ref), bool)), ("MADURAI", z["study"] == "MADURAI")):
            dsc, ev, ent = scores(z["p_hat"][m], z["output"][m], ref[m])
            r[f"{scope}: discrimination"], r[f"{scope}: evidence"], r[f"{scope}: entropy"] = dsc, ev, ent
            if a.boot and scope == "all":
                lo, hi = boot(z["p_hat"], z["output"], ref, z["participant"], a.boot, rng)
                r["discrimination 95%"] = f"{lo[0]:.3f}-{hi[0]:.3f}"
                r["evidence 95%"] = f"{lo[1]:.3f}-{hi[1]:.3f}"
        rows.append(r)
    if not rows:
        raise SystemExit(f"No BD runs with val_metrics.json under {a.runs}")
    t = pd.DataFrame(rows).sort_values(["analysis", "label", "KL"])
    t.to_csv(os.path.join(out_dir, "kl_validation.csv"), index=False)

    def best(g, col):
        g = g[g[col].notna()]
        return g.loc[g[col].idxmax(), "KL"] if len(g) else "n/a (undefined)"

    def picks(g, who):
        best_d = g["all: discrimination"].max()
        ok = g[g["all: discrimination"] >= best_d - a.tolerance] if np.isfinite(best_d) else g.iloc[:0]
        return {"choice for": who, "best discrimination": best(g, "all: discrimination"),
                "best evidence": best(g, "all: evidence"),
                f"rule (best evidence, discrimination within {a.tolerance})": best(ok, "all: evidence")}
    pk = [picks(g, f"{an}, {lb}") for (an, lb), g in t.groupby(["analysis", "label"])]
    pooled = (t.groupby(["analysis", "KL"])[["all: discrimination", "all: evidence"]].mean().reset_index())
    pk += [picks(g.reset_index(drop=True), f"{an}, soft and hard together") for an, g in pooled.groupby("analysis")]
    pk = pd.DataFrame(pk)
    pk.to_csv(os.path.join(out_dir, "kl_picks.csv"), index=False)

    show = ["analysis", "label", "KL", "validation images (patients)", "all: discrimination", "discrimination 95%",
            "all: evidence", "evidence 95%", "all: entropy", "MADURAI: discrimination", "MADURAI: evidence"]
    with open(os.path.join(out_dir, "kl_selection.md"), "w") as f:
        f.write("# BD's KL setting, chosen on the validation set\n\nAggregates only. Discrimination: mean AUROC of the "
                "two outputs against the analysis's reference. Evidence (and entropy): mean AUROC for flagging the "
                "validation errors. Intervals: patient-level bootstrap.\n\n## Picks\n\n" + md(pk)
                + "\n\n## Every setting\n\n" + md(t[[c for c in show if c in t.columns]]) + "\n")
    print(pk.to_string(index=False))
    print(f"\nReport: {out_dir}/kl_selection.md")


if __name__ == "__main__":
    main()
