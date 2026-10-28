"""
What each analysis gets: images, patients and labels, for everything in the paper's files ("full") and for what is
on our server now ("provisional", the runs we can do before the missing trial photos arrive).

Two analyses (decided 28 Oct 2026):
  LCA      all images. Soft = the paper's per-pathogen posteriors; hard = posterior > 0.5 (Travis's rule).
           Reference for scoring = the hard labels.
  culture  images with at least one culture (blood agar or PDA). Soft = the same posteriors; hard = the cultures
           (blood agar -> bacterial, PDA -> fungal). Reference for scoring = the cultures.
Both: two independent outputs (bacterial, fungal), the paper's train / validation / test split, the test split
divided 20/80 into calibration / evaluation by patient (seed 42), exactly as the training code does it.

Reads results/labels/paper_labels.csv (from src/paper_labels.py). Aggregates only.

    python src/analysis_sets.py --labels results/labels/paper_labels.csv --out results/analysis_sets
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eda_paper_data import md  # noqa: E402
from labels import has_culture, load_paper_table, split_paper, targets  # noqa: E402

ANALYSES = {"LCA": ("paper_hard", "all images; hard = LCA posterior > 0.5"),
            "culture": ("culture_hard", "images with a culture; hard = blood agar (bacterial), PDA (fungal)")}
VERSIONS = {"full": "everything in the paper's files", "provisional": "images on our server now"}
SPLITS = ["train", "val", "test"]
COMBOS = {"bacterial only": (1, 0), "fungal only": (0, 1), "both": (1, 1), "neither": (0, 0)}


def ip(d):
    """'images (patients)'."""
    return f"{len(d):,} ({d['subject'].nunique():,})" if len(d) else "0"


def build(path, seed):
    out = {}
    for v in VERSIONS:
        base = load_paper_table(path, verbose=False, require_image=(v == "provisional"))
        for a, (hard_type, _) in ANALYSES.items():
            d = base if a == "LCA" else base[has_culture(base)]
            d = d.reset_index(drop=True)
            hard = targets(d, hard_type, "subject")
            soft = targets(d, "paper_soft", "subject")
            d = d.assign(h_bac=hard[:, 0], h_fung=hard[:, 1], s_bac=soft[:, 0], s_fung=soft[:, 1])
            parts = split_paper(d, seed=seed, group_col="subject")
            part = np.array([""] * len(d), dtype=object)
            for k in ("train", "val"):
                part[parts[k]] = k
            part[parts["cal"]] = "cal"
            part[parts["eval"]] = "eval"
            out[(a, v)] = d.assign(part=part)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="results/labels/paper_labels.csv")
    ap.add_argument("--out", default="results/analysis_sets")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sets = build(a.labels, a.seed)
    S = []

    def save(df, name):
        df.to_csv(os.path.join(a.out, name), index=False)
        return df

    # ---- at a glance ------------------------------------------------------------------------------
    rows = []
    for (an, v), d in sets.items():
        r = {"analysis": an, "version": v}
        for sp in SPLITS:
            r[sp] = ip(d[d.split == sp])
        r["test: calibration"] = ip(d[d.part == "cal"])
        r["test: evaluation"] = ip(d[d.part == "eval"])
        r["all"] = ip(d)
        rows.append(r)
    glance = save(pd.DataFrame(rows), "at_a_glance.csv")
    S.append(("At a glance: images (patients)", md(glance)))

    # ---- by study ------------------------------------------------------------------------------------
    for an in ANALYSES:
        rows = []
        studies = sorted(set(sets[(an, "full")].study))
        for st in studies + ["all"]:
            r = {"study": st}
            for sp in SPLITS:
                for v in VERSIONS:
                    d = sets[(an, v)]
                    d = d if st == "all" else d[d.study == st]
                    r[f"{sp}, {v}"] = ip(d[d.split == sp])
            rows.append(r)
        S.append((f"{an} analysis by study: images (patients)", md(save(pd.DataFrame(rows), f"by_study_{an}.csv"))))

    # ---- hard labels ------------------------------------------------------------------------------------
    rows = []
    for (an, v), d in sets.items():
        for sp in SPLITS + ["all"]:
            g = d if sp == "all" else d[d.split == sp]
            r = {"analysis": an, "version": v, "split": sp}
            for name, (b, f) in COMBOS.items():
                r[name] = ip(g[(g.h_bac == b) & (g.h_fung == f)])
            r["bacterial +"] = f"{np.nanmean(g.h_bac):.1%}" if len(g) else ""
            r["fungal +"] = f"{np.nanmean(g.h_fung):.1%}" if len(g) else ""
            r["a label missing"] = int(np.isnan(g[["h_bac", "h_fung"]].to_numpy()).any(axis=1).sum())
            rows.append(r)
    S.append(("Hard labels: images (patients) per combination [bacterial, fungal]",
              md(save(pd.DataFrame(rows), "hard_labels.csv"))))

    # ---- soft vs hard ----------------------------------------------------------------------------------------
    rows = []
    for (an, v), d in sets.items():
        for sp in SPLITS:
            g = d[d.split == sp]
            r = {"analysis": an, "version": v, "split": sp, "images": len(g)}
            for pth in ("bac", "fung"):
                gap = (g[f"s_{pth}"] - g[f"h_{pth}"]).abs()
                r[f"{pth}: soft and hard differ > 0.1"] = f"{np.mean(gap > 0.1):.1%}" if len(g) else ""
                r[f"{pth}: mean soft, hard = 1"] = g.loc[g[f"h_{pth}"] == 1, f"s_{pth}"].mean()
                r[f"{pth}: mean soft, hard = 0"] = g.loc[g[f"h_{pth}"] == 0, f"s_{pth}"].mean()
            rows.append(r)
    S.append(("Soft vs hard labels (where a soft-vs-hard comparison can show anything)",
              md(save(pd.DataFrame(rows), "soft_vs_hard.csv"))))

    # ---- the evaluation set --------------------------------------------------------------------------------------
    rows = []
    for (an, v), d in sets.items():
        ev = d[d.part == "eval"]
        for scope, g in (("all", ev), ("MADURAI", ev[ev.study == "MADURAI"]), ("other studies", ev[ev.study != "MADURAI"])):
            r = {"analysis": an, "version": v, "evaluation images": scope, "images (patients)": ip(g)}
            for pth, name in (("bac", "bacterial"), ("fung", "fungal")):
                y = g[f"h_{pth}"].dropna()
                r[f"{name}: positives"] = int(y.sum())
                r[f"{name}: AUROC defined"] = "yes" if y.nunique() == 2 else "no"
            rows.append(r)
    S.append(("The evaluation set (80% of test patients; the other 20% calibrate conformal)",
              md(save(pd.DataFrame(rows), "evaluation_set.csv"))))

    # ---- culture vs LCA labels on the culture images ------------------------------------------------------------------
    d = sets[("culture", "full")]
    lca = targets(d, "paper_hard", "subject")
    rows = []
    for j, pth in enumerate(("bacterial", "fungal")):
        c, l_ = d[["h_bac", "h_fung"]].to_numpy()[:, j], lca[:, j]
        m = ~np.isnan(c)
        rows.append({"pathogen": pth, "images": int(m.sum()), "same hard label": f"{np.mean(c[m] == l_[m]):.1%}",
                     "culture +, LCA -": int(((c == 1) & (l_ == 0))[m].sum()),
                     "culture -, LCA +": int(((c == 0) & (l_ == 1))[m].sum())})
    rec = d[["bac_cult_positive", "fung_cult_positive"]].notna()
    rec_tab = pd.DataFrame({"study": d.study, "both cultures": rec.all(axis=1), "one culture": rec.sum(axis=1) == 1})
    rec_tab = rec_tab.groupby("study")[["both cultures", "one culture"]].sum().reset_index()
    S.append(("Culture analysis: cultures vs LCA hard labels (full)",
              md(save(pd.DataFrame(rows), "culture_vs_lca.csv")) + "\n\nCultures recorded:\n\n" + md(rec_tab, ".0f")))

    # ---- what the missing photos take away ------------------------------------------------------------------------------
    rows = []
    for an in ANALYSES:
        f, p = sets[(an, "full")], sets[(an, "provisional")]
        for sp in SPLITS:
            nf, np_ = int((f.split == sp).sum()), int((p.split == sp).sum())
            rows.append({"analysis": an, "split": sp, "full": nf, "provisional": np_, "missing": nf - np_,
                         "share missing": f"{(nf - np_) / nf:.1%}" if nf else "",
                         "missing by study": ", ".join(f"{k} {v}" for k, v in (
                             f[f.split == sp].study.value_counts() - p[p.split == sp].study.value_counts()
                             .reindex(f[f.split == sp].study.unique()).fillna(0)).astype(int).items() if v)})
    S.append(("What the missing photos take away (images)", md(save(pd.DataFrame(rows), "missing.csv"))))

    lines = [f"{an}: {ANALYSES[an][1]}." for an in ANALYSES] + \
            [f"{v}: {VERSIONS[v]}." for v in VERSIONS] + \
            ["Images (patients) throughout. Labels are the same for all of a patient's images."]
    with open(os.path.join(a.out, "analysis_sets.md"), "w") as fh:
        fh.write("# What each analysis gets\n\nAggregates only.\n\n" + "\n".join(f"- {x}" for x in lines) + "\n\n")
        for title, text in S:
            fh.write(f"## {title}\n\n{text}\n\n")
    print(glance.to_string(index=False))
    print(f"\nReport: {a.out}/analysis_sets.md")


if __name__ == "__main__":
    main()
