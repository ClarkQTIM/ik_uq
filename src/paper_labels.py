"""
Build the label table for the redo from the paper's own files (Travis, 23 Oct 2026), and find every
image on disk.

Labels per image:
  paper_soft_bac, paper_soft_fung   the paper's posteriors, as sent. Per pathogen: each comes from that
                                    pathogen's own two tests only (see the EDA), so they need not sum to 1.
  hard_bac, hard_fung               Travis's rule: positive smear or culture for that pathogen -> 1, else 0.
                                    [1, 1] and [0, 0] occur.
  twoclass_bac                      P(bacterial | all four tests) under the same two-class LCA, combined
                                    exactly from the two per-pathogen posteriors (naive Bayes, shared prior):
                                      odds = [b / (1 - b)] * [(1 - f) / f] / prior_odds,
                                    prior_odds = 0.44 / 0.56 (the prior the paper's posteriors were built with).
  y_bac                             twoclass_bac > 0.5 (the corrected, binary reference).

Images: looked up under each --img_roots folder, first by the paper's relative path
(<study folder>/<subfolder>/<file>, case-insensitive), then by study folder + file name, then file
name alone, then a normalised name (letters and digits only), each only when it identifies one file.

Split: the paper's train and test files; --val_csv adds the paper's validation file when we have it.

Writes, in --out:
  paper_labels.csv   one row per paper image with subject, local path and labels.
                     Contains identifiers: stays on the server (results/ is git-ignored).
  labels_report.md   aggregates only: what was found, label distributions, the label variants side by side.

    python src/paper_labels.py --train "training images.csv" --test "test images.csv" --out results/labels \
        [--val_csv "validation images.csv"] [--img_roots /pacs/OSHU/Corneal_Ulcer_Image_Sets /other/root]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eda_paper_data import DEFAULT_OLD_CSV, TESTS, load_paper, md, norm_name, short_study  # noqa: E402
from labels import FOLLOWUP_VISITS, PACS_PREFIX, parse_visit  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
PRIOR_BAC = 0.44  # prior the paper's posteriors were computed with (back-solved in the EDA)


def index_images(roots):
    """All image files under the roots: lists of paths keyed four ways (lower case)."""
    idx = {"rel": {}, "study_name": {}, "name": {}, "norm": {}}
    n = 0
    for root in roots:
        root = root.rstrip("/")
        for dirpath, _, files in os.walk(root):
            for fn in files:
                if os.path.splitext(fn)[1].lower() not in IMG_EXT:
                    continue
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root).replace("\\", "/")
                parts = rel.split("/")
                idx["rel"].setdefault(rel.lower(), []).append(full)
                idx["study_name"].setdefault((parts[0].lower(), fn.lower()), []).append(full)
                idx["name"].setdefault(fn.lower(), []).append(full)
                idx["norm"].setdefault(norm_name(fn), []).append(full)
                n += 1
    return idx, n


def resolve(row, idx):
    keys = [("relative path", "rel", row["relpath"]),
            ("study folder + file name", "study_name", (row["study_full"].lower(), row["basename"].lower())),
            ("file name", "name", row["basename"].lower()),
            ("normalised name", "norm", norm_name(row["basename"]))]
    for label, kind, key in keys:
        hits = idx[kind].get(key, [])
        if len(set(hits)) == 1:
            return hits[0], label
        if len(set(hits)) > 1 and kind == "rel":
            return sorted(hits)[0], label
    return "", "not found"


def twoclass(b, f, prior_bac=PRIOR_BAC):
    b = np.clip(b, 1e-9, 1 - 1e-9)
    f = np.clip(f, 1e-9, 1 - 1e-9)
    odds = (b / (1 - b)) * ((1 - f) / f) / (prior_bac / (1 - prior_bac))
    return odds / (1 + odds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--val_csv", default=None, help="the paper's validation file, if we get it")
    ap.add_argument("--img_roots", nargs="+", default=[PACS_PREFIX])
    ap.add_argument("--prior_bac", type=float, default=PRIOR_BAC)
    ap.add_argument("--old_csv", default=DEFAULT_OLD_CSV,
                    help="our study CSV: are the subjects of images without a file in it? ('none' to skip)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    notes = []
    parts = [load_paper(a.train, "train", notes)[0], load_paper(a.test, "test", notes)[0]]
    if a.val_csv:
        parts.append(load_paper(a.val_csv, "val", notes)[0])
    d = pd.concat(parts, ignore_index=True)
    dup = int(d.duplicated("Drive_image_path").sum())
    if dup:
        notes.append(f"{dup} image paths appear more than once across the files")
    leak = d.groupby("subject")["split"].nunique()
    if (leak > 1).any():
        notes.append(f"{int((leak > 1).sum())} subjects appear in more than one split")

    # ---- images --------------------------------------------------------------------------------
    print(f"Indexing images under {', '.join(a.img_roots)} ...")
    idx, n_files = index_images(a.img_roots)
    print(f"  {n_files:,} image files indexed.")
    res = d.apply(lambda r: resolve(r, idx), axis=1)
    d["local_path"] = res.map(lambda x: x[0])
    d["found_by"] = res.map(lambda x: x[1])
    d["has_image"] = d["local_path"] != ""
    reused = int(d.loc[d.has_image, "local_path"].duplicated().sum())
    if reused:
        notes.append(f"{reused} paper images resolve to a file already used by another paper image")

    # ---- labels ---------------------------------------------------------------------------------
    d["paper_soft_bac"], d["paper_soft_fung"] = d["bac_post_prob"], d["fung_post_prob"]
    d["hard_bac"] = ((d.bac_smear == 1) | (d.bac_cult_positive == 1)).astype(int)
    d["hard_fung"] = ((d.fung_smear == 1) | (d.fung_cult_positive == 1)).astype(int)
    d["twoclass_bac"] = twoclass(d.bac_post_prob.to_numpy(), d.fung_post_prob.to_numpy(), a.prior_bac)
    d["y_bac"] = (d["twoclass_bac"] > 0.5).astype(int)
    d["no_posterior"] = d[["bac_post_prob", "fung_post_prob"]].isna().any(axis=1)
    if d["no_posterior"].any():
        notes.append(f"{int(d.no_posterior.sum())} images without both posteriors (labels undefined)")
    d["pattern"] = d[TESTS].apply(lambda r: " ".join(
        f"{t}{'.' if np.isnan(r[c]) else ('+' if r[c] == 1 else '-')}"
        for t, c in zip(("Gram", "KOH", "BA", "PDA"), TESTS)), axis=1)

    cols = ["split", "subject", "study", "subfolder", "visit", "Drive_image_path", "local_path", "found_by",
            "has_image"] + TESTS + ["paper_soft_bac", "paper_soft_fung", "hard_bac", "hard_fung",
                                    "twoclass_bac", "y_bac", "pattern"]
    d[cols].to_csv(os.path.join(a.out, "paper_labels.csv"), index=False)

    # ---- report (aggregates only) --------------------------------------------------------------------
    found = (d.groupby(["study", "split"]).agg(images=("subject", "size"), found=("has_image", "sum"))
             .reset_index())
    found["missing"] = found["images"] - found["found"]
    how = pd.crosstab(d.study, d.found_by).reset_index()
    pat = (d.groupby("pattern").agg(images=("subject", "size"), studies=("study", lambda x: ", ".join(sorted(set(x)))),
                                    paper_soft_bac=("paper_soft_bac", "first"), paper_soft_fung=("paper_soft_fung", "first"),
                                    hard=("hard_bac", lambda x: ""), twoclass_bac=("twoclass_bac", "first"))
           .reset_index())
    pat["hard [bac, fung]"] = [f"[{int(d.loc[d.pattern == p, 'hard_bac'].iloc[0])}, "
                               f"{int(d.loc[d.pattern == p, 'hard_fung'].iloc[0])}]" for p in pat.pattern]
    pat["binary y_bac"] = (pat["twoclass_bac"] > 0.5).astype(int)
    pat = pat.drop(columns="hard").sort_values("images", ascending=False)
    combo = pd.crosstab(d.hard_bac.astype(str) + "," + d.hard_fung.astype(str), d.y_bac.map({1: "binary bacterial", 0: "binary fungal"}))
    combo = combo.reset_index().rename(columns={"row_0": "hard [bac,fung]"})
    combo.columns = ["hard [bac, fung]"] + list(combo.columns[1:])
    usable = d[d.has_image & ~d.no_posterior]
    # images without a file: are their subjects in our CSV, and with how many photos?
    subj_tab = pd.DataFrame()
    if (~d.has_image).any() and a.old_csv.lower() != "none" and os.path.exists(a.old_csv):
        old = pd.read_csv(a.old_csv)
        old["subject"] = old["subject"].astype(str).str.strip()
        old["study_s"] = old["study"].astype(str).map(short_study)
        nm = old["image_name"] if "image_name" in old else old["file_path"].astype(str).map(os.path.basename)
        old["visit"] = [parse_visit(st, n) for st, n in zip(old["study"].astype(str), nm.astype(str))]
        old["enrol"] = ~old["visit"].isin(FOLLOWUP_VISITS)
        rows = []
        for stdy, g in d[~d.has_image].groupby("study"):
            subs = set(g.subject)
            o = old[old.subject.isin(subs)]
            paper_all = d[d.subject.isin(subs)]
            rows.append({"study": stdy, "images without a file": len(g), "their subjects": len(subs),
                         "subjects in our CSV": int(o.subject.nunique()),
                         "paper images for these subjects": len(paper_all),
                         "our CSV images, enrolment": int(o.enrol.sum()),
                         "our CSV images, follow-up": int((~o.enrol).sum())})
        subj_tab = pd.DataFrame(rows)
    split_tab = (usable.groupby(["split", "study"]).agg(images=("subject", "size"), subjects=("subject", "nunique"),
                                                         bacterial_binary=("y_bac", "sum")).reset_index())
    lines = [f"Images: {len(d):,} in the paper's files ({', '.join(f'{s} {n:,}' for s, n in d.split.value_counts().items())}); "
             f"files found for {int(d.has_image.sum()):,}; usable (file and labels) {len(usable):,}.",
             f"Indexed {n_files:,} image files under: {', '.join(a.img_roots)}.",
             f"Two-class posterior prior: {a.prior_bac} bacterial."]
    with open(os.path.join(a.out, "labels_report.md"), "w") as f:
        f.write("# Labels for the redo, from the paper's files\n\nAggregates only.\n\n"
                + "\n".join(f"- {x}" for x in lines) + "\n\n## Things to look at\n\n"
                + ("\n".join(f"- {x}" for x in notes) if notes else "- none flagged") + "\n\n"
                + "## Image files by study and split\n\n" + md(found, ".0f") + "\n\nHow they were found:\n\n" + md(how, ".0f")
                + "\n\n## Labels by test pattern (Gram KOH BA PDA)\n\n" + md(pat)
                + "\n\n## Travis's hard labels against the binary two-class label\n\n" + md(combo, ".0f")
                + ("\n\n## Images without a file: their subjects in our CSV\n\n" + md(subj_tab, ".0f") if len(subj_tab) else "")
                + "\n\n## Usable images by split (file found, labels defined)\n\n" + md(split_tab, ".0f") + "\n")
    print("\n".join(lines))
    print("Missing files by study and split: " + (", ".join(
        f"{r.study} {r.split} {r.missing}" for r in found.itertuples() if r.missing) or "none"))
    if notes:
        print("Things to look at:\n" + "\n".join(f"  - {x}" for x in notes))
    print(f"\nWritten: {a.out}/paper_labels.csv (stays on the server), {a.out}/labels_report.md (send this)")


if __name__ == "__main__":
    main()
