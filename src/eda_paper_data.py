"""
EDA of the original paper's data (Travis, 23 Oct 2026): "training images.csv" and "test images.csv",
plus "validation images.csv" with --val (27 Oct 2026).

Columns: subject, Drive_image_path, fung_smear, bac_cult_positive, fung_cult_positive, bac_smear,
fung_post_prob, bac_post_prob.
Travis's hard-label rule: bacterial = 1 if bacterial smear or bacterial culture is positive, else 0;
the same for fungal. (So the hard labels are per pathogen: [1, 1] and [0, 0] can occur.)

What this answers (see the "Answers" section of the report):
  * Which images and participants the paper used (vs 4,471 / 2,100), with studies and visits.
  * Whether the split is grouped by participant (subjects, and pIDs if the old CSV is given).
  * Whether trial participants carry test results (MUTT smears, SCUT cultures).
  * Whether the posteriors are two-class (sum to 1, depend on all four tests) or per pathogen
    (depend on that pathogen's own tests), and what each test pattern gets, against a naive
    recomputation from the supplement's LCA estimates (e.g. all tests negative).
  * What Travis's hard rule gives: how many [1,0], [0,1], [1,1], [0,0], and for which images.
  * Co-positivity vs independence, by study.
  * Where soft and hard targets differ (where a soft-vs-hard comparison has any power).
  * With --old_csv: how the paper's images map onto our CSV (tests, posteriors, bins, visits,
    subject vs pID, image files on disk).

Prints and writes aggregates only: no subject IDs, paths or image names.

    python src/eda_paper_data.py --train "training images.csv" --test "test images.csv" \
        [--val "validation images.csv"] --out results/eda [--old_csv /persist/.../corneal_ulcer_study_data_7_24_2025_lambda.csv]
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import FOLLOWUP_VISITS, PACS_PREFIX, STUDY_SHORT, parse_visit  # noqa: E402
from independence_check import expected_both_one_pathogen, two_by_two  # noqa: E402

DEFAULT_OLD_CSV = ("/persist/QTIM/Active/26-0096/Corneal_Ulcer_Image_Sets/Data/"
                   "corneal_ulcer_study_data_7_24_2025_lambda.csv")
DEFAULT_IMG_ROOT = PACS_PREFIX
TESTS = ["bac_smear", "fung_smear", "bac_cult_positive", "fung_cult_positive"]
TEST_SHORT = {"bac_smear": "Gram", "fung_smear": "KOH", "bac_cult_positive": "BA", "fung_cult_positive": "PDA"}
POSTS = ["bac_post_prob", "fung_post_prob"]
EXPECTED = ["subject", "Drive_image_path"] + TESTS + POSTS
# Supplement LCA estimates (sensitivity, specificity); fungal prevalence 0.56.
SENS_SPEC = {"bac_smear": (0.40, 0.97), "fung_smear": (0.94, 0.94),
             "bac_cult_positive": (0.37, 0.96), "fung_cult_positive": (0.72, 0.98)}
PI_FUNGAL = 0.56
PAPER_IMAGES, PAPER_PARTICIPANTS = 4471, 2100


# =============================================================================
# Helpers
# =============================================================================

def md(df, floatfmt=".3f"):
    if df is None or len(df) == 0:
        return "_(none)_"
    df = df.astype(object)  # keep integer columns as integers when every column is numeric
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt)
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```"


def naive_bac(row, tests=TESTS):
    """Two-class naive-Bayes P(bacterial | observed tests), supplement estimates; NaN tests skipped."""
    lb, lf = 1 - PI_FUNGAL, PI_FUNGAL
    for t in tests:
        x = row[t]
        if np.isnan(x):
            continue
        se, sp = SENS_SPEC[t]
        pb, pf = (se, 1 - sp) if t.startswith("bac") else (1 - sp, se)
        lb *= pb if x == 1 else 1 - pb
        lf *= pf if x == 1 else 1 - pf
    return lb / (lb + lf)


def pattern(row):
    return " ".join(f"{TEST_SHORT[t]}{'.' if np.isnan(row[t]) else ('+' if row[t] == 1 else '-')}" for t in TESTS)


SITES = ["madurai", "coimbatore", "pondicherry", "tirunelveli", "lumbini", "bharatpur", "bharaptur",
         "dartmouth", "proctor"]


def find_study(parts):
    """(study folder, its index): the shallowest folder that is a known study folder, else the shallowest
    ending in 'photos' / 'pictures'. Shallowest first, because trial folders have site subfolders
    (e.g. MUTT1-Pictures/Madurai/...) that must not be read as the MADURAI study."""
    known = {k.lower(): k for k in STUDY_SHORT}
    for i, p in enumerate(parts[:-1]):
        if p.strip().lower() in known:
            return known[p.strip().lower()], i
    for i, p in enumerate(parts[:-1]):
        if re.search(r"[ _-](photos|pictures)$", p.strip().lower()):
            return p.strip(), i
    return "unknown", -1


def short_study(full):
    return STUDY_SHORT.get(full, re.sub(r"[ _-](photos|pictures)$", "", full, flags=re.I))


def name_shape(name):
    """File-name format with digit runs replaced by '#' (no identifiers)."""
    return re.sub(r"\d+", "#", str(name))


def norm_name(name):
    return re.sub(r"[^a-z0-9]", "", os.path.splitext(str(name).lower())[0])


def load_paper(path, split, notes):
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    missing = [c for c in EXPECTED if c not in raw.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}; found {list(raw.columns)}")
    extra = [c for c in raw.columns if c not in EXPECTED]
    if extra:
        notes.append(f"{split}: extra columns {extra}")
    df = raw.copy()
    df["split"] = split
    rawvals = {c: raw[c].astype(str).str.strip().value_counts(dropna=False).head(8).to_dict() for c in TESTS + POSTS}
    for c in TESTS + POSTS:
        text = raw[c].notna() & pd.to_numeric(raw[c], errors="coerce").isna()
        if text.any():
            notes.append(f"{split}: {c} has {int(text.sum())} non-numeric values (treated as missing; see raw values)")
    for c in TESTS:
        v = pd.to_numeric(raw[c], errors="coerce")
        odd = v.notna() & ~v.isin([0, 1, -1])
        if odd.any():
            notes.append(f"{split}: {c} has {int(odd.sum())} values other than 0, 1, -1 (treated as missing)")
        df[c] = v.where(v.isin([0, 1])).astype(float)
    for c in POSTS:
        v = pd.to_numeric(raw[c], errors="coerce")
        bad = v.notna() & (v != -1) & ~v.between(0, 1)
        if bad.any():
            notes.append(f"{split}: {c} has {int(bad.sum())} values outside [0, 1] other than -1 (treated as missing)")
        df[c] = v.where(v.between(0, 1)).astype(float)
    p = df["Drive_image_path"].astype(str).str.replace("\\", "/", regex=False)
    df["_parts"] = p.str.split("/")
    df["basename"] = df["_parts"].map(lambda x: x[-1].strip())
    found = df["_parts"].map(find_study)
    df["study_full"] = found.map(lambda x: x[0])
    idx = found.map(lambda x: x[1])
    df["study"] = df["study_full"].map(short_study)
    df["subfolder"] = [pp[i + 1].strip() if 0 <= i and i + 1 < len(pp) - 1 else "" for pp, i in zip(df["_parts"], idx)]
    df["relpath_orig"] = ["/".join(p.strip() for p in pp[i:]) if i >= 0 else "" for pp, i in zip(df["_parts"], idx)]
    df["relpath"] = df["relpath_orig"].str.lower()
    df["visit"] = [parse_visit(s, n) if s in STUDY_SHORT else "not encoded"
                   for s, n in zip(df["study_full"], df["basename"])]
    df["is_followup"] = df["visit"].isin(FOLLOWUP_VISITS)
    df["subject"] = df["subject"].astype(str).str.strip()
    return df, rawvals


def subject_tests(df):
    """Per-subject test results (max over images, missing ignored) and conflicts."""
    g = df.groupby("subject")
    res = g[TESTS].max()
    lo = g[TESTS].min()
    conflicts = {t: int(((lo[t] == 0) & (res[t] == 1)).sum()) for t in TESTS}
    return res, conflicts


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--val", default=None, help="the paper's validation file, if we have it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--old_csv", default=DEFAULT_OLD_CSV, help="our study CSV; 'none' to skip the comparison")
    ap.add_argument("--img_root", default=DEFAULT_IMG_ROOT, help="where the old CSV's image paths live")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    notes, answers, S = [], [], []  # S: (heading, text) report sections

    def section(title, *blocks):
        S.append((title, "\n\n".join(b for b in blocks if b)))

    def save(df, name):
        df.to_csv(os.path.join(a.out, name), index=False)
        return df

    tr, rv_tr = load_paper(a.train, "train", notes)
    te, rv_te = load_paper(a.test, "test", notes)
    loaded = [("train", tr, rv_tr), ("test", te, rv_te)]
    if a.val:
        va, rv_va = load_paper(a.val, "val", notes)
        loaded.insert(1, ("val", va, rv_va))
    SPLITS = [x[0] for x in loaded]
    d = pd.concat([x[1] for x in loaded], ignore_index=True)

    # ---- 0. Files and raw values -----------------------------------------------------
    files = pd.DataFrame([{"split": s, "rows": int((d.split == s).sum()), "subjects": int(d[d.split == s].subject.nunique())}
                          for s in SPLITS] + [{"split": "all", "rows": len(d), "subjects": int(d.subject.nunique())}])
    files["image share"] = files["rows"] / len(d)
    raw = pd.DataFrame([{"column": c, "split": s, "raw values (top 8, with counts)": "; ".join(f"{k}: {v}" for k, v in rv[c].items())}
                        for s, _, rv in loaded for c in TESTS + POSTS])
    section("0. Files", md(save(files, "files.csv")),
            f"Paper: {PAPER_IMAGES:,} images, {PAPER_PARTICIPANTS:,} participants. Here: {len(d):,} images, "
            f"{d.subject.nunique():,} subjects.", "Raw values as read (before coding; numbers only, no identifiers):\n\n" + md(raw))

    # ---- 1. Paths, studies, visits, duplicates ----------------------------------------------
    depth = d["_parts"].map(len).value_counts().sort_index()
    lines = ["Path depth (components): " + ", ".join(f"{k}: {v}" for k, v in depth.items())]
    for k in range(int(depth.index.max()) - 1):
        vals = d["_parts"].map(lambda x: x[k] if len(x) > k + 1 else None).dropna()
        vc = vals.value_counts()
        if len(vc) <= 40 and vc.min() >= 10:
            lines.append(f"Folder level {k}: " + ", ".join(f"'{x}' ({n})" for x, n in vc.items()))
        else:
            lines.append(f"Folder level {k}: {len(vc)} distinct values (not shown)")
    unknown = int((d.study == "unknown").sum())
    if unknown:
        notes.append(f"{unknown} images whose study could not be read from the path")
    st = (d.groupby(["study", "split"]).agg(images=("subject", "size"), subjects=("subject", "nunique"),
                                             followups=("is_followup", "sum")).reset_index())
    vis = pd.crosstab([d.study, d.visit], d.split).reset_index()
    dup_path = int(d.duplicated(["Drive_image_path"]).sum())
    dup_within = {s: int(d[d.split == s].duplicated(["Drive_image_path"]).sum()) for s in SPLITS}
    within_txt = ", ".join(f"within {s} {n}" for s, n in dup_within.items())
    cross_path = int((d.groupby("Drive_image_path")["split"].nunique() > 1).sum())
    cross_base = int((d.assign(_b=d["basename"].str.lower()).groupby("_b")["split"].nunique() > 1).sum())
    if dup_path:
        notes.append(f"{dup_path} duplicated image paths ({within_txt}, across splits {cross_path})")
    sub_rows = []
    for stdy, g in d.groupby("study"):
        sf = g["subfolder"]
        site = sf.str.lower().isin(SITES)
        sub_rows.append({"study": stdy, "images": len(g), "in a subfolder": int((sf != "").sum()),
                         "distinct subfolders": int(sf[sf != ""].nunique()),
                         "site-named subfolders": ", ".join(f"{k} {v}" for k, v in sf[site].value_counts().items()),
                         "other subfolders (not shown)": int(sf[(sf != "") & ~site].nunique())})
    section("1. Paths, studies and visits", "\n".join(f"- {x}" for x in lines),
            "Study = the study folder (shallowest folder named after a study); subfolders below it:\n\n"
            + md(save(pd.DataFrame(sub_rows), "subfolders_by_study.csv"), ".0f"),
            md(save(st, "studies_by_split.csv")), "Visits (parsed from image names):\n\n" + md(save(vis, "visits_by_study.csv")),
            f"Duplicated paths: {dup_path} ({within_txt}, same path in more than one split {cross_path}); "
            f"same file name in more than one split: {cross_base}.")

    # ---- 2. Subjects and the split ----------------------------------------------------------
    per_sub = d.groupby(["split", "subject"]).size()
    ips = per_sub.groupby(level=0).describe()[["count", "mean", "50%", "max"]].reset_index()
    subj_split = d.groupby("subject")["split"].nunique()
    both = set(subj_split[subj_split > 1].index)
    pair_txt = "; ".join(f"{x} and {y}: {len(set(d[d.split == x].subject) & set(d[d.split == y].subject))}"
                         for i, x in enumerate(SPLITS) for y in SPLITS[i + 1:])
    multi_study = int((d.groupby("subject")["study"].nunique() > 1).sum())
    if both:
        notes.append(f"{len(both)} subjects have images in more than one split ({pair_txt})")
    section("2. Subjects and the split",
            "Images per subject:\n\n" + md(ips.rename(columns={"count": "subjects", "50%": "median"})
                                             .astype({"subjects": int, "median": int, "max": int})),
            f"Subjects in more than one split: {len(both)} ({int(d[d.subject.isin(both)].shape[0])} images; {pair_txt}). "
            f"Subjects appearing in more than one study: {multi_study}.")

    # ---- 3. Tests ---------------------------------------------------------------------------
    rows = []
    for (s, sp), g in d.groupby(["study", "split"]):
        r = {"study": s, "split": sp, "images": len(g)}
        for t in TESTS:
            r[f"{TEST_SHORT[t]} +/-/missing"] = f"{int((g[t] == 1).sum())}/{int((g[t] == 0).sum())}/{int(g[t].isna().sum())}"
        r["no test at all"] = int(g[TESTS].isna().all(axis=1).sum())
        rows.append(r)
    tests_tab = save(pd.DataFrame(rows), "tests_by_study.csv")
    miss = d[TESTS].notna().apply(lambda r: "".join(TEST_SHORT[t][0] if r[t] else "." for t in TESTS), axis=1)
    miss_tab = (pd.crosstab(miss, d.study).reset_index().rename(columns={"row_0": "recorded (G K B P)", 0: "recorded"}))
    miss_tab.columns = ["recorded tests (Gram KOH BA PDA)"] + list(miss_tab.columns[1:])
    sub_t, conflicts = subject_tests(d)
    section("3. Tests (image level)", md(tests_tab), "Which tests are recorded together (images):\n\n" + md(save(miss_tab, "test_missingness.csv")),
            "Subjects whose images disagree on a test: " + ", ".join(f"{TEST_SHORT[t]} {n}" for t, n in conflicts.items()))

    # ---- 4. Posteriors ----------------------------------------------------------------------------
    hp = d[POSTS].notna().all(axis=1)
    one_only = int((d[POSTS].notna().sum(axis=1) == 1).sum())
    ssum = (d["bac_post_prob"] + d["fung_post_prob"])[hp]
    dev = (ssum - 1).abs()
    sum_tab = pd.DataFrame([{"images with both posteriors": int(hp.sum()), "with one only": one_only,
                             "with none": int((~d[POSTS].notna().any(axis=1)).sum()),
                             "sum within 0.001 of 1": float((dev <= 1e-3).mean()) if hp.any() else np.nan,
                             "median abs(sum - 1)": float(dev.median()) if hp.any() else np.nan,
                             "max abs(sum - 1)": float(dev.max()) if hp.any() else np.nan}])
    nop = d[~hp].groupby(["study", "split"]).size().reset_index(name="images without both posteriors")
    edges = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0001]
    labs = ["<0.01", "0.01-0.1", "0.1-0.25", "0.25-0.5", "0.5-0.75", "0.75-0.9", "0.9-0.99", ">=0.99"]
    hist = pd.DataFrame({c: pd.cut(d[c], edges, right=False, labels=labs).value_counts().reindex(labs) for c in POSTS}).reset_index()
    hist = hist.rename(columns={"index": "posterior"})
    amb = d.assign(amb_bac=d.bac_post_prob.between(0.1, 0.9, inclusive="neither"),
                   amb_fung=d.fung_post_prob.between(0.1, 0.9, inclusive="neither"))
    by_study_post = amb.groupby(["study", "split"]).agg(
        images=("subject", "size"), with_posteriors=("bac_post_prob", lambda x: int(x.notna().sum())),
        mean_bac_post=("bac_post_prob", "mean"), mean_fung_post=("fung_post_prob", "mean"),
        ambiguous_bac=("amb_bac", "sum"), ambiguous_fung=("amb_fung", "sum")).reset_index()

    # two-class or per pathogen? within-group spread of each posterior grouped by own tests vs all four
    def within_sd(col, keys):
        sub = d[d[col].notna()].copy()
        for k in keys:
            sub[k] = sub[k].fillna(-1)
        g = sub.groupby(keys)[col]
        sd = g.std(ddof=0).fillna(0)
        n = g.size()
        return float(np.average(sd, weights=n)), float(sd.max())
    diag = []
    for col, own in (("bac_post_prob", ["bac_smear", "bac_cult_positive"]), ("fung_post_prob", ["fung_smear", "fung_cult_positive"])):
        w_own, m_own = within_sd(col, own)
        w_all, m_all = within_sd(col, TESTS)
        diag.append({"posterior": col, "mean within-pattern SD, own two tests": w_own, "max, own": m_own,
                     "mean within-pattern SD, all four tests": w_all, "max, all four": m_all})
    diag = pd.DataFrame(diag)

    d["pattern"] = d.apply(pattern, axis=1)
    d["naive_bac_all4"] = d.apply(naive_bac, axis=1)
    d["naive_bac_own"] = d.apply(lambda r: naive_bac(r, ["bac_smear", "bac_cult_positive"]), axis=1)
    pat = (d.groupby("pattern").agg(images=("subject", "size"), subjects=("subject", "nunique"),
                                    studies=("study", lambda x: ", ".join(sorted(set(x)))),
                                    bac_post_mean=("bac_post_prob", "mean"), bac_post_min=("bac_post_prob", "min"),
                                    bac_post_max=("bac_post_prob", "max"), fung_post_mean=("fung_post_prob", "mean"),
                                    naive_two_class_all4=("naive_bac_all4", "first"),
                                    naive_two_class_own_tests=("naive_bac_own", "first"))
           .reset_index().sort_values("images", ascending=False))
    def own_table(col, own, prob_of):
        sub = d.assign(_p=d[own].apply(lambda r: " ".join(
            f"{TEST_SHORT[t]}{'.' if np.isnan(r[t]) else ('+' if r[t] == 1 else '-')}" for t in own), axis=1))
        sub["_naive"] = sub.apply(lambda r: naive_bac(r, own), axis=1)
        if prob_of == "fung":
            sub["_naive"] = 1 - sub["_naive"]
        t = (sub.groupby("_p").agg(images=("subject", "size"), subjects=("subject", "nunique"),
                                   studies=("study", lambda x: ", ".join(sorted(set(x)))),
                                   distinct_values=(col, "nunique"), paper=(col, "mean"),
                                   recomputed=("_naive", "first")).reset_index()
             .rename(columns={"_p": "own tests", "paper": f"paper {col}",
                              "recomputed": "supplement LCA, own tests only"}))
        t["difference"] = t[f"paper {col}"] - t["supplement LCA, own tests only"]
        return t.sort_values("images", ascending=False)
    own_b = own_table("bac_post_prob", ["bac_smear", "bac_cult_positive"], "bac")
    own_f = own_table("fung_post_prob", ["fung_smear", "fung_cult_positive"], "fung")
    max_own_diff = float(np.nanmax(np.abs(np.r_[own_b["difference"], own_f["difference"]])))
    section("4. Posteriors",
            md(save(sum_tab, "posterior_sums.csv")),
            "Images without both posteriors, by study:\n\n" + md(nop),
            "Distribution:\n\n" + md(save(hist, "posterior_hist.csv"), ".0f"),
            "By study (ambiguous = between 0.1 and 0.9):\n\n" + md(save(by_study_post, "posteriors_by_study.csv")),
            "**Two-class or per pathogen?** If a posterior depends only on its own pathogen's tests, the spread "
            "within groups of the same own-test pattern is 0 (and so is the all-four spread); if it depends on "
            "all four tests, only the all-four spread is 0.\n\n" + md(save(diag, "posterior_dependence.csv"), ".4f"),
            "**Each posterior by its own tests**, against the supplement's two-class LCA (44% bacterial; "
            "Gram 40/97, BA 37/96, KOH 94/94, PDA 72/98 sensitivity/specificity) applied to those two tests "
            f"only. Largest difference: {max_own_diff:.3f}.\n\n"
            + md(save(own_b, "bac_posterior_by_own_tests.csv")) + "\n\n" + md(save(own_f, "fung_posterior_by_own_tests.csv")),
            "**By test pattern** (`+` positive, `-` negative, `.` missing; Gram KOH BA PDA). Naive columns: "
            "P(bacterial) recomputed from the supplement's estimates assuming tests independent given the "
            "class, from all four tests or from the two bacterial tests only.\n\n" + md(save(pat, "posterior_by_pattern.csv")))

    # ---- 5. Travis's hard labels ---------------------------------------------------------------------
    d["hard_bac"] = ((d.bac_smear == 1) | (d.bac_cult_positive == 1)).astype(int)
    d["hard_fung"] = ((d.fung_smear == 1) | (d.fung_cult_positive == 1)).astype(int)
    d["hard_combo"] = d.hard_bac.astype(str) + "," + d.hard_fung.astype(str)
    d["no_tests"] = d[TESTS].isna().all(axis=1)
    combo = pd.crosstab([d.study, d.split], d.hard_combo).reset_index()
    combo_note = pd.crosstab(d.hard_combo, d.no_tests).reset_index().rename(columns={False: "some test recorded", True: "no test recorded"})
    agree = []
    for c, h in (("bac_post_prob", "hard_bac"), ("fung_post_prob", "hard_fung")):
        m = d[c].notna()
        agree.append({"pathogen": c.split("_")[0], "images with posterior": int(m.sum()),
                      "hard = (posterior > 0.5)": float(((d[c] > 0.5).astype(int) == d[h])[m].mean()) if m.any() else np.nan,
                      "mean abs(posterior - hard)": float((d[c] - d[h]).abs()[m].mean()) if m.any() else np.nan,
                      "share abs(posterior - hard) > 0.1": float(((d[c] - d[h]).abs() > 0.1)[m].mean()) if m.any() else np.nan})
    section("5. Travis's hard labels (bacterial = smear or culture +; fungal likewise)",
            "Combinations [bacterial, fungal] by study:\n\n" + md(save(combo, "hard_labels_by_study.csv"), ".0f"),
            "Combinations vs whether any test was recorded (missing tests count as not positive):\n\n" + md(combo_note, ".0f"),
            "Agreement with the posteriors:\n\n" + md(save(pd.DataFrame(agree), "hard_vs_posterior.csv")))

    # ---- 6. Co-positivity and independence (subject level) ----------------------------------------
    st_sub = d.groupby("subject")["study"].first()
    rows = []
    sub_any_b = ((sub_t.bac_smear == 1) | (sub_t.bac_cult_positive == 1)).astype(float).where(sub_t.notna().any(axis=1))
    sub_any_f = ((sub_t.fung_smear == 1) | (sub_t.fung_cult_positive == 1)).astype(float).where(sub_t.notna().any(axis=1))
    sub_t2 = sub_t.assign(any_b=sub_any_b, any_f=sub_any_f, study=st_sub.reindex(sub_t.index))
    for scope in ["All"] + sorted(sub_t2.study.dropna().unique()):
        g = sub_t2 if scope == "All" else sub_t2[sub_t2.study == scope]
        for name, tb, tf in (("cultures (BA, PDA)", "bac_cult_positive", "fung_cult_positive"),
                             ("smears (Gram, KOH)", "bac_smear", "fung_smear"), ("any bacterial + vs any fungal +", "any_b", "any_f")):
            sub = g[g[tb].notna() & g[tf].notna()]
            if len(sub) == 0:
                continue
            r = {"scope": scope, "tests": name}
            r.update(two_by_two(sub[tb].to_numpy(), sub[tf].to_numpy()))
            if tb in SENS_SPEC and "bacterial + rate" in r:
                e = expected_both_one_pathogen(r["participants"], r["bacterial + rate"], r["fungal + rate"], tb, tf)
                r["both + expected from test error (min-max)"] = f"{min(e.values()):.1f}-{max(e.values()):.1f}"
            rows.append(r)
    indep = save(pd.DataFrame(rows), "copositivity.csv")
    keep = ["scope", "tests", "participants", "both +", "bacterial only +", "fungal only +", "neither +",
            "both + expected if independent", "both + expected from test error (min-max)", "odds ratio", "OR ci_low", "OR ci_high"]
    section("6. Co-positivity and independence (subjects)", md(indep[[c for c in keep if c in indep.columns]]))

    # ---- 7. Where soft and hard targets differ ---------------------------------------------------
    diff = d.assign(differs=((d.bac_post_prob - d.hard_bac).abs() > 0.1) | ((d.fung_post_prob - d.hard_fung).abs() > 0.1))
    dt = diff.groupby(["study", "split"]).agg(images=("subject", "size"), differ=("differs", "sum"),
                                              subjects_differ=("subject", lambda x: int(diff.loc[x.index][diff.loc[x.index, "differs"]].subject.nunique())))
    dt = dt.reset_index()
    dt["share"] = dt["differ"] / dt["images"]
    section("7. Where soft (posterior) and hard (Travis) targets differ by more than 0.1",
            md(save(dt, "soft_vs_hard_targets.csv")))

    # ---- 8. Against our CSV ---------------------------------------------------------------------------
    if a.old_csv.lower() != "none" and os.path.exists(a.old_csv):
        old = pd.read_csv(a.old_csv)
        old["basename"] = old["file_path"].astype(str).map(lambda p: os.path.basename(p).strip()) \
            if "file_path" in old else old["image_name"].astype(str)
        if "image_name" in old:
            old["basename"] = old["basename"].where(old["file_path"].notna(), old["image_name"].astype(str))
        old["study_s"] = old["study"].astype(str).map(short_study)
        old["relpath"] = old["file_path"].astype(str).str.replace(PACS_PREFIX + "/", "", regex=False).str.strip().str.lower() \
            if "file_path" in old else ""
        old["k2"] = old["study"].astype(str).str.lower() + "|" + old["basename"].str.lower()
        old["k1"] = old["basename"].str.lower()
        old["k0s"] = old["study"].astype(str).str.lower() + "|" + old["basename"].map(norm_name)
        old["k0"] = old["basename"].map(norm_name)
        d["k2"] = d["study_full"].str.lower() + "|" + d["basename"].str.lower()
        d["k1"] = d["basename"].str.lower()
        d["k0s"] = d["study_full"].str.lower() + "|" + d["basename"].map(norm_name)
        d["k0"] = d["basename"].map(norm_name)
        methods = [("relative path", "relpath"), ("study + file name", "k2"), ("file name", "k1"),
                   ("study + normalised name", "k0s"), ("normalised name", "k0")]
        maps = {}
        for _, k in methods:  # only keys that identify one row of our CSV
            vc = old[k].value_counts()
            ok = old[old[k].isin(vc[vc == 1].index) & (old[k] != "")]
            maps[k] = dict(zip(ok[k], ok.index))
        hit, how_l = [], []
        for _, r in d.iterrows():
            for label, k in methods:
                if r[k] in maps[k]:
                    hit.append(maps[k][r[k]])
                    how_l.append(label)
                    break
            else:
                hit.append(None)
                how_l.append(None)
        how = pd.Series(how_l)
        matched = how.notna().to_numpy()
        o = old.reindex([h if h is not None else -1 for h in hit]).reset_index(drop=True)
        n_dupe = int(pd.Series([h for h in hit if h is not None]).duplicated().sum())
        lines = [f"Our CSV: {len(old):,} rows. Paper images matched to a row of ours: {int(matched.sum()):,} of "
                 f"{len(d):,} ({matched.mean():.1%}). By method: "
                 + ", ".join(f"{lab} {int((how == lab).sum()):,}" for lab, _ in methods)
                 + f". Rows of ours matched by more than one paper image: {n_dupe}."]
        if (~matched).any():
            lines.append("Unmatched paper images by study: " + ", ".join(
                f"{s} {n}" for s, n in d[~matched].study.value_counts().items()))
        # file-name formats where matching failed (digit runs replaced by #; formats with >= 5 images)
        old_hit = set(h for h in hit if h is not None)
        shape_rows = []
        for stdy in sorted(d[~matched].study.unique()):
            pu = d[~matched & (d.study == stdy)]["basename"].map(name_shape).value_counts()
            ou = old[(old.study_s == stdy) & ~old.index.isin(old_hit)]["basename"].map(name_shape).value_counts()
            shape_rows.append({"study": stdy, "unmatched paper images": int(pu.sum()),
                               "their name formats": "; ".join(f"{k} ({v})" for k, v in pu[pu >= 5].head(4).items()),
                               "our unmatched rows, same study": int(ou.sum()),
                               "our name formats": "; ".join(f"{k} ({v})" for k, v in ou[ou >= 5].head(4).items())})
        # studies agree?
        if matched.any():
            os_ = o.loc[matched, "study_s"].to_numpy()
            lines.append(f"Study agrees for {np.mean(os_ == d.loc[matched, 'study'].to_numpy()):.1%} of matched images.")
        # tests
        trows = []
        for t in TESTS:
            nv = d.loc[matched, t].to_numpy()
            ov = pd.to_numeric(o.loc[matched, t], errors="coerce").to_numpy()
            ov = np.where(np.isin(ov, [0, 1]), ov, np.nan)
            both_obs = ~np.isnan(nv) & ~np.isnan(ov)
            trows.append({"test": TEST_SHORT[t], "both recorded": int(both_obs.sum()),
                          "agree": float(np.mean(nv[both_obs] == ov[both_obs])) if both_obs.any() else np.nan,
                          "paper recorded, ours missing": int((~np.isnan(nv) & np.isnan(ov)).sum()),
                          "ours recorded, paper missing": int((np.isnan(nv) & ~np.isnan(ov)).sum())})
        filled_by_study = (d.loc[matched].assign(
            paper_has_test=d.loc[matched, TESTS].notna().any(axis=1).to_numpy(),
            ours_has_test=pd.DataFrame({t: pd.to_numeric(o.loc[matched, t], errors="coerce").isin([0, 1]).to_numpy() for t in TESTS}).any(axis=1).to_numpy())
            .groupby("study").agg(images=("subject", "size"), paper_has_a_test=("paper_has_test", "sum"),
                                  ours_has_a_test=("ours_has_test", "sum")).reset_index())
        # posteriors
        prow = []
        for c in POSTS:
            nv = d.loc[matched, c].to_numpy()
            ov = pd.to_numeric(o.loc[matched, c], errors="coerce").to_numpy()
            ov = np.where((ov >= 0) & (ov <= 1), ov, np.nan)
            b = ~np.isnan(nv) & ~np.isnan(ov)
            dd = np.abs(nv[b] - ov[b])
            prow.append({"posterior": c, "both present": int(b.sum()),
                         "median abs(paper - ours)": float(np.median(dd)) if b.any() else np.nan,
                         "share abs(diff) > 0.05": float(np.mean(dd > 0.05)) if b.any() else np.nan,
                         "max abs(diff)": float(dd.max()) if b.any() else np.nan,
                         "paper present, ours -1": int((~np.isnan(nv) & np.isnan(ov)).sum()),
                         "ours present, paper missing": int((np.isnan(nv) & ~np.isnan(ov)).sum())})
        # hard labels vs our bins
        brow = []
        for h, b in (("hard_bac", "bac_LCA_bin"), ("hard_fung", "fung_LCA_bin")):
            if b in o:
                ob = pd.to_numeric(o.loc[matched, b], errors="coerce").to_numpy()
                ok = np.isin(ob, [0, 1])
                brow.append({"label": h, "our bin present": int(ok.sum()),
                             "Travis hard = our bin": float(np.mean(d.loc[matched, h].to_numpy()[ok] == ob[ok])) if ok.any() else np.nan})
        # what the paper left out of our CSV
        old_m = pd.Series(old.index.isin(old_hit), index=old.index)
        old["visit"] = [parse_visit(s, n) if s in STUDY_SHORT else "unknown"
                        for s, n in zip(old["study"].astype(str), old["basename"])]
        left = pd.crosstab([old.study_s, old.visit], old_m.map({True: "in paper set", False: "not in paper set"})).reset_index()
        out_rows = old[~old_m.to_numpy()]
        out_fu = out_rows.visit.isin(FOLLOWUP_VISITS)
        left_out = (len(out_rows), int(out_fu.sum()), out_rows[~out_fu].study_s.value_counts())
        # are the ones that aren't follow-ups the photos without test results or labels?
        oth = out_rows[~out_fu].copy()
        rec = pd.DataFrame({t: pd.to_numeric(oth[t], errors='coerce').isin([0, 1]) for t in TESTS if t in oth}, index=oth.index)
        oth['_any_test'] = rec.any(axis=1) if len(rec.columns) else False
        oth['_all_tests'] = rec.all(axis=1) if len(rec.columns) == 4 else False
        pb = pd.to_numeric(oth.get('bac_post_prob', pd.Series(np.nan, index=oth.index)), errors='coerce')
        oth['_posterior'] = pb.between(0, 1)
        bb = pd.to_numeric(oth.get('bac_LCA_bin', pd.Series(np.nan, index=oth.index)), errors='coerce')
        oth['_bin'] = bb.isin([0, 1])
        out_tests = (oth.groupby('study_s').agg(photos=('basename', 'size'), with_any_test=('_any_test', 'sum'),
                                                with_all_four_tests=('_all_tests', 'sum'), with_a_posterior=('_posterior', 'sum'),
                                                with_an_LCA_bin=('_bin', 'sum')).reset_index()
                     .rename(columns={'study_s': 'study'}).sort_values('photos', ascending=False))
        left_out = left_out + (int((~oth['_any_test']).sum()), int((~oth['_bin']).sum()))
        # subject vs pID, split by pID
        sub_lines = []
        if "pID" in o and matched.any():
            mm = d.loc[matched].assign(pID=o.loc[matched, "pID"].astype(str).to_numpy(),
                                       old_subject=o.loc[matched, "subject"].astype(str).to_numpy() if "subject" in o else None)
            if "subject" in o:
                sub_lines.append(f"Paper subject equals our subject for {np.mean(mm.subject == mm.old_subject):.1%} of matched images.")
            s2p = mm.groupby("subject")["pID"].nunique()
            p2s = mm.groupby("pID")["subject"].nunique()
            sub_lines.append(f"Paper subjects mapping to more than one pID: {int((s2p > 1).sum())}; pIDs mapping to more "
                             f"than one paper subject: {int((p2s > 1).sum())}.")
            leak = mm.groupby("pID")["split"].nunique()
            sub_lines.append(f"pIDs with images in more than one paper split: {int((leak > 1).sum())} "
                             f"({int(mm[mm.pID.isin(leak[leak > 1].index)].shape[0])} images).")
            if (leak > 1).any():
                notes.append(f"{int((leak > 1).sum())} participants (by pID) are in more than one of the paper's splits")
        # image files on disk
        exist_line = ""
        if "file_path" in o and matched.any():
            lp = o.loc[matched, "file_path"].astype(str).str.replace(PACS_PREFIX, a.img_root, regex=False)
            ex = lp.map(os.path.exists)
            exist_line = f"Image files found on disk for matched images: {int(ex.sum()):,} of {int(matched.sum()):,}."
        section("8. Against our CSV", "\n".join(f"- {x}" for x in lines + sub_lines + ([exist_line] if exist_line else [])),
                "File-name formats where matching failed (digit runs shown as #; formats with at least 5 images):\n\n"
                + md(save(pd.DataFrame(shape_rows), "unmatched_name_formats.csv"), ".0f"),
                "Tests, paper vs ours (matched images):\n\n" + md(save(pd.DataFrame(trows), "old_vs_paper_tests.csv")),
                "Images with any test recorded, by study (matched):\n\n" + md(save(filled_by_study, "old_vs_paper_tests_by_study.csv"), ".0f"),
                "Posteriors, paper vs ours:\n\n" + md(save(pd.DataFrame(prow), "old_vs_paper_posteriors.csv")),
                "Travis's hard labels vs our LCA bins:\n\n" + md(pd.DataFrame(brow)),
                "Our images in and out of the paper set, by study and visit:\n\n" + md(save(left, "old_in_paper_set.csv"), ".0f"),
                "Our photos in none of the paper's files and not follow-ups: do they have tests and labels?\n\n"
                + md(save(out_tests, "old_not_in_paper_tests.csv"), ".0f"))
        answers_old = (matched, filled_by_study, left_out)
        d["_our_path"] = np.where(matched, o["file_path"].astype(str).str.replace(PACS_PREFIX, a.img_root, regex=False), "")
    else:
        section("8. Against our CSV", f"Skipped (no old CSV at {a.old_csv}).")
        answers_old = None
        d["_our_path"] = ""

    # ---- 9. Image files for every paper image -----------------------------------------------------------
    # via our CSV (matched rows), else the paper's own relative path under --img_root (the Drive layout
    # mirrors the image store for the matched images)
    d["_by_rel"] = (a.img_root.rstrip("/") + "/" + d["relpath_orig"]).where(d["relpath_orig"] != "", "")
    d["_found_csv"] = d["_our_path"].map(lambda x: bool(x) and os.path.exists(x))
    d["_found_rel"] = d["_by_rel"].map(lambda x: bool(x) and os.path.exists(x))
    d["_found"] = d["_found_csv"] | d["_found_rel"]
    files_tab = (d.groupby(["study", "split"]).agg(images=("subject", "size"), via_our_csv=("_found_csv", "sum"),
                                                     via_relative_path=("_found_rel", "sum"), found=("_found", "sum"))
                 .reset_index())
    files_tab["missing"] = files_tab["images"] - files_tab["found"]
    section("9. Image files on disk for the paper's images",
            f"Looked up through our CSV (matched images) and as `{a.img_root}/<study folder>/<subfolder>/<file>` "
            "(the paper's relative path).\n\n" + md(save(files_tab, "image_files.csv"), ".0f"))
    files_answer = (f"**Image files.** Found on disk for {int(d._found.sum()):,} of {len(d):,} paper images ("
                   + ", ".join(f"{s} {int(d[d.split == s]._found.sum()):,} of {int((d.split == s).sum()):,}" for s in SPLITS)
                   + "); missing by study: "
                   + (", ".join(f"{k} {v}" for k, v in d[~d._found].study.value_counts().items()) or "none") + ".")

    # ---- Answers ----------------------------------------------------------------------------------------
    answers.append(f"**Images and participants.** {len(d):,} images, {d.subject.nunique():,} subjects "
                   f"(paper: {PAPER_IMAGES:,} / {PAPER_PARTICIPANTS:,}); "
                   + " / ".join(f"{s} {int((d.split == s).sum()):,}" for s in SPLITS) + " images. "
                   f"Follow-up photos: {int(d.is_followup.sum()):,}. Studies: "
                   + ", ".join(f"{k} {v:,}" for k, v in d.study.value_counts().items()) + ".")
    answers.append(f"**{'All three files.' if a.val else 'A missing validation set?'}** These files hold "
                   f"{len(d) / PAPER_IMAGES:.1%} of the paper's images ({len(d):,} of {PAPER_IMAGES:,}) and "
                   f"{d.subject.nunique() / PAPER_PARTICIPANTS:.1%} of its participants ({d.subject.nunique():,} of "
                   f"{PAPER_PARTICIPANTS:,}); " + ", ".join(f"{s} {int((d.split == s).sum()) / PAPER_IMAGES:.1%}" for s in SPLITS)
                   + " of the paper's images (the paper split 70/15/15).")
    answers.append(f"**Split grouped by participant?** Subjects in more than one split: {len(both)} ({pair_txt}).")
    mutt = d[d.study.isin(["MUTT1", "MUTT2", "MUTT Pilot"])]
    scut = d[d.study == "SCUT"]
    if len(mutt) or len(scut):
        answers.append(f"**Trial test results.** MUTT images with a smear result: "
                       f"{int(mutt[['bac_smear', 'fung_smear']].notna().any(axis=1).sum()):,} of {len(mutt):,}; "
                       f"SCUT images with a culture result: {int(scut[['bac_cult_positive', 'fung_cult_positive']].notna().any(axis=1).sum()):,} "
                       f"of {len(scut):,}.")
    if hp.any():
        answers.append(f"**Two-class or per pathogen?** {float((dev <= 1e-3).mean()):.1%} of posterior pairs sum to 1 "
                       f"(within 0.001). Within-pattern SD given the pathogen's own two tests: bacterial "
                       f"{diag.iloc[0, 1]:.4f}, fungal {diag.iloc[1, 1]:.4f}. Supplement LCA applied to each pathogen's own "
                       f"tests reproduces the posteriors to within {max_own_diff:.3f}.")
    allneg = pat[pat.pattern == "Gram- KOH- BA- PDA-"]
    if len(allneg):
        r = allneg.iloc[0]
        answers.append(f"**All four tests negative.** {int(r.images)} images / {int(r.subjects)} subjects; paper "
                       f"bac_post_prob {r.bac_post_mean:.3f} (range {r.bac_post_min:.3f}-{r.bac_post_max:.3f}); naive "
                       f"two-class {r.naive_two_class_all4:.3f}; our old CSV 0.688.")
    agree_tab = pd.DataFrame(agree)
    answers.append("**Hard labels = posterior > 0.5?** " + ", ".join(
        f"{r['pathogen']} {r['hard = (posterior > 0.5)']:.1%}" for _, r in agree_tab.iterrows()) + ".")
    vc = d.hard_combo.value_counts()
    answers.append("**Travis's hard labels** [bacterial, fungal]: " + ", ".join(f"[{k}] {vc.get(k, 0):,}" for k in ("1,0", "0,1", "1,1", "0,0"))
                   + f"; [0,0] with no test recorded at all: {int((d.no_tests & (d.hard_combo == '0,0')).sum()):,}.")
    if answers_old is not None:
        matched, _, (n_out, n_out_fu, out_other, out_no_test, out_no_bin) = answers_old
        answers.append(f"**Against our CSV.** {int(matched.sum()):,} of {len(d):,} paper images matched to our rows.")
        answers.append(f"**Our photos in none of the paper's files.** {n_out:,} of our CSV rows: {n_out_fu:,} follow-ups and "
                       f"{int(out_other.sum()):,} others (" + ", ".join(f"{k} {v:,}" for k, v in out_other.items()) + "). "
                       f"Of those others, {out_no_test:,} have no test result and {out_no_bin:,} no LCA bin in our CSV.")
    answers.append(files_answer)

    with open(os.path.join(a.out, "eda_report.md"), "w") as f:
        f.write("# EDA of the paper's data (" + ", ".join(os.path.basename(x[0]) for x in ((a.train,), (a.val,), (a.test,)) if x[0])
                + ")\n\n"
                "Aggregates only; no identifiers.\n\n## Answers\n\n" + "\n".join(f"- {x}" for x in answers) + "\n\n")
        f.write("## Things to look at\n\n" + ("\n".join(f"- {x}" for x in notes) if notes else "- none flagged") + "\n\n")
        for title, text in S:
            f.write(f"## {title}\n\n{text}\n\n")
    print("\n".join(answers))
    print("\nThings to look at:\n" + ("\n".join(f"  - {x}" for x in notes) if notes else "  none"))
    print(f"\nReport: {os.path.join(a.out, 'eda_report.md')}")


if __name__ == "__main__":
    main()
