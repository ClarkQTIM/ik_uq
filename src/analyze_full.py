"""
Full analysis: hypotheses H1-H4 across strata (see ANALYSIS_PLAN.md).

Runs are grouped by model (head and KL arm) and label type from their config.json, so any mix can
be passed: the single-split runs now, or cross-validation folds later (one run per fold; folds of
the same model are pooled, each image predicted once).

    python src/analyze_full.py --runs RUN [RUN ...] --out OUT [--csv CSV] [--reps 1000]

--csv adds the strata that need the raw test results (one pathogen +, no positive test,
co-positive); without it those strata are skipped. Outputs are aggregates only.
"""

import argparse
import json
import os
import sys
import zlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ALPHA = 0.1
ARRAY_KEYS = ["output", "p_hat", "p_bac_binary", "y_bac", "soft_bac", "has_posterior", "is_followup",
              "study", "csv_row", "participant", "part"]


# =============================================================================
# Loading
# =============================================================================

def model_name(cfg):
    if cfg["head"] == "sigmoid":
        return "sigmoid"
    if cfg.get("kl_schedule", "icmla") == "icmla":
        arm = "ICMLA"
    else:
        arm = ("Sensoy" if cfg.get("kl_form", "sensoy") == "sensoy" else "eq5") + f" {cfg['kl_max']:g}"
    return ("BD " if cfg["head"] == "beta_dirichlet" else "Dirichlet ") + arm


def load_runs(dirs):
    groups = {}
    for d in dirs:
        npz, cfgp = os.path.join(d, "test_predictions.npz"), os.path.join(d, "config.json")
        if not (os.path.exists(npz) and os.path.exists(cfgp)):
            print(f"  skipping {d}: no test_predictions.npz / config.json")
            continue
        with open(cfgp) as f:
            cfg = json.load(f)
        z = np.load(npz, allow_pickle=False)
        key = (model_name(cfg), cfg["label_type"])
        fold = cfg.get("fold")
        groups.setdefault(key, []).append((0 if fold is None else int(fold), {k: z[k] for k in z.files}, d))
    data = {}
    for key, runs in groups.items():
        folds = [f for f, _, _ in runs]
        if len(set(folds)) != len(folds):
            raise SystemExit(f"{key[0]} / {key[1]}: more than one run for the same fold:\n  "
                             + "\n  ".join(d for _, _, d in runs))
        cat = {k: np.concatenate([z[k] for _, z, _ in runs]) for k in ARRAY_KEYS}
        cat["fold"] = np.concatenate([np.full(len(z["csv_row"]), f) for f, z, _ in runs])
        order = np.argsort(cat["csv_row"], kind="stable")
        cat = {k: v[order] for k, v in cat.items()}
        if len(np.unique(cat["csv_row"])) != len(cat["csv_row"]):
            raise SystemExit(f"{key}: the same image appears in more than one fold.")
        cat["head"] = str(runs[0][1]["head"])
        cat["n_folds"] = len(runs)
        data[key] = cat
    if not data:
        raise SystemExit("No runs loaded.")
    ref = next(iter(data.values()))["csv_row"]
    for key, d in data.items():
        if not np.array_equal(d["csv_row"], ref):
            raise SystemExit(f"{key} covers different test images from the others.")
    return data


# =============================================================================
# Metrics
# =============================================================================

def entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def auroc(lbl, s):
    lbl = np.asarray(lbl, bool)
    n1, n0 = lbl.sum(), (~lbl).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[lbl].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _bins(p, nb=10):
    return np.minimum((p * nb).astype(int), nb - 1)


def ece(p, y, nb=10):
    b = _bins(p, nb)
    tot = 0.0
    for k in np.unique(b):
        m = b == k
        tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return tot


def reliability(p, y, nb=10):
    """Murphy decomposition: the reliability (miscalibration) term."""
    b = _bins(p, nb)
    tot = 0.0
    for k in np.unique(b):
        m = b == k
        tot += m.mean() * (p[m].mean() - y[m].mean()) ** 2
    return tot


def sel_acc(ok, u, frac):
    if len(ok) == 0:
        return np.nan
    k = max(1, int(round(frac * len(ok))))
    return ok[np.argsort(u, kind="stable")[:k]].mean()


def aurc(ok, u):
    if len(ok) == 0:
        return np.nan
    o = np.argsort(u, kind="stable")
    return ((~ok[o]).cumsum() / np.arange(1, len(o) + 1)).mean()


def spearman(a, b):
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    return spearmanr(a, b)[0]


def qhat(scores):
    s = np.sort(scores)
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - ALPHA)))
    return 1.0 if k > n else s[k - 1]


def calibration_draws(part, fold, grp, rng, draws=1000):
    """One image per calibration participant, `draws` times, per fold: an array (draws x participants).
    Drawn once and shared by every model, so soft/hard and model-to-model comparisons use the same
    calibration images (common random numbers) and no result depends on which other runs are loaded."""
    out = {}
    for f in np.unique(fold):
        cal = np.where((fold == f) & (part == 0))[0]
        if len(cal) == 0:
            raise SystemExit(f"Fold {f} has no calibration images.")
        members = [cal[grp[cal] == g] for g in np.unique(grp[cal])]
        out[f] = np.stack([v[rng.integers(0, len(v), draws)] for v in members], 1)
    return out


def conformal(p, y, fold, cal_draws):
    """Binary LAC per fold, participant-level calibration (median threshold over the shared draws)."""
    P = np.stack([1 - p, p], 1)
    size = np.zeros(len(p), int)
    cover = np.zeros(len(p), bool)
    for f in np.unique(fold):
        m = fold == f
        ix = cal_draws[f]                      # draws x participants
        s = np.sort(1 - P[ix, y[ix]], axis=1)
        n = s.shape[1]
        k = int(np.ceil((n + 1) * (1 - ALPHA)))
        q = 1.0 if k > n else float(np.median(s[:, k - 1]))
        S = P[m] >= 1 - q
        size[m] = S.sum(1)
        cover[m] = S[np.arange(m.sum()), y[m]]
    return size, cover


# =============================================================================
# Bootstrap over participants
# =============================================================================

class Boot:
    def __init__(self, grp, idx, reps, rng):
        self.idx = idx
        g = grp[idx]
        self.ug = np.unique(g)
        self.members = [idx[g == u] for u in self.ug]
        self.samples = []
        for _ in range(reps):
            pick = rng.integers(0, len(self.ug), len(self.ug))
            self.samples.append(np.concatenate([self.members[i] for i in pick]))

    def ci(self, fn):
        vals = np.array([fn(b) for b in self.samples], float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 10:
            return np.nan, np.nan, np.nan
        p = 2 * min(np.mean(vals <= 0), np.mean(vals >= 0))
        return np.percentile(vals, 2.5), np.percentile(vals, 97.5), max(p, 1 / len(vals))


def verdict(lo, hi, direction):
    if direction == 0 or np.isnan(lo):
        return "-"
    if direction > 0:
        return "holds" if lo > 0 else ("reversed" if hi < 0 else "inconclusive")
    return "holds" if hi < 0 else ("reversed" if lo > 0 else "inconclusive")


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default=None, help="study CSV, for the test-result strata")
    ap.add_argument("--group_col", default="pID")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # Separate, named random streams: a result never depends on which other runs or strata are loaded.
    def rng_for(*name):
        return np.random.default_rng([a.seed] + [zlib.crc32(str(x).encode()) for x in name])

    data = load_runs(a.runs)
    keys = sorted(data)
    print("Models loaded:", ", ".join(f"{m} / {l} ({data[(m, l)]['n_folds']} fold(s))" for m, l in keys))
    r = data[keys[0]]
    y = r["y_bac"].astype(int)
    study = r["study"].astype(str)
    fu = r["is_followup"].astype(bool)
    post = r["soft_bac"].astype(float)
    has_post = r["has_posterior"].astype(bool)
    grp = r["participant"]
    part = r["part"]
    fold = r["fold"]
    mad = study == "MADURAI"
    amb = has_post & (post > 0.1) & (post < 0.9)
    Hpost = entropy(post)

    # ---- strata ---------------------------------------------------------------
    strata = {
        "All": np.ones(len(y), bool),
        "All, enrolment only": ~fu,
        "Single-etiology trials, enrolment only": ~mad & ~fu,
        "MADURAI": mad,
        "MADURAI, confident label": mad & has_post & ((post <= 0.1) | (post >= 0.9)),
        "MADURAI, ambiguous label": mad & amb,
    }
    if a.csv:
        from copositive_check import participant_groups
        from labels import load_labelled
        df = load_labelled(a.csv, img_root=None, group_col=a.group_col, verbose=False)
        cat_p = participant_groups(df, a.group_col)
        by_row = pd.Series(df[a.group_col].map(cat_p).to_numpy(), index=df["csv_row"].to_numpy())
        cat = by_row.reindex(r["csv_row"]).to_numpy()
        strata["MADURAI, one pathogen +"] = mad & (cat == "one pathogen +")
        strata["MADURAI, no positive test"] = mad & (cat == "no positive test")
        strata["MADURAI, co-positive"] = mad & np.isin(cat, ["bac+ and fung+", "both cultures +"])
    else:
        print("No --csv: skipping the test-result strata (one pathogen +, no positive test, co-positive).")
    design = {k: {"images": int(m.sum()), "participants": int(len(np.unique(grp[m]))),
                  "eval_images": int((m & (part == 1)).sum())} for k, m in strata.items()}
    print("\nStrata:")
    for k, v in design.items():
        print(f"  {k:42s} {v['images']:5d} images, {v['participants']:4d} participants")
    names = list(strata)
    same = [(x, z) for i, x in enumerate(names) for z in names[i + 1:]
            if strata[x].any() and np.array_equal(strata[x], strata[z])]
    for x, z in same:
        print(f"  Note: '{x}' and '{z}' contain exactly the same images.")

    # ---- signals per model ---------------------------------------------------
    M = {}
    cal_draws = calibration_draws(part, fold, grp, rng_for("conformal"))
    for key in keys:
        d = data[key]
        p = d["p_bac_binary"].astype(float)
        H = entropy(p)
        ok = (p > 0.5) == (y == 1)
        ev = d["output"][:, :4].astype(float).sum(1) / 2 if d["head"] != "sigmoid" else None
        size, cover = conformal(p, y, fold, cal_draws)
        t67 = np.zeros(len(p))
        if ev is not None:  # calibration-set 67th percentile of evidence, per fold
            for f in np.unique(fold):
                m = fold == f
                t67[m] = np.percentile(ev[m & (part == 0)], 67)
        M[key] = dict(p=p, H=H, ok=ok, ev=ev, low_ev=None if ev is None else -ev,
                      size=size, cover=cover, t67=t67)

    boots = {k: Boot(grp, np.where(m)[0], a.reps, rng_for("stratum", k)) for k, m in strata.items() if m.sum() > 0}
    evalm = part == 1

    # ---- H1: soft vs hard ----------------------------------------------------
    def m_(key):
        return M[key]

    H1_METRICS = [  # name, function(key, idx), direction (+1 higher is better for soft, -1 lower, 0 none)
        ("AUROC", lambda k, i: auroc(y[i], m_(k)["p"][i]), +1),
        ("accuracy", lambda k, i: m_(k)["ok"][i].mean() if len(i) else np.nan, +1),
        ("Brier", lambda k, i: np.mean((m_(k)["p"][i] - y[i]) ** 2) if len(i) else np.nan, -1),
        ("reliability (Murphy)", lambda k, i: reliability(m_(k)["p"][i], y[i]) if len(i) else np.nan, -1),
        ("ECE", lambda k, i: ece(m_(k)["p"][i], y[i]) if len(i) else np.nan, -1),
        ("error detection, entropy", lambda k, i: auroc(~m_(k)["ok"][i], m_(k)["H"][i]), +1),
        ("error detection, evidence",
         lambda k, i: np.nan if m_(k)["ev"] is None else auroc(~m_(k)["ok"][i], m_(k)["low_ev"][i]), +1),
        ("accuracy, 25% most confident", lambda k, i: sel_acc(m_(k)["ok"][i], m_(k)["H"][i], 0.25), +1),
        ("accuracy, 50% most confident", lambda k, i: sel_acc(m_(k)["ok"][i], m_(k)["H"][i], 0.50), +1),
        ("accuracy, 75% most confident", lambda k, i: sel_acc(m_(k)["ok"][i], m_(k)["H"][i], 0.75), +1),
        ("AURC (risk-coverage)", lambda k, i: aurc(m_(k)["ok"][i], m_(k)["H"][i]), -1),
        ("label ambiguity AUROC (entropy)",
         lambda k, i: auroc(amb[i][has_post[i]], m_(k)["H"][i][has_post[i]]), +1),
        ("label ambiguity Spearman (entropy)",
         lambda k, i: spearman(m_(k)["H"][i][has_post[i]], Hpost[i][has_post[i]]), +1),
        ("mean |p - posterior|",
         lambda k, i: np.mean(np.abs(m_(k)["p"][i] - post[i])[has_post[i]]) if has_post[i].any() else np.nan, -1),
        ("Brier against the posterior",
         lambda k, i: np.mean(((m_(k)["p"][i] - post[i]) ** 2)[has_post[i]]) if has_post[i].any() else np.nan, -1),
        ("conformal coverage (eval)",
         lambda k, i: m_(k)["cover"][i[evalm[i]]].mean() if evalm[i].any() else np.nan, 0),
        ("conformal two-label rate (eval)",
         lambda k, i: (m_(k)["size"][i[evalm[i]]] == 2).mean() if evalm[i].any() else np.nan, -1),
    ]
    pairs = sorted({m for m, _ in keys if (m, "soft") in M and (m, "hard") in M})
    rows = []
    for s_name, bt in boots.items():
        idx = bt.idx
        for model in pairs:
            ks, kh = (model, "soft"), (model, "hard")
            for name, fn, direction in H1_METRICS:
                vs, vh = fn(ks, idx), fn(kh, idx)
                if np.isnan(vs) and np.isnan(vh):
                    continue
                lo, hi, pv = bt.ci(lambda b: fn(ks, b) - fn(kh, b))
                rows.append({"stratum": s_name, "model": model, "metric": name, "soft": vs, "hard": vh,
                             "soft_minus_hard": vs - vh, "ci_low": lo, "ci_high": hi, "p_boot": pv,
                             "verdict": verdict(lo, hi, direction),
                             "primary": s_name == "MADURAI" and name in ("Brier", "label ambiguity AUROC (entropy)")})
    h1 = pd.DataFrame(rows)
    if len(h1) and h1["primary"].any():
        prim = h1["primary"] & h1["p_boot"].notna()
        h1.loc[prim, "p_holm"] = holm(h1.loc[prim, "p_boot"])
    h1.to_csv(os.path.join(a.out, "h1_soft_vs_hard.csv"), index=False)

    # Where the predictions move: mean p(bacterial) by reference class (descriptive; explains H1 shifts)
    rows = []
    for s_name, m in strata.items():
        for cls, cname in ((1, "bacterial"), (0, "fungal")):
            mc = m & (y == cls)
            if not mc.any():
                continue
            for key in keys:
                p = M[key]["p"]
                rows.append({"stratum": s_name, "reference": cname, "model": key[0], "labels": key[1],
                             "images": int(mc.sum()), "participants": int(len(np.unique(grp[mc]))),
                             "mean posterior": post[mc & has_post].mean() if (mc & has_post).any() else np.nan,
                             "mean p(bacterial)": p[mc].mean(), "share called bacterial": (p[mc] > 0.5).mean()})
    h1c = pd.DataFrame(rows)
    h1c.to_csv(os.path.join(a.out, "h1_by_reference_class.csv"), index=False)

    # ---- H2: uncertainty by stratum -------------------------------------------
    def signals(key):
        out = {"entropy": M[key]["H"]}
        if M[key]["ev"] is not None:
            out["low evidence (-evidence)"] = M[key]["low_ev"]
        return out

    rows = []
    for key in keys:
        for s_name, m in strata.items():
            if not m.any():
                continue
            row = {"model": key[0], "labels": key[1], "stratum": s_name, "images": int(m.sum()),
                   "mean entropy": M[key]["H"][m].mean(),
                   "two-label rate (eval)": (M[key]["size"][m & evalm] == 2).mean() if (m & evalm).any() else np.nan,
                   "accuracy": M[key]["ok"][m].mean()}
            if M[key]["ev"] is not None:
                row["mean evidence"] = M[key]["ev"][m].mean()
            rows.append(row)
    h2 = pd.DataFrame(rows)
    h2.to_csv(os.path.join(a.out, "h2_uncertainty_by_stratum.csv"), index=False)

    contrasts = [("MADURAI", "Single-etiology trials, enrolment only"),
                 ("MADURAI, confident label", "Single-etiology trials, enrolment only"),
                 ("MADURAI, ambiguous label", "MADURAI, confident label"),
                 ("MADURAI, no positive test", "MADURAI, one pathogen +"),
                 ("MADURAI, co-positive", "MADURAI, one pathogen +")]
    rows = []
    for sa, sb in contrasts:
        if sa not in strata or sb not in strata or not strata[sa].any() or not strata[sb].any():
            continue
        ma, mb = strata[sa], strata[sb]
        bt = Boot(grp, np.where(ma | mb)[0], a.reps, rng_for("contrast", sa, sb))
        for key in keys:
            for sig_name, sig in signals(key).items():
                def diff(b, sig=sig):
                    ia, ib = b[ma[b]], b[mb[b]]
                    return sig[ia].mean() - sig[ib].mean() if len(ia) and len(ib) else np.nan
                d0 = sig[ma].mean() - sig[mb].mean()
                lo, hi, pv = bt.ci(diff)
                rows.append({"contrast": f"{sa} minus {sb}", "model": key[0], "labels": key[1], "signal": sig_name,
                             "difference": d0, "ci_low": lo, "ci_high": hi, "p_boot": pv,
                             "verdict": verdict(lo, hi, +1)})
    h2c = pd.DataFrame(rows)
    h2c.to_csv(os.path.join(a.out, "h2_contrasts.csv"), index=False)

    # per study
    rows = []
    for key in keys:
        for s in np.unique(study):
            m = (study == s) & ~fu
            if m.sum() == 0:
                continue
            row = {"model": key[0], "labels": key[1], "study": s, "images": int(m.sum()),
                   "mean entropy": M[key]["H"][m].mean(), "accuracy": M[key]["ok"][m].mean(),
                   "AUROC": auroc(y[m], M[key]["p"][m])}
            if M[key]["ev"] is not None:
                row["mean evidence"] = M[key]["ev"][m].mean()
            rows.append(row)
    by_study = pd.DataFrame(rows)
    by_study.to_csv(os.path.join(a.out, "h2_h3_by_study.csv"), index=False)

    # ---- H3: uncertainty tracks performance -----------------------------------
    rows = []
    for key in keys:
        for s_name, bt in boots.items():
            i = bt.idx
            ok, H = M[key]["ok"][i], M[key]["H"][i]
            row = {"model": key[0], "labels": key[1], "stratum": s_name, "images": len(i),
                   "accuracy": ok.mean(), "error detection, entropy": auroc(~ok, H)}
            lo, hi, _ = bt.ci(lambda b, key=key: auroc(~M[key]["ok"][b], M[key]["H"][b]) - 0.5)
            row["error detection, entropy, ci"] = f"{lo + 0.5:.3f}-{hi + 0.5:.3f}" if np.isfinite(lo) else ""
            if M[key]["ev"] is not None:
                row["error detection, evidence"] = auroc(~ok, M[key]["low_ev"][i])
            t1, t2 = np.percentile(H, [100 / 3, 200 / 3])
            for lab, mm in (("accuracy, low-entropy third", H <= t1), ("accuracy, middle third", (H > t1) & (H <= t2)),
                            ("accuracy, high-entropy third", H > t2)):
                row[lab] = ok[mm].mean() if mm.any() else np.nan
            rows.append(row)
    h3 = pd.DataFrame(rows)
    # pooled vs within-study error detection (enrolment photos)
    rows = []
    for key in keys:
        pool_m = ~fu
        pooled = auroc(~M[key]["ok"][pool_m], M[key]["H"][pool_m])
        vals, ws = [], []
        for s in np.unique(study):
            m = (study == s) & ~fu
            ok = M[key]["ok"][m]
            if (~ok).sum() >= 5 and ok.sum() >= 5:
                vals.append(auroc(~ok, M[key]["H"][m]))
                ws.append(m.sum())
        within = np.average(vals, weights=ws) if vals else np.nan
        sub = by_study[(by_study.model == key[0]) & (by_study.labels == key[1]) & (by_study.images >= 10)]
        rows.append({"model": key[0], "labels": key[1], "pooled error detection (enrolment)": pooled,
                     "within-study error detection (weighted mean)": within, "studies used": len(vals),
                     "study-level Spearman(mean entropy, accuracy)": spearman(sub["mean entropy"].to_numpy(),
                                                                             sub["accuracy"].to_numpy()),
                     "studies (n >= 10)": len(sub)})
    h3p = pd.DataFrame(rows)
    h3.to_csv(os.path.join(a.out, "h3_tracking.csv"), index=False)
    h3p.to_csv(os.path.join(a.out, "h3_pooled_vs_within.csv"), index=False)

    # ---- H4: how the signals relate ---------------------------------------------
    rows, comb = [], []
    for key in keys:
        if M[key]["ev"] is None:
            continue
        sig_key = ("sigmoid", key[1])
        for s_name, m in strata.items():
            if m.sum() < 10:
                continue
            i = np.where(m)[0]
            ie = i[evalm[i]]
            H, lev = M[key]["H"][i], M[key]["low_ev"][i]
            k25 = max(1, int(round(0.25 * len(i))))
            top_h, top_e = set(i[np.argsort(-H)[:k25]]), set(i[np.argsort(-lev)[:k25]])
            row = {"model": key[0], "labels": key[1], "stratum": s_name, "images": len(i),
                   "Spearman(entropy, low evidence)": spearman(H, lev),
                   "Spearman(entropy, conformal set size) (eval)":
                       spearman(M[key]["H"][ie], M[key]["size"][ie].astype(float)) if len(ie) > 2 else np.nan,
                   "Jaccard, 25% most uncertain (entropy vs evidence)": len(top_h & top_e) / len(top_h | top_e)}
            if sig_key in M:
                row["Spearman(sigmoid entropy, BD low evidence)"] = spearman(M[sig_key]["H"][i], lev)
            rows.append(row)
            # combined filter vs each alone, eval images
            if sig_key in M and len(ie) >= 10:
                s_ok = M[sig_key]["ok"][ie]
                single = M[sig_key]["size"][ie] == 1
                sel = single & (M[key]["ev"][ie] > M[key]["t67"][ie])
                n = max(int(round(sel.mean() * len(ie))), 1)
                comb.append({"model": key[0], "labels": key[1], "stratum": s_name, "coverage": sel.mean(),
                             "accuracy, all": s_ok.mean(),
                             "accuracy, combined": s_ok[sel].mean() if sel.any() else np.nan,
                             "accuracy, sigmoid confidence alone": s_ok[np.argsort(M[sig_key]["H"][ie])[:n]].mean(),
                             "accuracy, evidence alone": s_ok[np.argsort(M[key]["low_ev"][ie])[:n]].mean()})
    h4 = pd.DataFrame(rows)
    h4c = pd.DataFrame(comb)
    h4.to_csv(os.path.join(a.out, "h4_signals.csv"), index=False)
    h4c.to_csv(os.path.join(a.out, "h4_combined_filter.csv"), index=False)

    # ---- per fold (training variation) -----------------------------------------
    rows = []
    for key in keys:
        for f in np.unique(fold):
            m = fold == f
            rows.append({"model": key[0], "labels": key[1], "fold": int(f), "images": int(m.sum()),
                         "AUROC": auroc(y[m], M[key]["p"][m]), "Brier": np.mean((M[key]["p"][m] - y[m]) ** 2),
                         "AUROC, MADURAI": auroc(y[m & mad], M[key]["p"][m & mad])})
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(os.path.join(a.out, "per_fold.csv"), index=False)

    with open(os.path.join(a.out, "design.json"), "w") as f:
        json.dump({"strata": design, "models": [f"{m} / {l}" for m, l in keys], "reps": a.reps,
                   "alpha": ALPHA, "csv_strata": bool(a.csv)}, f, indent=2)

    # ---- report ------------------------------------------------------------------
    def md(df, cols=None):
        d = df if cols is None else df[[c for c in cols if c in df.columns]]
        try:
            return d.to_markdown(index=False, floatfmt=".3f")
        except ImportError:
            return "```\n" + d.to_string(index=False, float_format=lambda v: f"{v:.3f}") + "\n```"

    with open(os.path.join(a.out, "report.md"), "w") as f:
        f.write("# Full analysis (H1-H4 x strata)\n\nSee ANALYSIS_PLAN.md for definitions. "
                f"Bootstrap over participants, {a.reps} resamples.\n\n")
        f.write("## Strata\n\n" + md(pd.DataFrame(design).T.reset_index().rename(columns={"index": "stratum"})) + "\n\n")
        for x, z in same:
            f.write(f"Note: '{x}' and '{z}' contain exactly the same images.\n\n")
        if len(h1):
            f.write("## H1. Soft vs hard\n\n### Primary (MADURAI, Holm-corrected)\n\n"
                    + md(h1[h1.primary], ["model", "metric", "soft", "hard", "soft_minus_hard", "ci_low",
                                          "ci_high", "p_boot", "p_holm", "verdict"]) + "\n\n")
            for s_name in h1.stratum.unique():
                f.write(f"### {s_name}\n\n" + md(h1[h1.stratum == s_name],
                                                  ["model", "metric", "soft", "hard", "soft_minus_hard",
                                                   "ci_low", "ci_high", "verdict"]) + "\n\n")
            f.write("### Predictions by reference class\n\n" + md(h1c) + "\n\n")
        f.write("## H2. Uncertainty by stratum\n\n" + md(h2) + "\n\n### Contrasts\n\n" + md(h2c) + "\n\n")
        f.write("### By study (enrolment photos)\n\n" + md(by_study) + "\n\n")
        f.write("## H3. Uncertainty tracks performance\n\n" + md(h3) + "\n\n### Pooled vs within-study\n\n"
                + md(h3p) + "\n\n")
        f.write("## H4. How the signals relate (BD models)\n\n" + md(h4) + "\n\n### Combined filter\n\n"
                + md(h4c) + "\n\n")
        f.write("## Per fold\n\n" + md(per_fold) + "\n")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    if len(h1):
        print("\nH1 primary (MADURAI):")
        cols = [c for c in ["model", "metric", "soft", "hard", "soft_minus_hard", "ci_low", "ci_high",
                            "p_holm", "verdict"] if c in h1.columns]
        print(h1[h1.primary][cols].round(3).to_string(index=False))
    print(f"\nReport: {os.path.join(a.out, 'report.md')}")


if __name__ == "__main__":
    main()
