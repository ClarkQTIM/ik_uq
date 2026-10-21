"""
Site-Level Uncertainty Analysis — Soft Label Models Only
=========================================================
Computes per-study uncertainty statistics for all three UQ signals:
  1. Sigmoid entropy (from sigmoid soft model, Cell A)
  2. Conformal CPSL  (from sigmoid soft model + LAC, Cell A)
  3. BD evidence alpha+beta (from BD soft model, Cell B)

Uses existing inference caches — no GPU needed.

Run:
    python site_uncertainty.py
        --cache_sig  runs/sigmoid_soft_.../uncertainty_analysis/cache_sig.npz
        --cache_bd   runs/bd_soft_.../uncertainty_analysis/cache_bd.npz
        --csv        /path/to/data.csv
        --alpha      0.1
"""

import os, json, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STUDY_SHORT = {
    "MADURAI photos": "MADURAI",   "MUTT2-Pictures": "MUTT2",
    "MUTT1-Pictures": "MUTT1",     "SCUT photos": "SCUT",
    "MUTT Pilot Photos": "MUTT Pilot",
    "CLAIR II photos": "CLAIR II", "CLAIR I photos": "CLAIR I",
}
STUDY_ORDER = ["SCUT", "MUTT2", "MUTT1", "MUTT Pilot", "CLAIR II", "CLAIR I", "MADURAI"]

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_sig", type=str, required=True,
        help="cache_sig.npz from dirichlet_analysis run on Cell A "
             "(runs/beta_dirichlet_soft_.../uncertainty_analysis/cache_sig.npz)")
    p.add_argument("--cache_bd",  type=str, required=True,
        help="cache_bd.npz from dirichlet_analysis run on Cell B "
             "(runs/beta_dirichlet_soft_.../uncertainty_analysis/cache_bd.npz)")
    p.add_argument("--csv", type=str,
        default="/persist/QTIM/Active/26-0096/Corneal_Ulcer_Image_Sets/Data/"
                "corneal_ulcer_study_data_7_24_2025_lambda.csv")
    p.add_argument("--alpha",   type=float, default=0.1)
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--out_dir", type=str,   default=None)
    return p.parse_args()


def load_test_set(csv_path, seed=42):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["file_path", "subject"])
    df = df[(df["fung_LCA_bin"] != -1) | (df["bac_LCA_bin"] != -1)].reset_index(drop=True)
    subj = df["subject"].values
    gss = GroupShuffleSplit(1, test_size=0.30, random_state=seed)
    _, ti = next(gss.split(df, groups=subj))
    df_t = df.iloc[ti].reset_index(drop=True)
    gss2 = GroupShuffleSplit(1, test_size=0.50, random_state=seed)
    _, ti2 = next(gss2.split(df_t, groups=df_t["subject"].values))
    return df_t.iloc[ti2].reset_index(drop=True)


def factored_joint(probs):
    b, f = probs[:, 0], probs[:, 1]
    return np.stack([(1-b)*(1-f), b*(1-f), (1-b)*f, b*f], axis=1)


def shannon_entropy(p4):
    eps = 1e-12
    return -(p4 * np.log(p4 + eps)).sum(axis=1)


def lac_cpsl_4class(sig_probs, labels, cal_idx, eval_idx, alpha):
    """4-class Vanilla LAC."""
    p4    = factored_joint(sig_probs)
    bac_b = (labels[:, 0] > 0.5).astype(int)
    fng_b = (labels[:, 1] > 0.5).astype(int)
    lab4  = bac_b + fng_b * 2
    n     = len(cal_idx)
    scores = 1 - p4[cal_idx][np.arange(n), lab4[cal_idx]]
    qlevel = min(np.ceil((n+1)*(1-alpha))/n, 1.0)
    qhat   = float(np.quantile(scores, qlevel, method="higher"))
    psets  = (p4[eval_idx] >= (1-qhat)).sum(axis=1)
    return psets, qhat

# Keep old name as alias for compatibility
lac_cpsl = lac_cpsl_4class

def lac_cpsl_2class(sig_probs, labels, cal_idx, eval_idx, alpha, pathogen=0):
    """
    2-class binary LAC for one pathogen.
    pathogen=0: BAC (p_bac vs 1-p_bac)
    pathogen=1: FUNG (p_fung vs 1-p_fung)
    CPSL is either 1 (singleton) or 2 (both classes included).
    """
    p     = sig_probs[:, pathogen]          # predicted prob for this pathogen
    lab   = (labels[:, pathogen] > 0.5).astype(int)  # binary true label
    # Score for true class: 1 - p_hat_y
    scores_cal = np.where(lab[cal_idx] == 1,
                          1 - p[cal_idx],      # score for positive class
                          p[cal_idx])           # score for negative class (= 1 - (1-p))
    n      = len(cal_idx)
    qlevel = min(np.ceil((n+1)*(1-alpha))/n, 1.0)
    qhat   = float(np.quantile(scores_cal, qlevel, method="higher"))
    # Prediction set: include class if its score <= qhat
    # Class 1 (positive): score = 1 - p;  included if 1-p <= qhat => p >= 1-qhat
    # Class 0 (negative): score = p;      included if p   <= qhat
    include_pos = p[eval_idx] >= (1 - qhat)
    include_neg = p[eval_idx] <= qhat
    psets = include_pos.astype(int) + include_neg.astype(int)
    return psets, qhat


def main():
    args = get_args()
    if args.out_dir is None:
        args.out_dir = os.path.dirname(args.cache_bd)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    df_test = load_test_set(args.csv, args.seed)
    subj    = df_test["subject"].values
    gss     = GroupShuffleSplit(1, test_size=0.80, random_state=args.seed)
    cal_idx, eval_idx = next(gss.split(df_test, groups=subj))
    df_eval = df_test.iloc[eval_idx].reset_index(drop=True)
    studies = df_eval["study"].values

    print(f"Eval set: {len(df_eval):,} images  Cal: {len(cal_idx):,}")

    # ── Load caches ───────────────────────────────────────────────────────────
    sig_cache = np.load(args.cache_sig)
    bd_cache  = np.load(args.cache_bd)

    sig_probs = sig_cache["probs"]    # (785, 2)
    sig_labels = sig_cache["labels"]  # (785, 2)
    bd_params  = bd_cache["params"]   # (785, 4)

    # ── Compute signals on eval subset ────────────────────────────────────────
    # Sigmoid entropy
    p4_eval   = factored_joint(sig_probs[eval_idx])
    entropy   = shannon_entropy(p4_eval)

    # Conformal CPSL — 4-class and 2-class per pathogen
    cpsl_4,  qhat_4  = lac_cpsl_4class(sig_probs, sig_labels, cal_idx, eval_idx, args.alpha)
    cpsl_bac, qhat_b = lac_cpsl_2class(sig_probs, sig_labels, cal_idx, eval_idx, args.alpha, pathogen=0)
    cpsl_fung, qhat_f = lac_cpsl_2class(sig_probs, sig_labels, cal_idx, eval_idx, args.alpha, pathogen=1)
    print(f"4-class  qhat={qhat_4:.4f}  mean CPSL={cpsl_4.mean():.3f}")
    print(f"2-class BAC  qhat={qhat_b:.4f}  mean CPSL={cpsl_bac.mean():.3f}")
    print(f"2-class FUNG qhat={qhat_f:.4f}  mean CPSL={cpsl_fung.mean():.3f}")
    # Keep backward compat
    cpsl = cpsl_4

    # BD evidence
    a_bac  = bd_params[eval_idx, 0]; b_bac  = bd_params[eval_idx, 1]
    a_fung = bd_params[eval_idx, 2]; b_fung = bd_params[eval_idx, 3]
    e_bac  = a_bac + b_bac
    e_fung = a_fung + b_fung
    e_mean = (e_bac + e_fung) / 2

    # ── Per-study summary ─────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  PER-STUDY UNCERTAINTY (Soft Label Models — Cell A sigmoid, Cell B BD)")
    print(f"{'='*75}")
    print(f"  {'Study':12s}  {'N':>5}  {'Entropy':>9}  "
          f"{'CPSL-4':>7}  {'CPSL-BAC':>9}  {'CPSL-FUNG':>10}  "
          f"{'BD e_bac':>9}  {'BD e_fung':>10}")
    print(f"  {'-'*85}")

    rows = []
    for study in np.unique(studies):
        mask  = studies == study
        short = STUDY_SHORT.get(study, study)
        n     = mask.sum()
        h     = entropy[mask].mean()
        c4    = cpsl_4[mask].mean()
        cb    = cpsl_bac[mask].mean()
        cf    = cpsl_fung[mask].mean()
        eb    = e_bac[mask].mean()
        ef    = e_fung[mask].mean()
        print(f"  {short:12s}  {n:>5}  {h:>9.4f}  "
              f"{c4:>7.3f}  {cb:>9.3f}  {cf:>10.3f}  "
              f"{eb:>9.3f}  {ef:>10.3f}")
        rows.append({"Study": short, "N": n,
                     "Entropy": h,
                     "CPSL_4class": c4,
                     "CPSL_2class_BAC": cb,
                     "CPSL_2class_FUNG": cf,
                     "BD_evidence_bac": eb,
                     "BD_evidence_fung": ef})

    df_out = pd.DataFrame(rows)
    # Sort by study order
    df_out["order"] = df_out["Study"].map(
        {s: i for i, s in enumerate(STUDY_ORDER)}).fillna(99)
    df_out = df_out.sort_values("order").drop("order", axis=1)
    df_out.to_csv(os.path.join(args.out_dir, "site_uncertainty.csv"), index=False)
    print(f"\n  Saved: site_uncertainty.csv")

    # ── Plot ──────────────────────────────────────────────────────────────────
    sorted_studies = [s for s in STUDY_ORDER if s in df_out["Study"].values]
    df_plot = df_out.set_index("Study").reindex(sorted_studies)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Per-Study Uncertainty — Soft Label Models Only\n"
        "MADURAI (mixed etiology) should show highest uncertainty across all three signals",
        fontsize=13
    )

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_studies)))

    # Entropy
    ax = axes[0]
    bars = ax.bar(sorted_studies, df_plot["Entropy"], color=colors)
    ax.set_title("Sigmoid Entropy\n(higher = more uncertain)", fontsize=11)
    ax.set_ylabel("Mean Entropy"); ax.set_xlabel("Study")
    ax.tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, df_plot["Entropy"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # CPSL — 4-class and 2-class side by side
    ax = axes[1]
    x  = np.arange(len(sorted_studies))
    w  = 0.28
    ax.bar(x - w, df_plot["CPSL_4class"],    w, label="4-class",  color="#4A5568", alpha=0.85)
    ax.bar(x,     df_plot["CPSL_2class_BAC"],w, label="2-class BAC",  color="#2D9E6B", alpha=0.85)
    ax.bar(x + w, df_plot["CPSL_2class_FUNG"],w,label="2-class FUNG", color="#1B8A8F", alpha=0.85)
    ax.set_title("Mean Conformal CPSL\n(4-class and 2-class per pathogen)", fontsize=11)
    ax.set_ylabel("Mean CPSL"); ax.set_xlabel("Study")
    ax.set_xticks(x); ax.set_xticklabels(sorted_studies, rotation=30, ha="right")
    ax.axhline(1.0, color="grey", ls=":", alpha=0.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # BD evidence — two bars per study
    ax = axes[2]
    x  = np.arange(len(sorted_studies))
    w  = 0.35
    ax.bar(x - w/2, df_plot["BD_evidence_bac"],  w, label="BAC (α+β)", color="#2D9E6B", alpha=0.85)
    ax.bar(x + w/2, df_plot["BD_evidence_fung"], w, label="FUNG (α+β)", color="#1B8A8F", alpha=0.85)
    ax.set_title("BD Evidence (α+β) per Pathogen\n(higher = model more confident)", fontsize=11)
    ax.set_ylabel("Mean α+β"); ax.set_xlabel("Study")
    ax.set_xticks(x); ax.set_xticklabels(sorted_studies, rotation=30, ha="right")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    for xi, eb, ef in zip(x, df_plot["BD_evidence_bac"], df_plot["BD_evidence_fung"]):
        ax.text(xi - w/2, eb + 0.05, f"{eb:.1f}", ha="center", va="bottom", fontsize=7)
        ax.text(xi + w/2, ef + 0.05, f"{ef:.1f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "site_uncertainty.png"), dpi=150)
    plt.close()
    print(f"  Saved: site_uncertainty.png")

    # ── Save JSON for docx insertion ──────────────────────────────────────────
    summary = df_out.set_index("Study").to_dict(orient="index")
    with open(os.path.join(args.out_dir, "site_uncertainty.json"), "w") as f:
        json.dump({k: {kk: round(float(vv), 4) for kk, vv in v.items()}
                   for k, v in summary.items()}, f, indent=2)
    print(f"  Saved: site_uncertainty.json")
    print(f"\nAll outputs: {args.out_dir}/")


if __name__ == "__main__":
    main()