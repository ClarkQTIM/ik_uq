"""
Native Sigmoid UQ Analysis
===========================
Extracts uncertainty signals directly from sigmoid model outputs —
no calibration set, no BD model, no extra machinery.

Three score functions:
    Entropy of 4-class factored joint distribution:
        H = -sum_k P(k) log P(k)
        where P(k) = factored joint from p_bac, p_fung
        High entropy = uncertain (distribution spread across classes)

    Margin (difference between top two class probabilities):
        margin = P(1st) - P(2nd)
        Low margin = uncertain (top two classes nearly tied)

    Max probability:
        max_p = max(P(neither), P(bac-only), P(fung-only), P(both))
        Low max_p = uncertain

All three are computed from the same [p_bac, p_fung] sigmoid outputs.
No calibration needed — purely descriptive UQ.

Key analyses:
    1. Do these signals correlate with accuracy?
    2. Do they correlate with conformal CPSL? (cross-framework check)
    3. Do they vary by study? (MADURAI should be most uncertain)
    4. Cell A (soft) vs Cell C (hard) — do soft labels produce
       more informative entropy signals?

Uses inference caches from conformal.py and conformal_regression.py
to avoid redundant forward passes.

Run:
    python sigmoid_uq.py
        --cache_a  runs/sigmoid_soft_.../conformal/cache.npz
        --cache_c  runs/sigmoid_hard_.../conformal/cache.npz   (optional)
        --cpsl_csv runs/sigmoid_soft_.../conformal/accuracy_vs_set_size.csv
        --csv      /path/to/data.csv
        --alpha    0.1
"""

import os, json, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# ARGUMENTS
# =============================================================================

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str,
        default="/persist/QTIM/Active/26-0096/Corneal_Ulcer_Image_Sets/Data/"
                "corneal_ulcer_study_data_7_24_2025_lambda.csv")
    p.add_argument("--cache_a", type=str, required=True,
        help="Path to cache_sig.npz from dirichlet_analysis run — "
             "built on full 785-row test set. "
             "(runs/beta_dirichlet_soft_.../uncertainty_analysis/cache_sig.npz)")
    p.add_argument("--cache_c", type=str, default=None,
        help="Path to sigmoid hard cache — must also be on full 785-row test set. "
             "Run dirichlet_analysis.py with Cell D and Cell C weights to generate.")
    p.add_argument("--alpha",   type=float, default=0.1)
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--out_dir", type=str,   default=None)
    return p.parse_args()


# =============================================================================
# DATA HELPERS
# =============================================================================

PACS_PREFIX = "/pacs/OSHU/Corneal_Ulcer_Image_Sets"
STUDY_SHORT = {
    "MADURAI photos":"MADURAI",   "MUTT2-Pictures":"MUTT2",
    "MUTT1-Pictures":"MUTT1",     "SCUT photos":"SCUT",
    "MUTT Pilot Photos":"MUTT Pilot",
    "CLAIR II photos":"CLAIR II", "CLAIR I photos":"CLAIR I",
}

def load_test_set(csv_path, seed=42):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["file_path", "subject"])
    df = df[(df["fung_LCA_bin"]!=-1)|(df["bac_LCA_bin"]!=-1)].reset_index(drop=True)
    subj = df["subject"].values
    gss = GroupShuffleSplit(1, test_size=0.30, random_state=seed)
    _, ti = next(gss.split(df, groups=subj))
    df_t = df.iloc[ti].reset_index(drop=True)
    gss2 = GroupShuffleSplit(1, test_size=0.50, random_state=seed)
    _, ti2 = next(gss2.split(df_t, groups=df_t["subject"].values))
    return df_t.iloc[ti2].reset_index(drop=True)


# =============================================================================
# UQ SCORE FUNCTIONS
# =============================================================================

def factored_joint(probs):
    """
    Construct 4-class factored joint distribution from [p_bac, p_fung].
    P(neither)  = (1-p_bac) * (1-p_fung)
    P(bac-only) = p_bac     * (1-p_fung)
    P(fung-only)= (1-p_bac) * p_fung
    P(both)     = p_bac     * p_fung
    Returns shape (N, 4), rows sum to 1.
    """
    b, f = probs[:, 0], probs[:, 1]
    return np.stack([
        (1-b)*(1-f),   # 0: neither
        b    *(1-f),   # 1: bac-only
        (1-b)*f,       # 2: fung-only
        b    *f,       # 3: both
    ], axis=1)

def entropy(p4):
    """Shannon entropy of 4-class distribution. Higher = more uncertain."""
    eps = 1e-12
    return -(p4 * np.log(p4 + eps)).sum(axis=1)

def margin(p4):
    """Difference between top-2 class probabilities. Lower = more uncertain."""
    sorted_p = np.sort(p4, axis=1)[:, ::-1]
    return sorted_p[:, 0] - sorted_p[:, 1]

def max_prob(p4):
    """Maximum class probability. Lower = more uncertain."""
    return p4.max(axis=1)

def compute_uq(probs):
    """Compute all three UQ signals from [p_bac, p_fung]."""
    p4 = factored_joint(probs)
    return {
        "entropy":  entropy(p4),
        "margin":   margin(p4),
        "max_prob": max_prob(p4),
        "p4":       p4,
        "pred_class": p4.argmax(axis=1),
    }

def get_labels_4class(df):
    """Binarized LCA posteriors -> 4-class integer label."""
    bac_bin  = (df["bac_post_prob"]  > 0.5).astype(int).values
    fung_bin = (df["fung_post_prob"] > 0.5).astype(int).values
    return bac_bin + fung_bin * 2

def get_cpsl(probs, df, alpha=0.1, seed=42):
    """
    Compute conformal set sizes (4-class Vanilla LAC) on the eval subset.
    Uses 20/80 cal/eval split — same as conformal.py.
    Returns CPSL values for eval subset and eval indices.
    """
    subj = df["subject"].values
    gss  = GroupShuffleSplit(1, test_size=0.80, random_state=seed)
    cal_idx, eval_idx = next(gss.split(df, groups=subj))

    lab4    = get_labels_4class(df)
    p4      = factored_joint(probs)
    n       = len(cal_idx)
    scores  = 1 - p4[cal_idx][np.arange(n), lab4[cal_idx]]
    q_level = min(np.ceil((n+1)*(1-alpha))/n, 1.0)
    qhat    = float(np.quantile(scores, q_level, method="higher"))
    psets   = (p4[eval_idx] >= (1-qhat)).sum(axis=1)
    return psets, eval_idx, qhat


# =============================================================================
# ANALYSES
# =============================================================================

def phdr(t): print(f"\n{'='*65}\n  {t}\n{'='*65}")

def analysis_accuracy_vs_uq(uq, correct, name, out_dir):
    """
    Do UQ signals predict accuracy?
    Bin by each signal and compute accuracy per bin.
    """
    phdr(f"ACCURACY vs. UQ SIGNALS — {name}")
    bins = [0, 20, 40, 60, 80, 100]
    blabels = [f"P{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]

    results = {}
    for signal_name, values, ascending in [
        ("Entropy",  uq["entropy"],  True),   # high entropy = uncertain
        ("Margin",   uq["margin"],   False),  # low margin = uncertain
        ("Max prob", uq["max_prob"], False),  # low max_p = uncertain
    ]:
        bin_edges = [np.percentile(values, b) for b in bins]
        bidx = np.clip(np.digitize(values, bin_edges[1:-1]), 0, len(blabels)-1)
        accs  = [correct[bidx==b].mean() if (bidx==b).sum()>0 else np.nan
                 for b in range(len(blabels))]
        meds  = [np.median(values[bidx==b]) for b in range(len(blabels))]
        ns    = [(bidx==b).sum() for b in range(len(blabels))]

        rho, p_val = spearmanr(values, correct.astype(float))
        # For entropy: expect negative rho (more uncertain -> less accurate)
        # For margin/max_prob: expect positive rho (more confident -> more accurate)
        print(f"\n  {signal_name}: Spearman rho={rho:.3f}, p={p_val:.4f}")
        for bl, acc, n, med in zip(blabels, accs, ns, meds):
            if not np.isnan(acc):
                print(f"    {bl}  median={med:.3f}  acc={acc*100:.1f}%  n={n}")
        results[signal_name] = {"rho": float(rho), "p": float(p_val),
                                  "accs": accs, "meds": meds}

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Accuracy vs. Native Sigmoid UQ Signals — {name}", fontsize=13)
    for ax, (signal_name, values, ascending) in zip(axes, [
        ("Entropy",  uq["entropy"],  True),
        ("Margin",   uq["margin"],   False),
        ("Max prob", uq["max_prob"], False),
    ]):
        r = results[signal_name]
        ax.plot(r["meds"], r["accs"], marker="o", linewidth=2, color="#1B8A8F")
        ax.set_title(f"{signal_name}\nSpearman rho={r['rho']:.3f}, p={r['p']:.4f}")
        ax.set_xlabel(f"Median {signal_name} in Bin")
        ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
        ax.axhline(0.5, color="grey", ls=":", alpha=0.5)

    plt.tight_layout()
    fname = f"sigmoid_uq_vs_accuracy_{name.lower().replace(' ','_')}.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()
    print(f"\n  Saved: {fname}")
    return results


def analysis_uq_vs_cpsl(uq_eval, cpsl, name, out_dir):
    """
    Do native sigmoid UQ signals correlate with conformal set size?
    Both should capture the same uncertainty — if they agree,
    the signals are measuring the same underlying difficulty.
    """
    phdr(f"NATIVE UQ vs. CONFORMAL CPSL — {name}")
    rows = []
    for signal_name, values in [
        ("Entropy",  uq_eval["entropy"]),
        ("Margin",   uq_eval["margin"]),
        ("Max prob", uq_eval["max_prob"]),
    ]:
        rho, p_val = spearmanr(cpsl, values)
        direction = "negative ✓" if rho < 0 else "positive"
        print(f"  {signal_name:10s}: rho={rho:.3f}  p={p_val:.4f}  "
              f"({'Expected negative' if signal_name=='Entropy' else 'Expected positive'}: {direction})")
        rows.append({"Signal": signal_name, "rho": rho, "p": p_val})

    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, f"uq_cpsl_correlation_{name.lower().replace(' ','_')}.csv"),
        index=False)


def analysis_uq_by_study(uq, df_eval, name, out_dir):
    """
    Do UQ signals vary by study?
    MADURAI (mixed etiology) should have highest entropy.
    Single-etiology studies should have lowest entropy.
    """
    phdr(f"NATIVE UQ BY STUDY — {name}")
    studies = df_eval["study"].values
    rows = []

    print(f"  {'Study':12s}  {'Entropy':>10}  {'Margin':>10}  {'Max prob':>10}  {'N':>6}")
    print(f"  {'-'*55}")
    for study in sorted(np.unique(studies)):
        mask  = studies == study
        short = STUDY_SHORT.get(study, study)
        h_mean = uq["entropy"][mask].mean()
        m_mean = uq["margin"][mask].mean()
        p_mean = uq["max_prob"][mask].mean()
        n      = mask.sum()
        print(f"  {short:12s}  {h_mean:10.4f}  {m_mean:10.4f}  {p_mean:10.4f}  {n:>6}")
        rows.append({"Model": name, "Study": short,
                     "Entropy": h_mean, "Margin": m_mean, "Max_prob": p_mean, "N": n})

    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, f"uq_by_study_{name.lower().replace(' ','_')}.csv"),
        index=False)


def analysis_soft_vs_hard(uq_a, uq_c, correct_a, correct_c, out_dir):
    """
    Cell A (soft) vs Cell C (hard): are soft label entropy signals more informative?
    Key question: does soft training produce entropy that better predicts accuracy?
    """
    phdr("CELL A (SOFT) vs CELL C (HARD) — UQ SIGNAL QUALITY")
    print(f"  {'Signal':12s}  {'Cell A rho':>12}  {'Cell C rho':>12}  {'Diff':>8}")
    print(f"  {'-'*50}")
    rows = []
    for signal_name, val_a, val_c in [
        ("Entropy",  uq_a["entropy"],  uq_c["entropy"]),
        ("Margin",   uq_a["margin"],   uq_c["margin"]),
        ("Max prob", uq_a["max_prob"], uq_c["max_prob"]),
    ]:
        rho_a, _ = spearmanr(val_a, correct_a.astype(float))
        rho_c, _ = spearmanr(val_c, correct_c.astype(float))
        diff = rho_a - rho_c
        better = "A better ✓" if abs(rho_a) > abs(rho_c) else "C better"
        print(f"  {signal_name:12s}  {rho_a:>+12.3f}  {rho_c:>+12.3f}  "
              f"{diff:>+8.3f}  {better}")
        rows.append({"Signal": signal_name, "rho_A": rho_a, "rho_C": rho_c,
                     "Diff": diff})
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "uq_soft_vs_hard.csv"), index=False)

    # Distribution comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Native UQ Distribution: Cell A (Soft) vs Cell C (Hard)\n"
                 "Soft labels should produce more spread entropy", fontsize=12)
    colors = {"Cell A (Soft)": "#2D9E6B", "Cell C (Hard)": "#E05252"}
    for ax, (signal_name, val_a, val_c) in zip(axes, [
        ("Entropy",  uq_a["entropy"],  uq_c["entropy"]),
        ("Margin",   uq_a["margin"],   uq_c["margin"]),
        ("Max prob", uq_a["max_prob"], uq_c["max_prob"]),
    ]):
        bins = np.linspace(min(val_a.min(), val_c.min()),
                            max(val_a.max(), val_c.max()), 35)
        ax.hist(val_a, bins=bins, alpha=0.6, color=colors["Cell A (Soft)"],
                label="Cell A (Soft)", density=True)
        ax.hist(val_c, bins=bins, alpha=0.6, color=colors["Cell C (Hard)"],
                label="Cell C (Hard)", density=True)
        ax.set_title(signal_name); ax.set_xlabel(signal_name)
        ax.set_ylabel("Density"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "uq_soft_vs_hard_distributions.png"), dpi=150)
    plt.close()
    print("  Saved: uq_soft_vs_hard_distributions.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = get_args()
    np.random.seed(args.seed)

    if args.out_dir is None:
        args.out_dir = os.path.join(
            os.path.dirname(args.cache_a), "sigmoid_uq")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output directory: {args.out_dir}\n")

    # ── Load test set ─────────────────────────────────────────────────────────
    df_test = load_test_set(args.csv, args.seed)
    print(f"Test set: {len(df_test):,} images\n")

    # ── Load Cell A cache ─────────────────────────────────────────────────────
    print("Loading Cell A cache...")
    c_a = np.load(args.cache_a)
    # conformal_regression caches save key 'probs' for sigmoid models
    probs_a  = c_a["probs"]
    labels_a = c_a["labels"]
    print(f"  Cell A: {probs_a.shape[0]:,} samples")

    # ── Compute UQ signals ────────────────────────────────────────────────────
    uq_a  = compute_uq(probs_a)
    lab4_a = get_labels_4class(df_test)
    correct_a = (uq_a["pred_class"] == lab4_a)
    print(f"  Cell A accuracy: {correct_a.mean()*100:.1f}%")

    # Eval subset (for CPSL comparison)
    cpsl_a, eval_idx, qhat = get_cpsl(probs_a, df_test, args.alpha, args.seed)
    df_eval = df_test.iloc[eval_idx].reset_index(drop=True)
    uq_a_eval = {k: v[eval_idx] if isinstance(v, np.ndarray) and len(v)==len(df_test)
                 else v for k, v in uq_a.items()}
    correct_a_eval = correct_a[eval_idx]
    print(f"  Conformal qhat: {qhat:.4f}  mean CPSL: {cpsl_a.mean():.3f}")

    # ── Cell A analyses ───────────────────────────────────────────────────────
    res_a = analysis_accuracy_vs_uq(uq_a_eval, correct_a_eval, "Cell A Soft", args.out_dir)
    analysis_uq_vs_cpsl(uq_a_eval, cpsl_a, "Cell A Soft", args.out_dir)
    analysis_uq_by_study(uq_a_eval, df_eval, "Cell A Soft", args.out_dir)

    # ── Cell C (optional) ─────────────────────────────────────────────────────
    if args.cache_c:
        print("\nLoading Cell C cache...")
        c_c = np.load(args.cache_c)
        probs_c  = c_c["probs"]
        print(f"  Cell C: {probs_c.shape[0]:,} samples")

        uq_c      = compute_uq(probs_c)
        lab4_c    = get_labels_4class(df_test)
        correct_c = (uq_c["pred_class"] == lab4_c)
        print(f"  Cell C accuracy: {correct_c.mean()*100:.1f}%")

        cpsl_c, _, _ = get_cpsl(probs_c, df_test, args.alpha, args.seed)
        uq_c_eval = {k: v[eval_idx] if isinstance(v, np.ndarray) and len(v)==len(df_test)
                     else v for k, v in uq_c.items()}
        correct_c_eval = correct_c[eval_idx]

        res_c = analysis_accuracy_vs_uq(uq_c_eval, correct_c_eval, "Cell C Hard", args.out_dir)
        analysis_uq_vs_cpsl(uq_c_eval, cpsl_c, "Cell C Hard", args.out_dir)
        analysis_uq_by_study(uq_c_eval, df_eval, "Cell C Hard", args.out_dir)
        analysis_soft_vs_hard(uq_a_eval, uq_c_eval,
                               correct_a_eval, correct_c_eval, args.out_dir)

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = {"Cell A": res_a}
    if args.cache_c:
        summary["Cell C"] = res_c
    def serialize(obj):
        if isinstance(obj, dict): return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, list): return [serialize(v) for v in obj]
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.ndarray): return [serialize(v) for v in obj.tolist()]
        if isinstance(obj, float) and obj != obj: return None  # NaN
        return obj

    with open(os.path.join(args.out_dir, "sigmoid_uq_results.json"), "w") as f:
        json.dump(serialize(summary), f, indent=2)

    print(f"\nAll outputs saved to: {args.out_dir}/")
    print("  sigmoid_uq_results.json")
    print("  sigmoid_uq_vs_accuracy_*.png")
    print("  uq_cpsl_correlation_*.csv")
    print("  uq_by_study_*.csv")
    if args.cache_c:
        print("  uq_soft_vs_hard.csv")
        print("  uq_soft_vs_hard_distributions.png")


if __name__ == "__main__":
    main()