"""
Paired DeLong Test for AUROC Comparisons
==========================================
Compares AUROC values across the four experimental cells using the
DeLong et al. (1988) method for correlated ROC curves.

All comparisons performed on the eval subset only (n=639),
consistent with all other analyses.

Bonferroni correction applied for 6 pairwise comparisons:
  corrected alpha = 0.05 / 6 = 0.0083

Comparisons:
  1. Cell A vs Cell B — BAC  (sigmoid soft vs BD soft)
  2. Cell A vs Cell B — FUNG
  3. Cell A vs Cell C — BAC  (soft vs hard, sigmoid)
  4. Cell A vs Cell C — FUNG
  5. Cell B vs Cell D — BAC  (soft vs hard, BD)
  6. Cell B vs Cell D — FUNG

Run:
    python src/delong_test.py \
        --cache_soft_sig  runs/sigmoid_2026-07-07_16-57-33/uncertainty_analysis/cache_sig.npz \
        --cache_soft_bd   runs/beta_dirichlet_2026-07-17_13-31-38/uncertainty_analysis/cache_bd.npz \
        --cache_hard_sig  runs/beta_dirichlet_hard_2026-08-19_20-47-35/uncertainty_analysis/cache_sig.npz \
        --cache_hard_bd   runs/beta_dirichlet_hard_2026-08-19_20-47-35/uncertainty_analysis/cache_bd.npz \
        --csv             /persist/QTIM/Active/26-0096/Corneal_Ulcer_Image_Sets/Data/corneal_ulcer_study_data_7_24_2025_lambda.csv

NOTE on cache layout:
  Cell A (sigmoid soft):  cache_sig.npz from sigmoid_2026-07-07_16-57-33/uncertainty_analysis/
  Cell B (BD soft):       cache_bd.npz  from beta_dirichlet_2026-07-17_13-31-38/uncertainty_analysis/
  Cell C (sigmoid hard):  cache_sig.npz from beta_dirichlet_hard_2026-08-19_20-47-35/uncertainty_analysis/
                          (dirichlet_analysis.py saves a cache_sig.npz alongside cache_bd.npz;
                           use the hard sigmoid run's cache, or re-run dirichlet_analysis.py
                           pointed at sigmoid_hard_2026-08-19_14-24-09/ weights)
  Cell D (BD hard):       cache_bd.npz  from beta_dirichlet_hard_2026-08-19_20-47-35/uncertainty_analysis/
"""

import os, json, argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupShuffleSplit


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_soft_sig", type=str, required=True,
        help="cache_sig.npz for Cell A (sigmoid + soft labels)")
    p.add_argument("--cache_soft_bd", type=str, required=True,
        help="cache_bd.npz for Cell B (BD + soft labels)")
    p.add_argument("--cache_hard_sig", type=str, required=True,
        help="cache_sig.npz for Cell C (sigmoid + hard labels)")
    p.add_argument("--cache_hard_bd", type=str, required=True,
        help="cache_bd.npz for Cell D (BD + hard labels)")
    p.add_argument("--csv", type=str,
        default="/persist/QTIM/Active/26-0096/Corneal_Ulcer_Image_Sets/Data/"
                "corneal_ulcer_study_data_7_24_2025_lambda.csv")
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.05,
        help="Nominal alpha before Bonferroni correction (default 0.05)")
    p.add_argument("--out_dir", type=str, default=None,
        help="Output directory (default: directory of --cache_soft_sig)")
    return p.parse_args()


# =============================================================================
# DELONG TEST — DeLong et al. (1988)
# =============================================================================

def _placement_values(y_true, y_score):
    """
    Compute placement values V10 and V01 for the DeLong estimator.

    V10[i] = P(score of negative < score of positive i)  (fraction of negatives beaten)
    V01[j] = P(score of positive > score of negative j)  (fraction of positives beaten)

    Ties contribute 0.5 (Mann-Whitney convention).
    AUC = mean(V10) = mean(V01).
    """
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n1, n0 = len(pos), len(neg)

    V10 = np.empty(n1)
    for i, p in enumerate(pos):
        V10[i] = np.mean(neg < p) + 0.5 * np.mean(neg == p)

    V01 = np.empty(n0)
    for j, n in enumerate(neg):
        V01[j] = np.mean(pos > n) + 0.5 * np.mean(pos == n)

    return V10, V01


def delong_paired(y_true, score1, score2):
    """
    Paired DeLong test for two correlated AUROCs evaluated on the same samples.

    Parameters
    ----------
    y_true  : (N,) binary array, 1 = positive class
    score1  : (N,) predicted scores from model 1
    score2  : (N,) predicted scores from model 2

    Returns
    -------
    auc1, auc2 : floats
    z          : z-statistic for AUC1 - AUC2
    p_value    : two-sided p-value
    """
    V10_1, V01_1 = _placement_values(y_true, score1)
    V10_2, V01_2 = _placement_values(y_true, score2)

    auc1 = V10_1.mean()
    auc2 = V10_2.mean()

    n1 = (y_true == 1).sum()   # positives
    n0 = (y_true == 0).sum()   # negatives

    # 2×2 structural-component covariance matrices
    # S10: covariance of placement values among positive examples
    # S01: covariance of placement values among negative examples
    mat_10 = np.vstack([V10_1, V10_2])   # shape (2, n1)
    mat_01 = np.vstack([V01_1, V01_2])   # shape (2, n0)

    # Use ddof=1 for unbiased estimate; np.cov returns 2×2
    if n1 > 1:
        S10 = np.cov(mat_10, ddof=1)
    else:
        S10 = np.zeros((2, 2))

    if n0 > 1:
        S01 = np.cov(mat_01, ddof=1)
    else:
        S01 = np.zeros((2, 2))

    # Variance of (AUC1 - AUC2) = Var(AUC1) + Var(AUC2) - 2·Cov(AUC1,AUC2)
    # Each term: (1/n1)·S10[i,j] + (1/n0)·S01[i,j]
    var_diff = (
        (S10[0, 0] + S10[1, 1] - 2 * S10[0, 1]) / n1 +
        (S01[0, 0] + S01[1, 1] - 2 * S01[0, 1]) / n0
    )

    if var_diff <= 0:
        # Can happen if the two score vectors are identical
        return auc1, auc2, 0.0, 1.0

    z = (auc1 - auc2) / np.sqrt(var_diff)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

    return auc1, auc2, z, p


# =============================================================================
# DATA HELPERS
# =============================================================================

def get_eval_idx(csv_path, seed=42):
    """Reproduce the exact eval split used everywhere else in this project."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["file_path", "subject"])
    df = df[(df["fung_LCA_bin"] != -1) | (df["bac_LCA_bin"] != -1)].reset_index(drop=True)

    subj = df["subject"].values

    # Train/test split (30% test)
    gss = GroupShuffleSplit(1, test_size=0.30, random_state=seed)
    _, ti = next(gss.split(df, groups=subj))
    df_t = df.iloc[ti].reset_index(drop=True)

    # Val/test split (50/50 of the 30%)
    gss2 = GroupShuffleSplit(1, test_size=0.50, random_state=seed)
    _, ti2 = next(gss2.split(df_t, groups=df_t["subject"].values))
    df_test = df_t.iloc[ti2].reset_index(drop=True)

    # Cal/eval split (20% cal, 80% eval)
    gss3 = GroupShuffleSplit(1, test_size=0.80, random_state=seed)
    cal_idx, eval_idx = next(gss3.split(df_test, groups=df_test["subject"].values))

    print(f"Test set: {len(df_test):,}  "
          f"Cal: {len(cal_idx):,}  Eval: {len(eval_idx):,}")
    return eval_idx


def bd_to_probs(params):
    """Convert BD (α_bac, β_bac, α_fung, β_fung) to mean probabilities."""
    p_bac  = params[:, 0] / (params[:, 0] + params[:, 1])
    p_fung = params[:, 2] / (params[:, 2] + params[:, 3])
    return np.stack([p_bac, p_fung], axis=1)


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = get_args()
    if args.out_dir is None:
        args.out_dir = os.path.dirname(args.cache_soft_sig)
    os.makedirs(args.out_dir, exist_ok=True)

    n_comp = 6
    bonf_alpha = args.alpha / n_comp

    print(f"\nPaired DeLong AUROC Test")
    print(f"========================")
    print(f"Nominal α = {args.alpha}  |  Bonferroni α = {bonf_alpha:.4f}  "
          f"({n_comp} comparisons)")

    # ── Eval indices ──────────────────────────────────────────────────────────
    eval_idx = get_eval_idx(args.csv, args.seed)

    # ── Load caches and extract eval-subset scores ────────────────────────────
    # Cell A — sigmoid soft
    c_A = np.load(args.cache_soft_sig)
    p_A = c_A["probs"][eval_idx]          # shape (n_eval, 2): [p_bac, p_fung]
    y   = c_A["labels"][eval_idx]         # shape (n_eval, 2): soft LCA posteriors

    # Cell B — BD soft
    c_B = np.load(args.cache_soft_bd)
    p_B = bd_to_probs(c_B["params"][eval_idx])

    # Cell C — sigmoid hard
    c_C = np.load(args.cache_hard_sig)
    p_C = c_C["probs"][eval_idx]

    # Cell D — BD hard
    c_D = np.load(args.cache_hard_bd)
    p_D = bd_to_probs(c_D["params"][eval_idx])

    # Binary labels (threshold LCA posteriors at 0.5)
    y_bac  = (y[:, 0] > 0.5).astype(int)
    y_fung = (y[:, 1] > 0.5).astype(int)

    print(f"\nLabel distribution (eval subset, n={len(eval_idx)}):")
    print(f"  BAC:  pos={y_bac.sum():,}  neg={(~y_bac.astype(bool)).sum():,}")
    print(f"  FUNG: pos={y_fung.sum():,}  neg={(~y_fung.astype(bool)).sum():,}")

    # ── Run all six comparisons ───────────────────────────────────────────────
    comparisons = [
        # (label, pathogen, score_1, score_2, y_binary, description)
        ("A vs B", "BAC",  p_A[:, 0], p_B[:, 0], y_bac,
         "Sigmoid+Soft vs BD+Soft"),
        ("A vs B", "FUNG", p_A[:, 1], p_B[:, 1], y_fung,
         "Sigmoid+Soft vs BD+Soft"),
        ("A vs C", "BAC",  p_A[:, 0], p_C[:, 0], y_bac,
         "Soft vs Hard (Sigmoid)"),
        ("A vs C", "FUNG", p_A[:, 1], p_C[:, 1], y_fung,
         "Soft vs Hard (Sigmoid)"),
        ("B vs D", "BAC",  p_B[:, 0], p_D[:, 0], y_bac,
         "Soft vs Hard (BD)"),
        ("B vs D", "FUNG", p_B[:, 1], p_D[:, 1], y_fung,
         "Soft vs Hard (BD)"),
    ]

    header = (f"  {'Comparison':10s}  {'Path':6s}  "
              f"{'AUC1':>7}  {'AUC2':>7}  {'Δ':>7}  "
              f"{'Z':>7}  {'p-value':>10}  {'Result':>12}")
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"  {'-'*(len(header)-2)}")

    records = []
    for label, pathogen, s1, s2, y_bin, desc in comparisons:
        auc1, auc2, z, p = delong_paired(y_bin, s1, s2)

        if p < bonf_alpha:
            result = f"p<{bonf_alpha:.4f} ***"
        elif p < args.alpha:
            result = f"p<{args.alpha} *"
        else:
            result = "ns"

        print(f"  {label:10s}  {pathogen:6s}  "
              f"{auc1:.4f}  {auc2:.4f}  {auc1-auc2:+.4f}  "
              f"{z:+.4f}  {p:10.4f}  {result:>12}")

        records.append({
            "comparison": label,
            "pathogen": pathogen,
            "description": desc,
            "auc1": round(float(auc1), 4),
            "auc2": round(float(auc2), 4),
            "delta_auc": round(float(auc1 - auc2), 4),
            "z_stat": round(float(z), 4),
            "p_value": float(p),
            "bonferroni_significant": bool(p < bonf_alpha),
            "uncorrected_significant": bool(p < args.alpha),
        })

    print(f"\n  *** Bonferroni-corrected significant (p < {bonf_alpha:.4f})")
    print(f"  *   Uncorrected significant (p < {args.alpha})")
    print(f"  ns  Not significant")

    # ── Save outputs ──────────────────────────────────────────────────────────
    summary = {
        "n_eval": int(len(eval_idx)),
        "n_comparisons": n_comp,
        "nominal_alpha": args.alpha,
        "bonferroni_alpha": bonf_alpha,
        "method": "DeLong et al. (1988), two-sided, paired",
        "comparisons": records,
    }

    json_path = os.path.join(args.out_dir, "delong_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {json_path}")

    csv_path = os.path.join(args.out_dir, "delong_results.csv")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Pretty text summary for copy-paste into manuscript
    txt_path = os.path.join(args.out_dir, "delong_results_summary.txt")
    with open(txt_path, "w") as f:
        f.write("Paired DeLong AUROC Test Results\n")
        f.write("="*60 + "\n")
        f.write(f"Method: DeLong et al. (1988), two-sided\n")
        f.write(f"Eval subset: n={len(eval_idx)}\n")
        f.write(f"Comparisons: {n_comp} pairs\n")
        f.write(f"Bonferroni-corrected alpha: {bonf_alpha:.4f}\n\n")
        for r in records:
            sig = "***" if r["bonferroni_significant"] else ("*" if r["uncorrected_significant"] else "ns")
            f.write(f"Cell {r['comparison']} ({r['pathogen']}): "
                    f"AUC={r['auc1']:.3f} vs {r['auc2']:.3f}, "
                    f"Δ={r['delta_auc']:+.3f}, "
                    f"Z={r['z_stat']:+.3f}, p={r['p_value']:.4f} [{sig}]\n")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()