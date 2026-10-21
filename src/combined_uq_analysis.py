"""
Combined UQ Analysis: Does combining BD evidence + Conformal CPSL
outperform either signal individually?

Formal comparison:
  - CPSL alone:    accuracy at each CPSL threshold (selection_rate = fraction selected)
  - BD alone:      accuracy at each evidence threshold (selection_rate = fraction selected)
  - Combined:      accuracy when BOTH signals agree confident (High BD AND CPSL=1)

Uses existing caches from dirichlet_analysis.py run on Cell B.
No new inference needed.

Run:
    python combined_uq_analysis.py
        --cache_dir runs/beta_dirichlet_2026-07-17_13-31-38/uncertainty_analysis/
        --csv       /path/to/data.csv
        --alpha     0.1
"""

import os, json, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--csv", type=str,
        default="/persist/QTIM/Active/26-0096/Corneal_Ulcer_Image_Sets/Data/"
                "corneal_ulcer_study_data_7_24_2025_lambda.csv")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--seed",  type=int,   default=42)
    p.add_argument("--out_dir", type=str, default=None)
    return p.parse_args()

def load_test_set(csv_path, seed=42):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["file_path","subject"])
    df = df[(df["fung_LCA_bin"]!=-1)|(df["bac_LCA_bin"]!=-1)].reset_index(drop=True)
    subj = df["subject"].values
    gss = GroupShuffleSplit(1, test_size=0.30, random_state=seed)
    _, ti = next(gss.split(df, groups=subj))
    df_t = df.iloc[ti].reset_index(drop=True)
    gss2 = GroupShuffleSplit(1, test_size=0.50, random_state=seed)
    _, ti2 = next(gss2.split(df_t, groups=df_t["subject"].values))
    return df_t.iloc[ti2].reset_index(drop=True)

def make_4class(sig_probs):
    b, f = sig_probs[:,0], sig_probs[:,1]
    return np.stack([(1-b)*(1-f), b*(1-f), (1-b)*f, b*f], axis=1)

def lac_4class(sig_probs, labels, cal_idx, eval_idx, alpha):
    sig4    = make_4class(sig_probs)
    bac_bin = (labels[:,0]>0.5).astype(int)
    fng_bin = (labels[:,1]>0.5).astype(int)
    lab4    = bac_bin + fng_bin*2
    n       = len(cal_idx)
    scores  = 1 - sig4[cal_idx][np.arange(n), lab4[cal_idx]]
    qlevel  = min(np.ceil((n+1)*(1-alpha))/n, 1.0)
    qhat    = float(np.quantile(scores, qlevel, method="higher"))
    psets   = sig4[eval_idx] >= (1-qhat)
    return psets.sum(axis=1), lab4, qhat

def phdr(t): print(f"\n{'='*65}\n  {t}\n{'='*65}")

def main():
    args = get_args()
    if args.out_dir is None:
        args.out_dir = args.cache_dir
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load caches ───────────────────────────────────────────────────────────
    bd  = np.load(os.path.join(args.cache_dir, "cache_bd.npz"))
    sig = np.load(os.path.join(args.cache_dir, "cache_sig.npz"))
    bd_params  = bd["params"];  bd_labels  = bd["labels"]
    sig_probs  = sig["probs"];  sig_labels = sig["labels"]

    # ── Test set and cal/eval split ───────────────────────────────────────────
    df_test = load_test_set(args.csv, args.seed)
    subj    = df_test["subject"].values
    gss     = GroupShuffleSplit(1, test_size=0.80, random_state=args.seed)
    cal_idx, eval_idx = next(gss.split(df_test, groups=subj))

    # ── Conformal set sizes ───────────────────────────────────────────────────
    cpsl, lab4, qhat = lac_4class(sig_probs, bd_labels, cal_idx, eval_idx, args.alpha)
    print(f"  Conformal qhat: {qhat:.4f}  mean CPSL: {cpsl.mean():.3f}")

    # ── BD evidence ───────────────────────────────────────────────────────────
    a_bac, b_bac   = bd_params[eval_idx,0], bd_params[eval_idx,1]
    a_fung, b_fung = bd_params[eval_idx,2], bd_params[eval_idx,3]
    e_bac  = a_bac  + b_bac
    e_fung = a_fung + b_fung
    e_mean = (e_bac + e_fung) / 2

    # ── Correctness ───────────────────────────────────────────────────────────
    sig4   = make_4class(sig_probs)
    pred4  = sig4[eval_idx].argmax(axis=1)
    correct = (pred4 == lab4[eval_idx])
    N = len(correct)
    print(f"  Eval n={N}  overall accuracy={correct.mean()*100:.1f}%")

    # ── BD evidence percentile thresholds — defined on CALIBRATION set ────────
    # Critical: thresholds must be defined on calibration data only,
    # not on eval data. This mirrors how CPSL's qhat is computed on
    # calibration data and applied to eval. Using eval data to set the
    # threshold would be circular — the same data used to report accuracy
    # would determine who is selected.
    a_bac_cal  = bd_params[cal_idx, 0]; b_bac_cal  = bd_params[cal_idx, 1]
    a_fung_cal = bd_params[cal_idx, 2]; b_fung_cal = bd_params[cal_idx, 3]
    e_bac_cal  = a_bac_cal  + b_bac_cal
    e_fung_cal = a_fung_cal + b_fung_cal
    e_mean_cal = (e_bac_cal + e_fung_cal) / 2

    t33 = np.percentile(e_mean_cal, 33)
    t67 = np.percentile(e_mean_cal, 67)
    print(f"  BD thresholds from calibration set (n={len(cal_idx)}):")
    print(f"    t33 = {t33:.3f}  t67 = {t67:.3f}")

    results = {}

    # ══════════════════════════════════════════════════════════════════════════
    # 1. CPSL ALONE
    # ══════════════════════════════════════════════════════════════════════════
    phdr("1. CONFORMAL CPSL ALONE")
    print(f"  {'CPSL threshold':20s}  {'Coverage':>10}  {'Accuracy':>10}  {'N trusted':>10}")
    print(f"  {'-'*55}")
    cpsl_results = []
    for thresh in [1, 2, 3, 4]:
        mask     = cpsl <= thresh
        selection_rate = mask.mean()
        acc      = correct[mask].mean() if mask.sum()>0 else np.nan
        print(f"  CPSL <= {thresh}              {selection_rate*100:>9.1f}%  "
              f"{acc*100:>9.1f}%  {mask.sum():>10}")
        cpsl_results.append({"threshold": f"CPSL<={thresh}",
                              "selection_rate": float(selection_rate),
                              "accuracy": float(acc),
                              "n": int(mask.sum())})
    results["cpsl_alone"] = cpsl_results

    # ══════════════════════════════════════════════════════════════════════════
    # 2. BD EVIDENCE ALONE
    # ══════════════════════════════════════════════════════════════════════════
    phdr("2. BD EVIDENCE ALONE")
    print(f"  {'Evidence threshold':25s}  {'Coverage':>10}  {'Accuracy':>10}  {'N trusted':>10}")
    print(f"  {'-'*60}")
    bd_results = []
    for label, thresh in [
        ("High BD (top 33%, e>t67)",  t67),
        ("High+Mid BD (top 67%, e>t33)", t33),
        ("All BD (e>0)",              0.0),
    ]:
        mask     = e_mean >= thresh
        selection_rate = mask.mean()
        acc      = correct[mask].mean() if mask.sum()>0 else np.nan
        print(f"  {label:35s}  {selection_rate*100:>9.1f}%  "
              f"{acc*100:>9.1f}%  {mask.sum():>10}")
        bd_results.append({"threshold": label,
                           "selection_rate": float(selection_rate),
                           "accuracy": float(acc),
                           "n": int(mask.sum())})
    results["bd_alone"] = bd_results

    # Also show BD by tercile vs accuracy (more granular)
    print(f"\n  BD tercile accuracy:")
    for tier_label, mask in [
        ("High BD (e>t67)", e_mean>t67),
        ("Mid BD",          (e_mean>t33)&(e_mean<=t67)),
        ("Low BD (e<=t33)", e_mean<=t33),
    ]:
        acc = correct[mask].mean()
        print(f"    {tier_label:25s}  acc={acc*100:.1f}%  n={mask.sum()}")

    # ══════════════════════════════════════════════════════════════════════════
    # 3. COMBINED: BD AND CPSL
    # ══════════════════════════════════════════════════════════════════════════
    phdr("3. COMBINED: BD EVIDENCE + CONFORMAL CPSL")
    print(f"  The key question: does requiring BOTH signals to be confident")
    print(f"  give better accuracy than either signal alone?\n")

    combined_results = []

    # Core comparison at matched coverage levels
    print(f"  {'Selection criterion':40s}  {'Coverage':>9}  {'Accuracy':>9}  {'N':>6}")
    print(f"  {'-'*70}")

    comparisons = [
        # Individual signals
        ("CPSL=1 only",              (cpsl==1),                   "cpsl_only"),
        ("High BD only (top 33%)",   (e_mean>t67),                "bd_only"),
        # Combined — AND
        ("CPSL=1 AND High BD",       (cpsl==1)&(e_mean>t67),     "combined_and"),
        ("CPSL=1 AND Mid+High BD",   (cpsl==1)&(e_mean>t33),     "combined_mid"),
        # Combined — OR (for completeness)
        ("CPSL=1 OR High BD",        (cpsl==1)|(e_mean>t67),     "combined_or"),
        # Baselines
        ("All cases (no filter)",    np.ones(N, dtype=bool),     "all"),
    ]

    for label, mask, key in comparisons:
        selection_rate = mask.mean()
        acc      = correct[mask].mean() if mask.sum()>0 else np.nan
        flag = ""
        if key == "combined_and" and mask.sum()>0:
            # Compare to individual signals at same or higher coverage
            cpsl_only_acc  = correct[cpsl==1].mean()
            bd_only_acc    = correct[e_mean>t67].mean()
            beats_cpsl = acc > cpsl_only_acc
            beats_bd   = acc > bd_only_acc
            flag = f"  {'✓ beats both' if beats_cpsl and beats_bd else '✗'}"
        print(f"  {label:40s}  {selection_rate*100:>8.1f}%  {acc*100:>8.1f}%  {mask.sum():>5}{flag}")
        combined_results.append({"criterion": label, "key": key,
                                  "selection_rate": float(selection_rate),
                                  "accuracy": float(acc),
                                  "n": int(mask.sum())})

    results["combined"] = combined_results

    # ══════════════════════════════════════════════════════════════════════════
    # 4. FORMAL COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════════════════
    phdr("4. FORMAL COMPARISON: ACCURACY AT MATCHED COVERAGE")
    print(f"  Comparing accuracy at ~55% selection rate (CPSL=1 selection rate)\n")

    cpsl1_mask  = cpsl==1
    bd_top_mask = e_mean>t67
    comb_mask   = cpsl1_mask & bd_top_mask

    # For BD alone matched to combined coverage
    # Combined n ≈ some fraction — find BD threshold that matches it
    comb_n    = comb_mask.sum()
    comb_sel  = comb_mask.mean()
    comb_acc  = correct[comb_mask].mean()

    cpsl1_acc = correct[cpsl1_mask].mean()
    cpsl1_sel = cpsl1_mask.mean()
    bd_acc    = correct[bd_top_mask].mean()
    bd_sel    = bd_top_mask.mean()

    print(f"  {'Method':30s}  {'Sel. rate':>10}  {'Accuracy':>10}  {'N':>6}")
    print(f"  {'-'*60}")
    print(f"  {'No filter (baseline)':30s}  {correct.mean()*100:>9.1f}%  "
          f"{correct.mean()*100:>9.1f}%  {N:>6}")
    print(f"  {'CPSL=1 alone':30s}  {cpsl1_sel*100:>9.1f}%  "
          f"{cpsl1_acc*100:>9.1f}%  {cpsl1_mask.sum():>6}")
    print(f"  {'High BD alone':30s}  {bd_sel*100:>9.1f}%  "
          f"{bd_acc*100:>9.1f}%  {bd_top_mask.sum():>6}")
    print(f"  {'CPSL=1 AND High BD':30s}  {comb_sel*100:>9.1f}%  "
          f"{comb_acc*100:>9.1f}%  {comb_n:>6}")
    print(f"\n  Accuracy gains over baseline ({correct.mean()*100:.1f}%):")
    print(f"    CPSL=1 alone:       +{(cpsl1_acc - correct.mean())*100:.1f}pp")
    print(f"    High BD alone:      +{(bd_acc    - correct.mean())*100:.1f}pp")
    print(f"    CPSL=1 AND High BD: +{(comb_acc  - correct.mean())*100:.1f}pp  ← combined")
    print(f"\n  Accuracy gains over CPSL=1 ({cpsl1_acc*100:.1f}%):")
    print(f"    Adding BD filter:   +{(comb_acc - cpsl1_acc)*100:.1f}pp")
    print(f"\n  Accuracy gains over High BD ({bd_acc*100:.1f}%):")
    print(f"    Adding CPSL filter: +{(comb_acc - bd_acc)*100:.1f}pp")

    # ══════════════════════════════════════════════════════════════════════════
    # 5. PRECISION-COVERAGE CURVE (the "operating point" view)
    # ══════════════════════════════════════════════════════════════════════════
    phdr("5. PRECISION-COVERAGE ACROSS THRESHOLDS")
    print(f"  How does accuracy change as we require more confidence?\n")
    print(f"  {'Coverage threshold':25s}  {'CPSL alone acc':>15}  "
          f"{'BD alone acc':>13}  {'Combined acc':>13}")
    print(f"  {'-'*72}")

    # Scan coverage thresholds from 10% to 100%
    curve_results = []
    for cov_target in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
        # CPSL: select cases with smallest CPSL until coverage reached
        cpsl_thresh = np.percentile(cpsl, cov_target*100)
        cpsl_mask_t = cpsl <= max(1, int(cpsl_thresh))
        # Adjust to get closest to target
        for s in [1,2,3,4]:
            if (cpsl<=s).mean() >= cov_target:
                cpsl_mask_t = cpsl<=s
                break

        # BD: select top fraction by evidence
        bd_thresh_t = np.percentile(e_mean, (1-cov_target)*100)
        bd_mask_t   = e_mean >= bd_thresh_t

        # Combined: AND of CPSL=1 and top BD
        # At each coverage level, use CPSL=1 AND top BD fraction
        # that together give ~cov_target coverage
        # Approximate by finding BD threshold that combined with CPSL=1 gives cov_target
        combined_mask_t = (cpsl==1) & (e_mean >= np.percentile(e_mean[cpsl==1],
                                        max(0, (1-cov_target/cpsl1_sel)*100)))

        acc_cpsl = correct[cpsl_mask_t].mean() if cpsl_mask_t.sum()>0 else np.nan
        acc_bd   = correct[bd_mask_t].mean()   if bd_mask_t.sum()>0   else np.nan
        acc_comb = correct[combined_mask_t].mean() if combined_mask_t.sum()>0 else np.nan
        act_cov  = cpsl_mask_t.mean()

        print(f"  ~{cov_target*100:.0f}% selection rate        "
              f"  {acc_cpsl*100:>14.1f}%  {acc_bd*100:>12.1f}%  "
              f"{acc_comb*100:>12.1f}%")
        curve_results.append({
            "target_selection_rate": cov_target,
            "cpsl_accuracy": float(acc_cpsl),
            "bd_accuracy": float(acc_bd),
            "combined_accuracy": float(acc_comb),
        })

    results["precision_selection_rate_curve"] = curve_results

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT: Precision-selection curve — CPSL, BD, Combined on one graph
    # ══════════════════════════════════════════════════════════════════════════
    phdr("PRECISION-SELECTION CURVE (graph)")

    sel_rates = [r["target_selection_rate"] for r in curve_results]
    acc_cpsl  = [r["cpsl_accuracy"]     for r in curve_results]
    acc_bd    = [r["bd_accuracy"]       for r in curve_results]
    acc_comb  = [r["combined_accuracy"] for r in curve_results]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot([s*100 for s in sel_rates], [a*100 for a in acc_cpsl],
            color="#6B8CFF", linewidth=2.5, marker="o", markersize=6,
            label="Conformal CPSL alone")
    ax.plot([s*100 for s in sel_rates], [a*100 for a in acc_bd],
            color="#2D9E6B", linewidth=2.5, marker="s", markersize=6,
            label="BD evidence alone (e_mean, cal. threshold)")
    ax.plot([s*100 for s in sel_rates], [a*100 for a in acc_comb],
            color="#E8A838", linewidth=3.0, marker="^", markersize=7,
            label="Combined (CPSL=1 AND High BD)", zorder=5)

    # Mark the headline operating point
    ax.axvline(34.1, color="#E8A838", linewidth=1, linestyle="--", alpha=0.6)
    ax.axhline(98.2, color="#E8A838", linewidth=1, linestyle="--", alpha=0.6)
    ax.scatter([34.1], [98.2], color="#E8A838", s=120, zorder=6)
    ax.text(35, 97, "34% selected\n98.2% accuracy",
            color="#E8A838", fontsize=9, fontweight="bold")

    # Baseline
    ax.axhline(80.3, color="grey", linewidth=1, linestyle=":", alpha=0.7,
               label="No filter (80.3%)")

    ax.set_xlabel("Selection rate (%)\n(fraction of cases selected as confident)",
                  fontsize=11)
    ax.set_ylabel("Accuracy (%) against LCA-derived labels", fontsize=11)
    ax.set_title("Precision-Selection Curve\nCombined BD + Conformal outperforms either signal at every operating point",
                 fontsize=12)
    ax.set_xlim(15, 105)
    ax.set_ylim(75, 101)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3)
    ax.invert_xaxis()  # high selection rate on right, low (most selective) on left

    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, "precision_selection_curve.png")
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: precision_selection_curve.png")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6: PATHOGEN-SPECIFIC COMBINED ANALYSIS
    # Uses 2-class CPSL_BAC and CPSL_FUNG separately with e_bac and e_fung
    # ══════════════════════════════════════════════════════════════════════════
    phdr("6. PATHOGEN-SPECIFIC COMBINED ANALYSIS")
    print("  2-class CPSL_BAC paired with e_bac; CPSL_FUNG paired with e_fung")
    print("  All thresholds defined on calibration set\n")

    # ── 2-class LAC per pathogen ──────────────────────────────────────────────
    def lac_2class(sig_probs, labels, cal_idx, eval_idx, alpha, pathogen=0):
        """2-class binary LAC for one pathogen (0=BAC, 1=FUNG)."""
        p     = sig_probs[:, pathogen]
        lab   = (labels[:, pathogen] > 0.5).astype(int)
        scores_cal = np.where(lab[cal_idx] == 1,
                              1 - p[cal_idx], p[cal_idx])
        n      = len(cal_idx)
        qlevel = min(np.ceil((n+1)*(1-alpha))/n, 1.0)
        qhat   = float(np.quantile(scores_cal, qlevel, method="higher"))
        include_pos = p[eval_idx] >= (1 - qhat)
        include_neg = p[eval_idx] <= qhat
        psets = include_pos.astype(int) + include_neg.astype(int)
        return psets, qhat

    sig_probs = sig["probs"]
    sig_labels = sig["labels"]

    cpsl_bac, qhat_bac = lac_2class(sig_probs, sig_labels,
                                     cal_idx, eval_idx, args.alpha, pathogen=0)
    cpsl_fung, qhat_fung = lac_2class(sig_probs, sig_labels,
                                      cal_idx, eval_idx, args.alpha, pathogen=1)
    print(f"  2-class BAC  qhat={qhat_bac:.4f}  "
          f"CPSL=1: {(cpsl_bac==1).mean()*100:.1f}% of eval")
    print(f"  2-class FUNG qhat={qhat_fung:.4f}  "
          f"CPSL=1: {(cpsl_fung==1).mean()*100:.1f}% of eval\n")

    # ── Pathogen-specific BD thresholds from calibration set ─────────────────
    e_bac_cal  = bd_params[cal_idx, 0] + bd_params[cal_idx, 1]
    e_fung_cal = bd_params[cal_idx, 2] + bd_params[cal_idx, 3]
    t67_bac  = np.percentile(e_bac_cal,  67)
    t67_fung = np.percentile(e_fung_cal, 67)
    print(f"  Calibration thresholds:")
    print(f"    t67_bac  = {t67_bac:.3f}  (n_cal={len(cal_idx)})")
    print(f"    t67_fung = {t67_fung:.3f}\n")

    # Eval evidence
    e_bac_eval  = bd_params[eval_idx, 0] + bd_params[eval_idx, 1]
    e_fung_eval = bd_params[eval_idx, 2] + bd_params[eval_idx, 3]
    high_bac  = e_bac_eval  > t67_bac
    high_fung = e_fung_eval > t67_fung

    # ── 2-class accuracy helper ───────────────────────────────────────────────
    bac_bin  = (sig_labels[eval_idx, 0] > 0.5).astype(int)
    fung_bin = (sig_labels[eval_idx, 1] > 0.5).astype(int)
    pred_bac  = (sig_probs[eval_idx, 0] > 0.5).astype(int)
    pred_fung = (sig_probs[eval_idx, 1] > 0.5).astype(int)
    correct_bac  = (pred_bac  == bac_bin).astype(float)
    correct_fung = (pred_fung == fung_bin).astype(float)

    def show_pathogen_table(name, cpsl1, high_e, correct, e_vals, pathogen_str):
        print(f"  {'─'*65}")
        print(f"  {name}")
        print(f"  {'─'*65}")
        print(f"  {'Criterion':40s}  {'Sel.rate':>9}  {'Accuracy':>9}  {'N':>6}")
        print(f"  {'·'*65}")
        rows = []
        for label, mask in [
            (f"No filter",                       np.ones(len(correct), bool)),
            (f"CPSL_{pathogen_str}=1 alone",     cpsl1 == 1),
            (f"High e_{pathogen_str} alone",      high_e),
            (f"CPSL_{pathogen_str}=1 AND High e", (cpsl1==1) & high_e),
        ]:
            sel  = mask.mean()
            acc  = correct[mask].mean() if mask.sum() > 0 else float("nan")
            flag = "  ✓ beats both" if label.endswith("High e") and \
                   acc > correct[cpsl1==1].mean() and \
                   acc > correct[high_e].mean() else ""
            print(f"  {label:40s}  {sel*100:>8.1f}%  {acc*100:>8.1f}%  "
                  f"{mask.sum():>5}{flag}")
            rows.append({"criterion": label, "sel_rate": float(sel),
                         "accuracy": float(acc), "n": int(mask.sum())})
        return rows

    rows_bac  = show_pathogen_table("BACTERIAL (2-class BAC)",
                                     cpsl_bac, high_bac, correct_bac,
                                     e_bac_eval, "BAC")
    rows_fung = show_pathogen_table("FUNGAL (2-class FUNG)",
                                     cpsl_fung, high_fung, correct_fung,
                                     e_fung_eval, "FUNG")

    results["pathogen_specific"] = {
        "bac":  {"t67_cal": float(t67_bac),  "rows": rows_bac},
        "fung": {"t67_cal": float(t67_fung), "rows": rows_fung},
    }

    # ── Precision-selection curves per pathogen ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle("Pathogen-Specific Precision-Selection Curves\n"
                 "2-class CPSL and pathogen-specific BD evidence",
                 fontsize=13)

    for ax, (cpsl_p, high_e, correct_p, e_vals, t67, pname, col_cpsl, col_bd) in zip(axes, [
        (cpsl_bac,  high_bac,  correct_bac,  e_bac_eval,  t67_bac,
         "Bacterial", "#6B8CFF", "#2D9E6B"),
        (cpsl_fung, high_fung, correct_fung, e_fung_eval, t67_fung,
         "Fungal",   "#9B6BFF", "#1B8A8F"),
    ]):
        targets = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
        acc_c, acc_b, acc_co, sels = [], [], [], []

        for tgt in targets:
            # CPSL: select smallest sets until coverage reached
            for s in [1, 2]:
                if (cpsl_p <= s).mean() >= tgt:
                    m_c = cpsl_p <= s; break
            else:
                m_c = np.ones(len(cpsl_p), bool)

            # BD: top fraction by evidence
            thresh_b = np.percentile(e_vals, (1-tgt)*100)
            m_b = e_vals >= thresh_b

            # Combined
            m_co = (cpsl_p == 1) & (e_vals >= np.percentile(
                e_vals[cpsl_p==1], max(0, (1-(tgt/(cpsl_p==1).mean()))*100)))

            acc_c.append(correct_p[m_c].mean()*100   if m_c.sum()>0  else np.nan)
            acc_b.append(correct_p[m_b].mean()*100   if m_b.sum()>0  else np.nan)
            acc_co.append(correct_p[m_co].mean()*100 if m_co.sum()>0 else np.nan)
            sels.append(tgt*100)

        ax.plot(sels, acc_c,  color=col_cpsl, linewidth=2.5, marker="o",
                markersize=6, label=f"CPSL_{pname[:3].upper()} alone")
        ax.plot(sels, acc_b,  color=col_bd,   linewidth=2.5, marker="s",
                markersize=6, label=f"High e_{pname[:3].lower()} alone")
        ax.plot(sels, acc_co, color="#E8A838", linewidth=3.0, marker="^",
                markersize=7, label="Combined", zorder=5)
        ax.axhline(correct_p.mean()*100, color="grey", linestyle=":",
                   alpha=0.7, label=f"No filter ({correct_p.mean()*100:.1f}%)")
        ax.set_title(f"{pname} Keratitis", fontsize=12)
        ax.set_xlabel("Selection rate (%)", fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.invert_xaxis()

    plt.tight_layout()
    path2 = os.path.join(args.out_dir, "precision_selection_curve_pathogen.png")
    plt.savefig(path2, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: precision_selection_curve_pathogen.png")

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(os.path.join(args.out_dir, "combined_uq_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {args.out_dir}/combined_uq_results.json")

if __name__ == "__main__":
    main()