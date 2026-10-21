"""
UQ vs Accuracy Scatter Plots
=============================
Four panels, each showing one uncertainty signal vs correctness,
with a smoothed accuracy curve overlaid.

  Panel 1: Sigmoid entropy vs accuracy
  Panel 2: CPSL vs accuracy
  Panel 3: BD evidence (mean alpha+beta) vs accuracy
  Panel 4: Combined signal (BD evidence + CPSL) vs accuracy

Each panel:
  - x axis: uncertainty signal value
  - background scatter: individual eval samples (correct=green, wrong=red),
    jittered vertically so dots don't all sit on 0/1
  - overlaid curve: smoothed accuracy (rolling window or isotonic regression)
  - annotation: Spearman rho

Run:
    python uq_accuracy_scatter.py
        --cache_sig  runs/bd_soft_.../uncertainty_analysis/cache_sig.npz
        --cache_bd   runs/bd_soft_.../uncertainty_analysis/cache_bd.npz
        --csv        /path/to/data.csv
        --out_dir    runs/bd_soft_.../uncertainty_analysis/
"""

import os, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY    = "#FFFFFF"
TEAL    = "#1B8A8F"
GREEN   = "#2D9E6B"
RED     = "#E05252"
GOLD    = "#E8A838"
OFFWHITE= "#1A1A2E"
SLATE   = "#4A5568"

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_sig", type=str, required=True)
    p.add_argument("--cache_bd",  type=str, required=True)
    p.add_argument("--csv", type=str,
        default="/persist/QTIM/Active/26-0096/Corneal_Ulcer_Image_Sets/Data/"
                "corneal_ulcer_study_data_7_24_2025_lambda.csv")
    p.add_argument("--alpha",   type=float, default=0.1)
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--out_dir", type=str,   default=None)
    p.add_argument("--dpi",     type=int,   default=180)
    return p.parse_args()


def load_eval(csv_path, seed=42):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["file_path","subject"])
    df = df[(df["fung_LCA_bin"]!=-1)|(df["bac_LCA_bin"]!=-1)].reset_index(drop=True)
    subj = df["subject"].values
    gss = GroupShuffleSplit(1, test_size=0.30, random_state=seed)
    _, ti = next(gss.split(df, groups=subj))
    df_t = df.iloc[ti].reset_index(drop=True)
    gss2 = GroupShuffleSplit(1, test_size=0.50, random_state=seed)
    _, ti2 = next(gss2.split(df_t, groups=df_t["subject"].values))
    df_test = df_t.iloc[ti2].reset_index(drop=True)
    gss3 = GroupShuffleSplit(1, test_size=0.80, random_state=seed)
    cal_idx, eval_idx = next(gss3.split(df_test, groups=df_test["subject"].values))
    return df_test, cal_idx, eval_idx


def factored_joint(probs):
    b, f = probs[:,0], probs[:,1]
    return np.stack([(1-b)*(1-f), b*(1-f), (1-b)*f, b*f], axis=1)


def compute_signals(sig_probs, sig_labels, bd_params,
                    cal_idx, eval_idx, alpha=0.1):
    """Compute all four UQ signals on the eval subset."""
    p4   = factored_joint(sig_probs)
    eps  = 1e-12
    entropy = -(p4 * np.log(p4 + eps)).sum(axis=1)   # shape (N,)

    # 4-class correctness
    bac_b = (sig_labels[:,0] > 0.5).astype(int)
    fng_b = (sig_labels[:,1] > 0.5).astype(int)
    lab4  = bac_b + fng_b * 2
    pred4 = p4.argmax(axis=1)
    correct = (pred4 == lab4).astype(float)           # shape (N,)

    # CPSL (vanilla 4-class LAC)
    n      = len(cal_idx)
    scores = 1 - p4[cal_idx][np.arange(n), lab4[cal_idx]]
    qlevel = min(np.ceil((n+1)*(1-alpha))/n, 1.0)
    qhat   = float(np.quantile(scores, qlevel, method="higher"))
    cpsl   = (p4[eval_idx] >= (1-qhat)).sum(axis=1).astype(float)

    # BD evidence
    e_bac  = bd_params[eval_idx,0] + bd_params[eval_idx,1]
    e_fung = bd_params[eval_idx,2] + bd_params[eval_idx,3]
    e_mean = (e_bac + e_fung) / 2

    return {
        "entropy":  entropy[eval_idx],
        "cpsl":     cpsl,
        "e_mean":   e_mean,
        "correct":  correct[eval_idx],
        "qhat":     qhat,
    }


def smoothed_accuracy(x, correct, n_bins=25, sigma=1.5):
    """
    Bin x into n_bins, compute accuracy per bin,
    apply Gaussian smoothing for the curve.
    Returns (bin_centers, smoothed_acc, bin_ns).
    """
    order      = np.argsort(x)
    x_s        = x[order]
    c_s        = correct[order]
    bins       = np.array_split(np.arange(len(x_s)), n_bins)
    centers    = np.array([x_s[b].mean() for b in bins if len(b)>0])
    accs       = np.array([c_s[b].mean() for b in bins if len(b)>0])
    ns         = np.array([len(b)         for b in bins if len(b)>0])
    smoothed   = gaussian_filter1d(accs, sigma=sigma)
    return centers, smoothed, accs, ns


def scatter_panel(ax, x, correct, xlabel, title, color_curve,
                  invert_x=False, n_bins=25):
    """
    Draw one scatter+curve panel.
    correct: array of 0.0 / 1.0
    """
    rng = np.random.default_rng(42)

    # Jitter y so dots don't stack on 0/1
    jitter = rng.uniform(-0.08, 0.08, size=len(x))
    y_jit  = correct + jitter

    # Color by correctness
    colors = np.where(correct == 1, GREEN, RED)

    ax.scatter(x, y_jit, c=colors, alpha=0.25, s=12,
               linewidths=0, zorder=2)

    # Smoothed accuracy curve
    centers, smoothed, raw_accs, ns = smoothed_accuracy(x, correct, n_bins=n_bins)
    ax.plot(centers, smoothed, color=color_curve, linewidth=2.5,
            zorder=4, label="Smoothed accuracy")

    # Horizontal reference lines
    ax.axhline(correct.mean(), color=SLATE, linewidth=1,
               linestyle=":", alpha=0.7, label=f"Overall {correct.mean()*100:.1f}%")
    ax.axhline(1.0, color=OFFWHITE, linewidth=0.5, alpha=0.3)
    ax.axhline(0.0, color=OFFWHITE, linewidth=0.5, alpha=0.3)

    # Spearman rho
    # For CPSL and entropy: expect negative rho (higher = worse)
    # For evidence: expect positive rho (higher = better)
    rho, pval = spearmanr(x, correct)
    p_str = "p<0.0001" if pval < 0.0001 else f"p={pval:.4f}"
    ax.text(0.97, 0.06, f"Spearman ρ = {rho:+.3f}\n{p_str}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color=OFFWHITE,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_curve,
                      alpha=0.95))

    ax.set_xlabel(xlabel, fontsize=11, color="#1A1A2E", labelpad=6)
    ax.set_ylabel("Correct (1) / Wrong (0)", fontsize=10, color="#1A1A2E")
    ax.set_title(title, fontsize=13, color=color_curve, fontweight="bold", pad=10)
    ax.set_ylim(-0.25, 1.25)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["Wrong", "50%", "Correct"], fontsize=9, color="#1A1A2E")
    ax.tick_params(colors="#1A1A2E", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#CCCCCC")
    ax.set_facecolor("#FFFFFF")
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.8, alpha=0.8)
    ax.legend(fontsize=8, loc="upper left" if not invert_x else "upper right",
              framealpha=0.9, facecolor="white", edgecolor="#CCCCCC",
              labelcolor="#1A1A2E")

    if invert_x:
        ax.invert_xaxis()

    return rho


def combined_panel(ax, e_mean, cpsl, correct, e_mean_cal, n_bins=20):
    """
    Panel 4: combined signal.
    Thresholds defined on calibration data (e_mean_cal), applied to eval.
    """
    # Calibration-set thresholds — same approach as CPSL's qhat
    t67_cal = np.percentile(e_mean_cal, 67)

    # Percentile rank of eval evidence relative to calibration distribution
    # For the x-axis: use a combined score based on cal-defined thresholds
    e_pct   = np.array([np.mean(e_mean_cal <= v) for v in e_mean])  # cal-referenced rank
    c_pct   = pd.Series(cpsl).rank(pct=True).values
    combined = e_pct - c_pct

    rho = scatter_panel(ax, combined, correct,
                        xlabel="Combined score  (BD evidence cal-rank − CPSL rank)",
                        title="Combined BD + Conformal vs Accuracy\n(BD threshold from calibration set)",
                        color_curve=GOLD,
                        n_bins=n_bins)

    # Mark the threshold: CPSL=1 AND e_mean > t67_cal
    # On the combined axis, find where these cases sit
    high_bd_eval = e_mean > t67_cal
    cpsl1_eval   = cpsl == 1
    combined_mask = high_bd_eval & cpsl1_eval
    n_sel = combined_mask.sum()
    sel_rate = combined_mask.mean()
    acc = correct[combined_mask].mean() if n_sel > 0 else float("nan")

    # Mark the minimum combined score for selected cases
    if n_sel > 0:
        t_combined = combined[combined_mask].min()
        ax.axvline(t_combined, color=GOLD, linewidth=1.5,
                   linestyle="--", alpha=0.8)
        ax.text(t_combined + 0.02, 1.15,
                f"CPSL=1 AND High BD\n{sel_rate*100:.0f}% selected: {acc*100:.1f}% acc",
                fontsize=9, color=GOLD, fontweight="bold")

    ax.text(0.03, 0.06,
            f"Cal threshold t₆₇ = {t67_cal:.2f}",
            transform=ax.transAxes, fontsize=8, color=SLATE,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=SLATE, alpha=0.8))

    ax.legend(fontsize=8, loc="upper left", framealpha=0.9,
              facecolor="white", edgecolor="#CCCCCC", labelcolor="#1A1A2E")
    return rho, combined


def main():
    args = get_args()
    if args.out_dir is None:
        args.out_dir = os.path.dirname(args.cache_bd)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    sig_c  = np.load(args.cache_sig)
    bd_c   = np.load(args.cache_bd)
    df_test, cal_idx, eval_idx = load_eval(args.csv, args.seed)

    sig  = compute_signals(sig_c["probs"], sig_c["labels"],
                           bd_c["params"], cal_idx, eval_idx, args.alpha)

    # BD evidence on calibration set — for threshold definition
    bd_params  = bd_c["params"]
    e_bac_cal  = bd_params[cal_idx,0] + bd_params[cal_idx,1]
    e_fung_cal = bd_params[cal_idx,2] + bd_params[cal_idx,3]
    e_mean_cal = (e_bac_cal + e_fung_cal) / 2
    print(f"BD cal threshold t67 = {np.percentile(e_mean_cal,67):.3f}  (n_cal={len(cal_idx)})")

    print(f"Eval n={len(sig['correct'])}  "
          f"overall accuracy={sig['correct'].mean()*100:.1f}%")
    print(f"Conformal qhat={sig['qhat']:.4f}  mean CPSL={sig['cpsl'].mean():.3f}")

    for name, vals in [("entropy",sig["entropy"]),
                       ("cpsl",sig["cpsl"]),
                       ("e_mean",sig["e_mean"])]:
        rho,_ = spearmanr(vals, sig["correct"])
        print(f"  {name:12s}: rho={rho:+.3f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14), facecolor="white")
    fig.suptitle(
        "Uncertainty Signals vs Classification Accuracy\n"
        "Soft label models — Cell A (sigmoid) + Cell B (BD)",
        fontsize=15, color="#0D1B2A", y=0.98, fontweight="bold"
    )

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.38, wspace=0.32,
                           left=0.07, right=0.97,
                           top=0.93, bottom=0.07)

    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,0])
    ax4 = fig.add_subplot(gs[1,1])

    scatter_panel(ax1, sig["entropy"], sig["correct"],
                  xlabel="Sigmoid Entropy  H = -ΣP(k)logP(k)",
                  title="Entropy vs Accuracy",
                  color_curve=TEAL,
                  invert_x=False)

    scatter_panel(ax2, sig["cpsl"], sig["correct"],
                  xlabel="Conformal Set Size (CPSL)",
                  title="CPSL vs Accuracy",
                  color_curve="#6B8CFF",
                  invert_x=False,
                  n_bins=4)   # CPSL is discrete 1-4

    # For CPSL: set x ticks to 1,2,3,4
    ax2.set_xticks([1,2,3,4])
    ax2.set_xticklabels(["1\n(confident)","2","3","4\n(uncertain)"],
                         fontsize=9, color=OFFWHITE)

    scatter_panel(ax3, sig["e_mean"], sig["correct"],
                  xlabel="BD Evidence  (α+β mean across pathogens)",
                  title="BD Evidence vs Accuracy",
                  color_curve=GREEN)

    _, combined = combined_panel(ax4, sig["e_mean"], sig["cpsl"], sig["correct"], e_mean_cal)

    # Save
    out_path = os.path.join(args.out_dir, "uq_vs_accuracy_scatter_white.png")
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"\nSaved: {out_path}")

    def make_fig():
        fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
        ax.set_facecolor("white")
        return fig, ax

    # Plot 5: Entropy vs CPSL
    fig5, ax5 = make_fig()
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.15, 0.15, size=len(sig["cpsl"]))
    ax5.scatter(sig["entropy"], sig["cpsl"] + jitter,
                alpha=0.25, s=12, color=TEAL, linewidths=0)
    # Smoothed mean CPSL per entropy bin
    ent_sorted = np.argsort(sig["entropy"])
    bins = np.array_split(ent_sorted, 20)
    centers = [sig["entropy"][b].mean() for b in bins if len(b)>0]
    means   = [sig["cpsl"][b].mean()    for b in bins if len(b)>0]
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(means, sigma=1.5)
    ax5.plot(centers, smoothed, color="#E8A838", linewidth=2.5, zorder=4)
    rho5, p5 = spearmanr(sig["entropy"], sig["cpsl"])
    p5_str = "p<0.0001" if p5 < 0.0001 else f"p={p5:.4f}"
    ax5.text(0.97, 0.06, f"Spearman ρ = {rho5:+.3f}\n{p5_str}",
             transform=ax5.transAxes, ha="right", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=TEAL, alpha=0.95))
    ax5.set_xlabel("Sigmoid Entropy", fontsize=11, color="#1A1A2E", labelpad=6)
    ax5.set_ylabel("CPSL (jittered)", fontsize=10, color="#1A1A2E")
    ax5.set_title("Entropy vs CPSL",
                  fontsize=13, color="#1A1A2E", fontweight="bold")
    ax5.set_yticks([1,2,3,4])
    ax5.set_yticklabels(["1 (confident)","2","3","4 (uncertain)"],
                         fontsize=9, color="#1A1A2E")
    ax5.tick_params(colors="#1A1A2E", labelsize=9)
    for sp in ax5.spines.values(): sp.set_color("#CCCCCC")
    ax5.set_facecolor("white"); ax5.grid(alpha=0.3, color="#E2E8F0")
    fig5.tight_layout()
    p5_path = os.path.join(args.out_dir, "uq_scatter_5_entropy_vs_cpsl.png")
    fig5.savefig(p5_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig5)
    print(f"Saved: {p5_path}")

    # Plot 6: CPSL vs BD evidence
    fig6, ax6 = make_fig()
    jitter2 = rng.uniform(-0.15, 0.15, size=len(sig["cpsl"]))
    sc = ax6.scatter(sig["cpsl"] + jitter2, sig["e_mean"],
                     alpha=0.25, s=12, color=GREEN, linewidths=0)
    # Smoothed mean evidence per CPSL bin
    for cpsl_val in [1,2,3,4]:
        mask = sig["cpsl"] == cpsl_val
        if mask.sum() > 0:
            ax6.scatter([cpsl_val], [sig["e_mean"][mask].mean()],
                        color="#E8A838", s=120, zorder=5,
                        marker="D", label=f"Mean e at CPSL={cpsl_val}" if cpsl_val==1 else "")
    rho6, p6 = spearmanr(sig["cpsl"], sig["e_mean"])
    p6_str = "p<0.0001" if p6 < 0.0001 else f"p={p6:.4f}"
    ax6.text(0.97, 0.94, f"Spearman ρ = {rho6:+.3f}\n{p6_str}",
             transform=ax6.transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREEN, alpha=0.95))
    ax6.set_xlabel("CPSL (jittered)", fontsize=11, color="#1A1A2E", labelpad=6)
    ax6.set_ylabel("BD Evidence (α+β mean)", fontsize=10, color="#1A1A2E")
    ax6.set_title("CPSL vs BD Evidence",
                  fontsize=13, color="#1A1A2E", fontweight="bold")
    ax6.set_xticks([1,2,3,4])
    ax6.set_xticklabels(["1 (confident)","2","3","4 (uncertain)"],
                         fontsize=9, color="#1A1A2E")
    ax6.tick_params(colors="#1A1A2E", labelsize=9)
    for sp in ax6.spines.values(): sp.set_color("#CCCCCC")
    ax6.set_facecolor("white"); ax6.grid(alpha=0.3, color="#E2E8F0")
    fig6.tight_layout()
    p6_path = os.path.join(args.out_dir, "uq_scatter_6_cpsl_vs_evidence.png")
    fig6.savefig(p6_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig6)
    print(f"Saved: {p6_path}")

    # Also save the combined signal data for reference
    pd.DataFrame({
        "entropy":  sig["entropy"],
        "cpsl":     sig["cpsl"].astype(int),
        "e_mean":   sig["e_mean"],
        "combined": combined,
        "correct":  sig["correct"].astype(int),
    }).to_csv(os.path.join(args.out_dir, "uq_signals_eval.csv"), index=False)
    print(f"Saved: uq_signals_eval.csv")


if __name__ == "__main__":
    main()