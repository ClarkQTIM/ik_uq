"""
Conformal Regression for Infectious Keratitis
==============================================
Three cleanly separated experiments:

EXPERIMENT 1 — Sigmoid only (label type comparison)
    Score: Joint Chebyshev  s_i = max(|y_bac - p_bac|, |y_fung - p_fung|)
    Models: Cell A (sigmoid + soft) vs Cell C (sigmoid + hard)
    Question: Does soft label training produce narrower intervals?
    No BD involved.

EXPERIMENT 2 — Sigmoid + BD combined (cross-framework)
    Score: BD-Scaled Chebyshev
           s_i = max(|y_bac - p_bac| / (sigma_bac + eps),
                     |y_fung - p_fung| / (sigma_fung + eps))
           where sigma = 1 / (alpha + beta) from BD model
    Models: Cell A (sigmoid) + Cell B (BD), both soft labels
    Question: Does BD evidence improve sigmoid conformal intervals?
    Patient-specific widths — low BD evidence -> wider intervals.

EXPERIMENT 3 — BD standalone (novel score)
    Score: Beta Density  s_i = max(-log f_Beta(y_bac; a_bac, b_bac),
                                   -log f_Beta(y_fung; a_fung, b_fung))
    Model: Cell B (BD + soft labels) only
    Question: Can the BD model's distributional output directly
              generate valid conformal intervals?
    Highest-density region (HDR) intervals per patient.
    Novel: likelihood-based conformal on probabilistic soft labels.

Run:
    python conformal_regression.py
        --weights      runs/sigmoid_soft_.../best_model.pt    # Cell A (required)
        --weights_hard runs/sigmoid_hard_.../best_model.pt   # Cell C (optional)
        --bd_weights   runs/beta_dirichlet_soft_.../best_model.pt  # Cell B (optional)
        --alpha 0.1
"""

import os, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import beta as beta_dist, spearmanr
import timm
from tqdm import tqdm
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
    p.add_argument("--img_root", type=str,
        default="/pacs/OSHU/Corneal_Ulcer_Image_Sets")
    p.add_argument("--weights",      type=str, required=True,
        help="Cell A: sigmoid + soft labels")
    p.add_argument("--weights_hard", type=str, default=None,
        help="Cell C: sigmoid + hard labels")
    p.add_argument("--bd_weights",   type=str, default=None,
        help="Cell B: beta-dirichlet + soft labels")
    p.add_argument("--alpha",  type=float, default=0.1)
    p.add_argument("--batch",  type=int,   default=8)
    p.add_argument("--seed",   type=int,   default=42)
    p.add_argument("--out_dir",type=str,   default=None)
    return p.parse_args()


# =============================================================================
# DATA
# =============================================================================

PACS_PREFIX = "/pacs/OSHU/Corneal_Ulcer_Image_Sets"
STUDY_SHORT = {
    "MADURAI photos":"MADURAI",   "MUTT2-Pictures":"MUTT2",
    "MUTT1-Pictures":"MUTT1",     "SCUT photos":"SCUT",
    "MUTT Pilot Photos":"MUTT Pilot",
    "CLAIR II photos":"CLAIR II", "CLAIR I photos":"CLAIR I",
}

def load_test_set(csv_path, img_root, seed=42):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["file_path","subject"])
    df = df[(df["fung_LCA_bin"]!=-1)|(df["bac_LCA_bin"]!=-1)].reset_index(drop=True)
    df["local_path"] = df["file_path"].apply(
        lambda x: x.replace(PACS_PREFIX, img_root))

    # ── REGRESSION FILTER ────────────────────────────────────────────────────
    # Conformal regression requires BOTH soft posteriors to be valid (in [0,1]).
    # Single-etiology studies have bac_post_prob=-1 (MUTT) or fung_post_prob=-1
    # (SCUT) for the non-relevant pathogen — the LCA never estimated those.
    # These rows cannot serve as regression targets and must be excluded.
    #
    # Impact: 5,385 -> 2,534 rows. Remaining breakdown:
    #   MADURAI:    2,001 (79% of valid set — only truly mixed-etiology study)
    #   CLAIR I/II:   126
    #   MUTT*:        341 (partial — only rows where both posteriors exist)
    #   SCUT:          66
    #
    # IMPORTANT: This means conformal regression is predominantly a
    # MADURAI-specific analysis. Results do not generalise to the full dataset.
    # Classification conformal (conformal.py) uses bac/fung_LCA_bin labels
    # which are always valid — use that for full-dataset UQ.
    # ─────────────────────────────────────────────────────────────────────────
    n_before = len(df)
    df = df[(df["bac_post_prob"] >= 0) & (df["fung_post_prob"] >= 0)].reset_index(drop=True)
    n_after = len(df)
    print(f"Regression filter: {n_before:,} -> {n_after:,} rows "
          f"({n_before-n_after:,} dropped — single-etiology studies with "
          f"missing posteriors)")
    print(f"Valid rows by study:")
    for study, n in df.groupby("study").size().items():
        short = STUDY_SHORT.get(study, study)
        print(f"  {short:12s}: {n:,}")
    print()

    subj = df["subject"].values
    gss = GroupShuffleSplit(1, test_size=0.30, random_state=seed)
    _, ti = next(gss.split(df, groups=subj))
    df_t = df.iloc[ti].reset_index(drop=True)
    gss2 = GroupShuffleSplit(1, test_size=0.50, random_state=seed)
    _, ti2 = next(gss2.split(df_t, groups=df_t["subject"].values))
    df_test = df_t.iloc[ti2].reset_index(drop=True)
    gss3 = GroupShuffleSplit(1, test_size=0.80, random_state=seed)
    cal_idx, eval_idx = next(gss3.split(df_test,
                                         groups=df_test["subject"].values))
    print(f"Test: {len(df_test):,}  Cal: {len(cal_idx):,}  "
          f"Eval: {len(eval_idx):,}\n")
    return df_test, cal_idx, eval_idx

class KeratitisDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df; self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:    img = Image.open(row["local_path"]).convert("RGB")
        except: img = Image.new("RGB",(600,600),0)
        if self.transform: img = self.transform(img)
        # Always soft posteriors as regression target
        label = torch.tensor(
            [row["bac_post_prob"], row["fung_post_prob"]],
            dtype=torch.float32)
        return img, label

val_tf = transforms.Compose([
    transforms.Resize((600,600)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


# =============================================================================
# MODELS
# =============================================================================

class SigmoidModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnet_b7.ra_in1k",
            pretrained=False, num_classes=0, global_pool="avg")
        f = self.backbone.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(f), nn.Linear(f,512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512,32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32,2), nn.Sigmoid())
    def forward(self, x): return self.head(self.backbone(x))

class BDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnet_b7.ra_in1k",
            pretrained=False, num_classes=0, global_pool="avg")
        f = self.backbone.num_features
        self.trunk = nn.Sequential(
            nn.BatchNorm1d(f), nn.Linear(f,512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512,32), nn.ReLU(), nn.Dropout(0.3))
        self.bac_head  = nn.Sequential(nn.Linear(32,2), nn.Softplus())
        self.fung_head = nn.Sequential(nn.Linear(32,2), nn.Softplus())
    def forward(self, x):
        f = self.trunk(self.backbone(x))
        return torch.cat([self.bac_head(f), self.fung_head(f)], dim=-1)

def load_sig(path, device):
    m = SigmoidModel()
    raw = torch.load(path, map_location=device)
    fixed = {}
    for k, v in raw.items():
        if k.startswith("head.net."): fixed["head."+k[len("head.net."):]] = v
        elif k.startswith("backbone."): fixed[k] = v
    m.load_state_dict(fixed); return m.to(device).eval()

def load_bd_model(path, device):
    m = BDModel()
    raw = torch.load(path, map_location=device)
    fixed = {(k[len("head."):] if k.startswith("head.") else k): v
             for k, v in raw.items()}
    m.load_state_dict(fixed); return m.to(device).eval()

@torch.no_grad()
def infer(model, loader, device, cache_path, out_key):
    if os.path.exists(cache_path):
        c = np.load(cache_path)
        print(f"  Cache loaded: {cache_path}")
        return c[out_key], c["labels"]
    all_out, all_lbl = [], []
    for imgs, lbls in tqdm(loader, desc="  Inference", leave=False):
        all_out.append(model(imgs.to(device)).cpu())
        all_lbl.append(lbls)
    out    = torch.cat(all_out).numpy()
    labels = torch.cat(all_lbl).numpy()
    np.savez(cache_path, **{out_key: out}, labels=labels)
    print(f"  Cache saved:  {cache_path}")
    return out, labels


# =============================================================================
# CONFORMAL QUANTILE
# =============================================================================

def conformal_quantile(scores, alpha):
    n = len(scores)
    q_level = min(np.ceil((n+1)*(1-alpha))/n, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


# =============================================================================
# SCORE FUNCTIONS
# =============================================================================

def chebyshev_scores(preds, targets):
    """
    Joint Chebyshev score.
    s_i = max(|y_bac - p_bac|, |y_fung - p_fung|)
    Bounded in [0,1] since both preds and targets are in [0,1].
    """
    return np.maximum(
        np.abs(targets[:,0] - preds[:,0]),
        np.abs(targets[:,1] - preds[:,1])
    )

def bd_scaled_scores(preds, targets, bd_params, eps=0.01):
    """
    BD-scaled Chebyshev score.
    sigma = 1 / (alpha + beta)  — inverse evidence, proxy for uncertainty.
    s_i = max(|y_bac  - p_bac|  / (sigma_bac  + eps),
              |y_fung - p_fung| / (sigma_fung + eps))
    Unbounded — patient-specific scaling means scores can be large
    when BD is very uncertain.
    """
    e_bac  = bd_params[:,0] + bd_params[:,1]
    e_fung = bd_params[:,2] + bd_params[:,3]
    sigma_bac  = 1.0 / (e_bac  + eps)
    sigma_fung = 1.0 / (e_fung + eps)
    res_bac  = np.abs(targets[:,0] - preds[:,0]) / (sigma_bac  + eps)
    res_fung = np.abs(targets[:,1] - preds[:,1]) / (sigma_fung + eps)
    return np.maximum(res_bac, res_fung)

def beta_density_scores(targets, bd_params, eps=1e-6):
    """
    Beta density nonconformity score (novel).
    s_i = max(-log f_Beta(y_bac;  alpha_bac,  beta_bac),
              -log f_Beta(y_fung; alpha_fung, beta_fung))
    High score = at least one true posterior was surprising under
    the BD model's predicted Beta distribution.
    Unbounded above — log-likelihood space.
    """
    y_bac  = np.clip(targets[:,0], eps, 1-eps)
    y_fung = np.clip(targets[:,1], eps, 1-eps)
    a_bac  = np.maximum(bd_params[:,0], eps)
    b_bac  = np.maximum(bd_params[:,1], eps)
    a_fung = np.maximum(bd_params[:,2], eps)
    b_fung = np.maximum(bd_params[:,3], eps)
    nll_bac  = -beta_dist.logpdf(y_bac,  a_bac,  b_bac)
    nll_fung = -beta_dist.logpdf(y_fung, a_fung, b_fung)
    return np.maximum(nll_bac, nll_fung)


# =============================================================================
# INTERVAL CONSTRUCTION
# =============================================================================

def intervals_chebyshev(preds, q_hat):
    """
    Symmetric intervals, same width for all samples.
    [p_hat - q_hat, p_hat + q_hat] clipped to [0,1].
    Width = 2 * q_hat (constant).
    """
    lo = np.clip(preds - q_hat, 0, 1)
    hi = np.clip(preds + q_hat, 0, 1)
    widths = np.full((len(preds), 2), 2*q_hat)
    return lo, hi, widths

def intervals_bd_scaled(preds, q_hat, bd_params, eps=0.01):
    """
    Patient-specific intervals scaled by BD uncertainty.
    half_bac  = q_hat * (sigma_bac  + eps)
    half_fung = q_hat * (sigma_fung + eps)
    Where sigma = 1/(alpha+beta). Low evidence -> wider interval.
    """
    e_bac  = bd_params[:,0] + bd_params[:,1]
    e_fung = bd_params[:,2] + bd_params[:,3]
    sigma_bac  = 1.0 / (e_bac  + eps)
    sigma_fung = 1.0 / (e_fung + eps)
    half_bac   = q_hat * (sigma_bac  + eps)
    half_fung  = q_hat * (sigma_fung + eps)
    lo = np.stack([np.clip(preds[:,0]-half_bac,  0,1),
                   np.clip(preds[:,1]-half_fung, 0,1)], axis=1)
    hi = np.stack([np.clip(preds[:,0]+half_bac,  0,1),
                   np.clip(preds[:,1]+half_fung, 0,1)], axis=1)
    widths = np.stack([2*half_bac, 2*half_fung], axis=1)
    return lo, hi, widths

def intervals_beta_hdr(bd_params, q_hat, eps=1e-6):
    """
    Highest-density region (HDR) intervals for each Beta distribution.
    Finds the contiguous interval [lo, hi] such that
    -log f_Beta(y) <= q_hat for all y in [lo, hi].
    Numerically solved per sample using Brent's method.
    """
    from scipy.optimize import brentq
    N = len(bd_params)
    lo = np.zeros((N,2)); hi = np.ones((N,2))

    for i in tqdm(range(N), desc="  Beta HDR intervals", leave=False):
        for j, (a, b) in enumerate([
            (bd_params[i,0], bd_params[i,1]),
            (bd_params[i,2], bd_params[i,3])
        ]):
            a = max(a, eps); b = max(b, eps)

            def nll(y):
                return -beta_dist.logpdf(np.clip(y, eps, 1-eps), a, b)

            # Mode of Beta(a,b)
            if a > 1 and b > 1:
                mode = (a-1)/(a+b-2)
            elif a <= 1 and b > 1:
                mode = eps
            elif a > 1 and b <= 1:
                mode = 1-eps
            else:
                mode = 0.5

            if nll(mode) >= q_hat:
                # Even the mode exceeds threshold — use full interval
                lo[i,j] = 0.0; hi[i,j] = 1.0
                continue

            def f_lo(y): return nll(y) - q_hat
            def f_hi(y): return nll(y) - q_hat

            try:
                lo[i,j] = brentq(f_lo, eps, mode, xtol=1e-4)
            except ValueError:
                lo[i,j] = 0.0

            try:
                hi[i,j] = brentq(f_hi, mode, 1-eps, xtol=1e-4)
            except ValueError:
                hi[i,j] = 1.0

    widths = hi - lo
    return lo, hi, widths


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(lo, hi, targets, widths):
    """
    Joint coverage: BOTH dimensions must be covered simultaneously.
    Marginal coverage: each dimension independently.
    """
    cov_bac  = (targets[:,0] >= lo[:,0]) & (targets[:,0] <= hi[:,0])
    cov_fung = (targets[:,1] >= lo[:,1]) & (targets[:,1] <= hi[:,1])
    joint    = cov_bac & cov_fung
    return {
        "joint_coverage":  float(joint.mean()),
        "bac_coverage":    float(cov_bac.mean()),
        "fung_coverage":   float(cov_fung.mean()),
        "mean_width_bac":  float(widths[:,0].mean()),
        "mean_width_fung": float(widths[:,1].mean()),
        "std_width_bac":   float(widths[:,0].std()),
        "std_width_fung":  float(widths[:,1].std()),
        "widths_bac":      widths[:,0],
        "widths_fung":     widths[:,1],
        "covered_joint":   joint,
        "lo": lo, "hi": hi,
    }


# =============================================================================
# PRINT & SAVE
# =============================================================================

def phdr(t): print(f"\n{'='*70}\n  {t}\n{'='*70}")

def print_result(label, q_hat, m, alpha):
    target = (1-alpha)*100
    print(f"  {label}")
    print(f"    q_hat         = {q_hat:.4f}")
    print(f"    Joint coverage= {m['joint_coverage']*100:.1f}%  "
          f"(target {target:.0f}%)")
    print(f"    BAC  coverage = {m['bac_coverage']*100:.1f}%  "
          f"width = {m['mean_width_bac']:.4f} "
          f"(+/- {m['std_width_bac']:.4f})")
    print(f"    FUNG coverage = {m['fung_coverage']*100:.1f}%  "
          f"width = {m['mean_width_fung']:.4f} "
          f"(+/- {m['std_width_fung']:.4f})")


# =============================================================================
# ANALYSIS: INTERVAL WIDTH BY STUDY
# =============================================================================

def analysis_by_study(experiments, df_eval, alpha, out_dir):
    phdr("INTERVAL WIDTH BY STUDY")
    studies = df_eval["study"].values
    rows = []
    n_exp = len(experiments)
    fig, axes = plt.subplots(1, n_exp, figsize=(7*n_exp, 6),
                              sharey=False)
    if n_exp == 1: axes = [axes]
    fig.suptitle(f"Mean Interval Width by Study  (alpha={alpha})", fontsize=13)

    for ax, (exp_name, results) in zip(axes, experiments.items()):
        study_bac  = {}
        study_fung = {}
        for study in np.unique(studies):
            mask  = studies == study
            short = STUDY_SHORT.get(study, study)
            study_bac[short]  = results["widths_bac"][mask].mean()
            study_fung[short] = results["widths_fung"][mask].mean()
            rows.append({"Experiment": exp_name, "Study": short,
                         "Width_BAC":  study_bac[short],
                         "Width_FUNG": study_fung[short]})

        sorted_s = sorted(study_bac.keys(),
                          key=lambda s: (study_bac[s]+study_fung[s])/2,
                          reverse=True)
        x = np.arange(len(sorted_s))
        ax.bar(x-0.2, [study_bac[s]  for s in sorted_s], 0.38,
               label="BAC",  color="#2D9E6B", alpha=0.85)
        ax.bar(x+0.2, [study_fung[s] for s in sorted_s], 0.38,
               label="FUNG", color="#1B8A8F", alpha=0.85)
        ax.set_title(exp_name, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_s, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Mean Interval Width"); ax.legend(); ax.grid(alpha=0.3)
        print(f"\n  {exp_name}:")
        for s in sorted_s:
            print(f"    {s:12s}  BAC={study_bac[s]:.4f}  "
                  f"FUNG={study_fung[s]:.4f}")

    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "reg_width_by_study.csv"), index=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reg_width_by_study.png"), dpi=150)
    plt.close()
    print("\n  Saved: reg_width_by_study.png")


# =============================================================================
# ANALYSIS: WIDTH VS BD EVIDENCE
# =============================================================================

def analysis_width_vs_evidence(experiments, bd_params_eval, out_dir):
    phdr("INTERVAL WIDTH vs. BD EVIDENCE")
    e_bac  = bd_params_eval[:,0] + bd_params_eval[:,1]
    e_fung = bd_params_eval[:,2] + bd_params_eval[:,3]
    rows   = []

    fig, axes = plt.subplots(len(experiments), 2,
                              figsize=(12, 5*len(experiments)))
    if len(experiments) == 1: axes = [axes]
    fig.suptitle("Interval Width vs. BD Evidence (alpha+beta)\n"
                 "Negative correlation = framework captures same uncertainty",
                 fontsize=12)

    for row_axes, (exp_name, results) in zip(axes, experiments.items()):
        for ax, (pathogen, w, e) in zip(row_axes, [
            ("BAC",  results["widths_bac"],  e_bac),
            ("FUNG", results["widths_fung"], e_fung)
        ]):
            if np.std(w) < 1e-8:
                rho_str = "N/A (non-adaptive — constant width)"
                rho = p_val = float("nan")
                ax.text(0.5, 0.5, "Constant width\n(non-adaptive)",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=12, color="grey")
            else:
                rho, p_val = spearmanr(w, e)
                rho_str = f"rho={rho:.3f}, p={p_val:.4f}"
                # Bin by evidence
                pcts = [0,20,40,60,80,100]
                bins_e = [np.percentile(e, p) for p in pcts]
                blabels = [f"P{pcts[i]}-{pcts[i+1]}" for i in range(len(pcts)-1)]
                bidx = np.clip(np.digitize(e, bins_e[1:-1]), 0, len(blabels)-1)
                means = [w[bidx==b].mean() for b in range(len(blabels))]
                meds  = [np.median(e[bidx==b]) for b in range(len(blabels))]
                ax.plot(meds, means, marker="o", color="#1B8A8F", linewidth=2)
                ax.set_xlabel("Median BD Evidence (alpha+beta)")

            ax.set_title(f"{exp_name}\n{pathogen}  {rho_str}", fontsize=9)
            ax.set_ylabel("Mean Interval Width"); ax.grid(alpha=0.3)
            print(f"  {exp_name:45s}  {pathogen}: {rho_str}")
            rows.append({"Experiment": exp_name, "Pathogen": pathogen,
                         "rho": rho, "p": p_val})

    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "reg_evidence_correlation.csv"), index=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reg_width_vs_evidence.png"), dpi=150)
    plt.close()
    print("  Saved: reg_width_vs_evidence.png")


# =============================================================================
# ANALYSIS: WIDTH VS POSTERIOR VALUE
# =============================================================================

def analysis_width_vs_posterior(experiments, df_eval, out_dir):
    phdr("INTERVAL WIDTH vs. LCA POSTERIOR VALUE")
    bins    = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    blabels = ["0-0.1","0.1-0.3","0.3-0.5","0.5-0.7","0.7-0.9","0.9-1.0"]
    n_exp   = len(experiments)

    fig, axes = plt.subplots(n_exp, 2, figsize=(12, 4*n_exp))
    if n_exp == 1: axes = [axes]
    fig.suptitle("Interval Width vs. LCA Posterior Value\n"
                 "Ambiguous posteriors (near 0.5) should yield wider intervals",
                 fontsize=12)

    for row_axes, (exp_name, results) in zip(axes, experiments.items()):
        for ax, (pathogen, pcol, w) in zip(row_axes, [
            ("BAC",  "bac_post_prob",  results["widths_bac"]),
            ("FUNG", "fung_post_prob", results["widths_fung"])
        ]):
            post = df_eval[pcol].values
            bidx = np.clip(np.digitize(post, bins)-1, 0, len(blabels)-1)
            means = [w[bidx==b].mean() if (bidx==b).sum()>0 else np.nan
                     for b in range(len(blabels))]
            ns    = [(bidx==b).sum() for b in range(len(blabels))]

            x = np.arange(len(blabels))
            valid = ~np.isnan(means)
            ax.bar(x[valid], np.array(means)[valid],
                   color="#1B8A8F", alpha=0.75)
            ax.set_xticks(x)
            ax.set_xticklabels(blabels, rotation=45, ha="right", fontsize=8)
            ax.set_title(f"{exp_name} — {pathogen}", fontsize=9)
            ax.set_ylabel("Mean Width"); ax.grid(alpha=0.3)
            for xi, n in zip(x, ns):
                if n > 0:
                    ax.text(xi, 0.001, f"n={n}", ha="center",
                            va="bottom", fontsize=7, rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reg_width_vs_posterior.png"), dpi=150)
    plt.close()
    print("  Saved: reg_width_vs_posterior.png")


# =============================================================================
# ANALYSIS: ACCURACY WITHIN COVERED VS NOT COVERED
# =============================================================================

def analysis_coverage_vs_accuracy(experiments, sig_probs_eval,
                                   soft_eval, out_dir):
    phdr("ACCURACY: JOINTLY COVERED vs. NOT COVERED")
    bac_bin  = (soft_eval[:,0] > 0.5).astype(int)
    fung_bin = (soft_eval[:,1] > 0.5).astype(int)
    labels_4 = bac_bin + fung_bin*2
    preds_4  = np.stack([
        (1-sig_probs_eval[:,0])*(1-sig_probs_eval[:,1]),
        sig_probs_eval[:,0]*(1-sig_probs_eval[:,1]),
        (1-sig_probs_eval[:,0])*sig_probs_eval[:,1],
        sig_probs_eval[:,0]*sig_probs_eval[:,1],
    ], axis=1).argmax(axis=1)
    correct = (preds_4 == labels_4)

    rows = []
    for exp_name, results in experiments.items():
        covered = results["covered_joint"]
        n_cov   = covered.sum()
        n_not   = (~covered).sum()
        acc_cov = correct[covered].mean()  if n_cov > 0 else float("nan")
        acc_not = correct[~covered].mean() if n_not > 0 else float("nan")
        gap     = acc_cov - acc_not
        print(f"  {exp_name}")
        print(f"    Covered    (n={n_cov:4d}): accuracy={acc_cov*100:.1f}%")
        print(f"    Not covered(n={n_not:4d}): accuracy={acc_not*100:.1f}%")
        print(f"    Gap: {gap*100:+.1f}pp  "
              f"({'covered more accurate ✓' if gap>0 else 'covered less accurate ✗'})")
        rows.append({"Experiment": exp_name, "Acc_covered": acc_cov,
                     "Acc_not_covered": acc_not, "Gap_pp": gap*100,
                     "N_covered": int(n_cov), "N_not_covered": int(n_not)})

    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "reg_coverage_vs_accuracy.csv"), index=False)


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = get_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    if args.out_dir is None:
        args.out_dir = os.path.join(
            os.path.dirname(args.weights), "conformal_regression")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output directory: {args.out_dir}\n")
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    df_test, cal_idx, eval_idx = load_test_set(
        args.csv, args.img_root, args.seed)
    df_eval = df_test.iloc[eval_idx].reset_index(drop=True)
    loader  = DataLoader(KeratitisDataset(df_test, val_tf),
                         batch_size=args.batch, shuffle=False,
                         num_workers=4, pin_memory=True)

    # ── Inference with caching ────────────────────────────────────────────────
    print("Loading Cell A (sigmoid + soft)...")
    m_a = load_sig(args.weights, device)
    probs_a, soft_labels = infer(m_a, loader, device,
                                  os.path.join(args.out_dir, "cache_sig_soft.npz"),
                                  out_key="probs")

    probs_c = None
    if args.weights_hard:
        print("Loading Cell C (sigmoid + hard)...")
        m_c = load_sig(args.weights_hard, device)
        probs_c, _ = infer(m_c, loader, device,
                            os.path.join(args.out_dir, "cache_sig_hard.npz"),
                            out_key="probs")

    bd_params = None
    if args.bd_weights:
        print("Loading Cell B (BD + soft)...")
        m_b = load_bd_model(args.bd_weights, device)
        bd_params, _ = infer(m_b, loader, device,
                              os.path.join(args.out_dir, "cache_bd_soft.npz"),
                              out_key="params")

    # Soft targets are always the regression target
    soft_targets = soft_labels
    soft_cal     = soft_targets[cal_idx]
    soft_eval    = soft_targets[eval_idx]
    bd_cal       = bd_params[cal_idx]  if bd_params is not None else None
    bd_eval      = bd_params[eval_idx] if bd_params is not None else None

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 1 — Sigmoid only, Chebyshev score
    # ══════════════════════════════════════════════════════════════════════════
    phdr("EXPERIMENT 1 — SIGMOID ONLY  (Chebyshev score, label comparison)")
    print("  Score: s_i = max(|y_bac - p_bac|, |y_fung - p_fung|)")
    print("  Question: does soft label training -> narrower intervals?\n")

    exp1_results = {}

    # Cell A: sigmoid + soft
    s1_cal_a   = chebyshev_scores(probs_a[cal_idx], soft_cal)
    q1_a       = conformal_quantile(s1_cal_a, args.alpha)
    lo1a, hi1a, w1a = intervals_chebyshev(probs_a[eval_idx], q1_a)
    m1a = compute_metrics(lo1a, hi1a, soft_eval, w1a)
    m1a["q_hat"] = q1_a
    exp1_results["Exp1: Cell A (Sigmoid+Soft)"] = m1a
    print_result("Cell A  sigmoid + soft", q1_a, m1a, args.alpha)

    # Cell C: sigmoid + hard
    if probs_c is not None:
        s1_cal_c = chebyshev_scores(probs_c[cal_idx], soft_cal)
        q1_c     = conformal_quantile(s1_cal_c, args.alpha)
        lo1c, hi1c, w1c = intervals_chebyshev(probs_c[eval_idx], q1_c)
        m1c = compute_metrics(lo1c, hi1c, soft_eval, w1c)
        m1c["q_hat"] = q1_c
        exp1_results["Exp1: Cell C (Sigmoid+Hard)"] = m1c
        print_result("Cell C  sigmoid + hard", q1_c, m1c, args.alpha)

        # Direct comparison
        for pathogen, wa, wc in [
            ("BAC",  m1a["mean_width_bac"],  m1c["mean_width_bac"]),
            ("FUNG", m1a["mean_width_fung"], m1c["mean_width_fung"])
        ]:
            diff = wc - wa
            print(f"\n  {pathogen}: soft={wa:.4f}  hard={wc:.4f}  "
                  f"diff={diff:+.4f}  "
                  f"({'soft narrower ✓' if diff>0 else 'hard narrower ✗'})")

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 2 — Sigmoid + BD combined, BD-scaled score
    # Note: run after Experiment 1 is validated. Uncomment when ready.
    # ══════════════════════════════════════════════════════════════════════════
    exp2_results = {}
    if bd_params is not None:  # Experiment 2 enabled
        phdr("EXPERIMENT 2 — SIGMOID + BD COMBINED  (BD-scaled score)")
        print("  Score: s_i = max(|y_bac-p_bac|/(sigma_bac+eps),")
        print("                   |y_fung-p_fung|/(sigma_fung+eps))")
        print("  sigma = 1/(alpha+beta)  — inverse BD evidence")
        print("  Question: does BD uncertainty improve sigmoid intervals?\n")

        s2_cal   = bd_scaled_scores(probs_a[cal_idx], soft_cal, bd_cal)
        q2       = conformal_quantile(s2_cal, args.alpha)
        lo2, hi2, w2 = intervals_bd_scaled(probs_a[eval_idx], q2, bd_eval)
        m2 = compute_metrics(lo2, hi2, soft_eval, w2)
        m2["q_hat"] = q2
        exp2_results["Exp2: Cell A+B (BD-Scaled)"] = m2
        print_result("Cell A+B  BD-scaled (adaptive)", q2, m2, args.alpha)

        # Compare adaptive vs non-adaptive for Cell A
        print(f"\n  Adaptivity (std of widths):")
        print(f"    Exp1 Cell A (non-adaptive): "
              f"BAC std={m1a['std_width_bac']:.4f}  "
              f"FUNG std={m1a['std_width_fung']:.4f}")
        print(f"    Exp2 Cell A+B (adaptive):   "
              f"BAC std={m2['std_width_bac']:.4f}  "
              f"FUNG std={m2['std_width_fung']:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 3 — BD standalone, Beta density score
    # Note: run after Experiment 1 is validated. Uncomment when ready.
    # ══════════════════════════════════════════════════════════════════════════
    exp3_results = {}
    if bd_params is not None:  # Experiment 3 enabled
        phdr("EXPERIMENT 3 — BD STANDALONE  (Beta density score, novel)")
        print("  Score: s_i = max(-log f_Beta(y_bac; a_bac, b_bac),")
        print("                   -log f_Beta(y_fung; a_fung, b_fung))")
        print("  Intervals: highest-density region (HDR) of each Beta")
        print("  Question: can BD distributional output directly give")
        print("             valid conformal intervals? (novel)\n")

        s3_cal = beta_density_scores(soft_cal, bd_cal)
        q3     = conformal_quantile(s3_cal, args.alpha)
        print(f"  q_hat (log-likelihood space) = {q3:.4f}")
        print(f"  Building HDR intervals (numerical)...")
        lo3, hi3, w3 = intervals_beta_hdr(bd_eval, q3)
        m3 = compute_metrics(lo3, hi3, soft_eval, w3)
        m3["q_hat"] = q3
        exp3_results["Exp3: Cell B (Beta Density)"] = m3
        print_result("Cell B  BD standalone (HDR)", q3, m3, args.alpha)

    # ══════════════════════════════════════════════════════════════════════════
    # ALL ANALYSES
    # ══════════════════════════════════════════════════════════════════════════
    all_experiments = {**exp1_results, **exp2_results, **exp3_results}

    print("\nRunning analyses...")
    analysis_by_study(all_experiments, df_eval, args.alpha, args.out_dir)
    analysis_width_vs_posterior(all_experiments, df_eval, args.out_dir)
    analysis_coverage_vs_accuracy(all_experiments,
                                   probs_a[eval_idx], soft_eval, args.out_dir)
    if bd_eval is not None:
        analysis_width_vs_evidence(all_experiments, bd_eval, args.out_dir)

    # ── Save summary ──────────────────────────────────────────────────────────
    def serialize(obj):
        if isinstance(obj, dict):
            return {str(k): serialize(v) for k,v in obj.items()}
        if isinstance(obj, np.ndarray):
            return {"mean": float(obj.mean()), "std": float(obj.std()),
                    "min":  float(obj.min()),  "max": float(obj.max())}
        if isinstance(obj, (np.floating, float)):
            return None if (obj != obj) else float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, bool): return obj
        return obj

    summary = {name: serialize({k: v for k,v in res.items()
                                 if k not in ("lo","hi","covered_joint",
                                              "widths_bac","widths_fung")})
               for name, res in all_experiments.items()}

    with open(os.path.join(args.out_dir, "reg_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs saved to: {args.out_dir}/")
    print("  reg_results.json")
    print("  reg_width_by_study.csv / .png")
    print("  reg_evidence_correlation.csv")
    print("  reg_width_vs_evidence.png")
    print("  reg_width_vs_posterior.png")
    print("  reg_coverage_vs_accuracy.csv")


if __name__ == "__main__":
    main()