"""
Linear Probe: Can uncertainty signals predict the true class?
=============================================================
Fits a multinomial logistic regression (linear probe) on top of the
uncertainty representations from both models to predict the true 4-class
label {neither=0, bac-only=1, fung-only=2, both=3}.

Features (5 total):
    p_bac, p_fung   — Beta-Dirichlet predicted probabilities
    e_bac, e_fung   — Beta-Dirichlet evidence (uncertainty signal)
    CPSL            — Conformal Prediction Set Length (4-class Vanilla LAC)

The probe answers three questions:
    1. How well can you predict the true class from just these 5 numbers?
    2. Does adding CPSL to BD features improve prediction over BD alone?
    3. Which features matter most? (coefficients)

Baseline comparison:
    - Sigmoid probabilities only [p_bac_sig, p_fung_sig]
    - BD probabilities only      [p_bac_bd, p_fung_bd]
    - BD full                    [p_bac_bd, p_fung_bd, e_bac, e_fung]
    - BD full + CPSL             [p_bac_bd, p_fung_bd, e_bac, e_fung, CPSL]

Run:
    python linear_probe.py
        --csv        /path/to/data.csv
        --img_root   /path/to/images
        --bd_weights runs/beta_dirichlet_.../best_model.pt
        --sig_weights runs/sigmoid_.../best_model.pt
        --out_dir    runs/beta_dirichlet_.../linear_probe
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, roc_auc_score)
import timm
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# ARGUMENTS
# =============================================================================

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",         type=str,
                        default="/persist/QTIM/Active/26-0096/Corneal_Ulcer_Image_Sets/Data/corneal_ulcer_study_data_7_24_2025_lambda.csv")
    parser.add_argument("--img_root",    type=str,
                        default="/pacs/OSHU/Corneal_Ulcer_Image_Sets")
    parser.add_argument("--bd_weights",  type=str, required=True)
    parser.add_argument("--sig_weights", type=str, required=True)
    parser.add_argument("--alpha",       type=float, default=0.1)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--out_dir",     type=str,   default=None)
    parser.add_argument("--cache_dir",   type=str,   default=None,
                        help="Directory with cached inference .npz files from dirichlet_analysis.py")
    return parser.parse_args()


# =============================================================================
# DATA
# =============================================================================

PACS_PREFIX = "/pacs/OSHU/Corneal_Ulcer_Image_Sets"
CLASS4_NAMES = ["Neither", "Bac only", "Fung only", "Both"]

def remap_path(p, img_root):
    return p.replace(PACS_PREFIX, img_root)


def load_test_set(csv_path, img_root, seed=42):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["file_path", "subject"])
    unlabelable = (df["fung_LCA_bin"] == -1) & (df["bac_LCA_bin"] == -1)
    df = df[~unlabelable].reset_index(drop=True)
    df["local_path"] = df["file_path"].apply(lambda p: remap_path(p, img_root))

    subjects = df["subject"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    _, temp_idx = next(gss.split(df, groups=subjects))
    df_temp = df.iloc[temp_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    _, test_idx = next(gss2.split(df_temp, groups=df_temp["subject"].values))
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)

    # Split test into train_probe / test_probe (50/50 within test set)
    # Use a different seed so it doesn't overlap with the conformal cal/eval split
    gss3 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed+1)
    train_idx, test_probe_idx = next(gss3.split(df_test, groups=df_test["subject"].values))

    df_probe_train = df_test.iloc[train_idx].reset_index(drop=True)
    df_probe_test  = df_test.iloc[test_probe_idx].reset_index(drop=True)

    print(f"Probe train: {len(df_probe_train):,} | Probe test: {len(df_probe_test):,}\n")
    return df_test, train_idx, test_probe_idx


class KeratitisDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            image = Image.open(row["local_path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (600, 600), color=0)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(
            [row["bac_post_prob"], row["fung_post_prob"]],
            dtype=torch.float32
        )
        return image, label

val_transforms = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# =============================================================================
# MODELS
# =============================================================================

class BetaDirichletModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b7.ra_in1k",
            pretrained=False, num_classes=0, global_pool="avg"
        )
        f = self.backbone.num_features
        self.trunk = nn.Sequential(
            nn.BatchNorm1d(f),
            nn.Linear(f, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 32), nn.ReLU(), nn.Dropout(0.3),
        )
        self.bac_head  = nn.Sequential(nn.Linear(32, 2), nn.Softplus())
        self.fung_head = nn.Sequential(nn.Linear(32, 2), nn.Softplus())

    def forward(self, x):
        f = self.trunk(self.backbone(x))
        return torch.cat([self.bac_head(f), self.fung_head(f)], dim=-1)


class SigmoidModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b7.ra_in1k",
            pretrained=False, num_classes=0, global_pool="avg"
        )
        f = self.backbone.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(f),
            nn.Linear(f, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 2), nn.Sigmoid()
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def load_bd(path, device):
    model = BetaDirichletModel()
    raw = torch.load(path, map_location=device)
    fixed = {(k[len("head."):] if k.startswith("head.") else k): v
             for k, v in raw.items()}
    model.load_state_dict(fixed)
    return model.to(device).eval()


def load_sig(path, device):
    model = SigmoidModel()
    raw = torch.load(path, map_location=device)
    fixed = {}
    for k, v in raw.items():
        if k.startswith("head.net."):
            fixed["head." + k[len("head.net."):]] = v
        elif k.startswith("backbone."):
            fixed[k] = v
    model.load_state_dict(fixed)
    return model.to(device).eval()


@torch.no_grad()
def run_inference(model, loader, device, is_bd=True, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cache: {cache_path}")
        cached = np.load(cache_path)
        if is_bd:
            return cached["params"], cached["labels"]
        return cached["probs"], cached["labels"]

    model.eval()
    all_out, all_labels = [], []
    for images, labels in tqdm(loader, desc="  Inference", leave=False):
        all_out.append(model(images.to(device)).cpu())
        all_labels.append(labels)

    out    = torch.cat(all_out).numpy()
    labels = torch.cat(all_labels).numpy()

    if cache_path:
        if is_bd:
            np.savez(cache_path, params=out, labels=labels)
        else:
            np.savez(cache_path, probs=out, labels=labels)
        print(f"  Saved cache: {cache_path}")

    return out, labels


# =============================================================================
# CONFORMAL (LAC) — 4-class
# =============================================================================

def compute_cpsl(sig_probs, labels, cal_idx, eval_idx, alpha):
    """Compute 4-class Vanilla LAC prediction set sizes."""
    sig_4 = np.stack([
        (1-sig_probs[:,0])*(1-sig_probs[:,1]),
        sig_probs[:,0]*(1-sig_probs[:,1]),
        (1-sig_probs[:,0])*sig_probs[:,1],
        sig_probs[:,0]*sig_probs[:,1],
    ], axis=1)

    bac_bin  = (labels[:, 0] > 0.5).astype(int)
    fung_bin = (labels[:, 1] > 0.5).astype(int)
    labels_4 = bac_bin * 1 + fung_bin * 2

    cal_p = sig_4[cal_idx]; cal_l = labels_4[cal_idx]
    eval_p = sig_4[eval_idx]

    n = len(cal_l)
    scores  = 1 - cal_p[np.arange(n), cal_l]
    q_level = min(np.ceil((n+1)*(1-alpha))/n, 1.0)
    qhat    = float(np.quantile(scores, q_level, method="higher"))
    psets   = eval_p >= (1 - qhat)
    cpsl    = psets.sum(axis=1)

    return cpsl, labels_4


# =============================================================================
# LINEAR PROBE
# =============================================================================

def run_probe(X_train, y_train, X_test, y_test, name, classes=4):
    """
    Fit multinomial logistic regression and evaluate.
    Returns dict of metrics.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0
    )
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # Per-class AUROC (one-vs-rest)
    try:
        present = sorted(np.unique(y_test))
        auroc = roc_auc_score(y_test, y_prob[:, present], multi_class="ovr",
                              average="macro", labels=present)
    except Exception:
        auroc = float("nan")

    print(f"\n  {'─'*50}")
    print(f"  {name}")
    print(f"  {'─'*50}")
    print(f"  Accuracy:      {acc*100:.1f}%")
    print(f"  Macro F1:      {f1:.4f}")
    print(f"  Macro AUROC:   {auroc:.4f}")
    # Only report classes present in this split
    present_classes = sorted(np.unique(np.concatenate([y_train, y_test])))
    present_names   = [CLASS4_NAMES[i] for i in present_classes]
    print(f"\n  {classification_report(y_test, y_pred, labels=present_classes, target_names=present_names, zero_division=0)}")

    return {
        "name":    name,
        "accuracy": acc,
        "macro_f1": f1,
        "macro_auroc": auroc,
        "coef":    clf.coef_.tolist(),
        "classes": clf.classes_.tolist(),
        "scaler_mean":  scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }



def run_correctness_probe(X_train, y_train_correct, X_test, y_test_correct,
                           name, feature_names):
    """
    Binary probe: predict whether the model is CORRECT on this sample.
    Target: 1 = model got it right, 0 = model got it wrong.

    The key metrics here are not just accuracy but:
    - Precision at high confidence: when the probe says "trust this prediction",
      how often is the model actually right?
    - Coverage: what fraction of cases does the probe say "trust"?
    - The precision/coverage tradeoff across thresholds.

    This is a selective classifier — we want to find a threshold that
    identifies the high-accuracy "doubly confident" region.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (precision_recall_curve, roc_auc_score,
                                  average_precision_score, classification_report)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(X_tr, y_train_correct)
    y_prob = clf.predict_proba(X_te)[:, 1]   # P(correct)
    y_pred = clf.predict(X_te)

    auroc = roc_auc_score(y_test_correct, y_prob)
    ap    = average_precision_score(y_test_correct, y_prob)

    print(f"\n  {chr(8212)*50}")
    print(f"  CORRECTNESS PROBE: {name}")
    print(f"  {chr(8212)*50}")
    print(f"  AUROC (correct vs wrong): {auroc:.4f}")
    print(f"  Avg Precision:            {ap:.4f}")
    print(f"  Base rate (% correct):    {y_test_correct.mean()*100:.1f}%")

    # Precision/coverage at various thresholds
    print(f"\n  Precision/Coverage tradeoff (threshold = P(correct) > t):")
    print(f"  {'Threshold':>10} {'Coverage':>10} {'Precision':>10} {'N trusted':>10}")
    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        trusted = y_prob >= thresh
        if trusted.sum() == 0:
            continue
        prec = y_test_correct[trusted].mean()
        cov  = trusted.mean()
        print(f"  {thresh:>10.1f} {cov*100:>9.1f}% {prec*100:>9.1f}% {trusted.sum():>10}")

    # Coefficient interpretation
    print(f"\n  Feature importances (positive = predicts correctness):")
    coef = clf.coef_[0]
    for fname, c in sorted(zip(feature_names, coef), key=lambda x: -abs(x[1])):
        bar = chr(9608) * int(abs(c)*10)
        sign = "+" if c>0 else "-"
        print(f"    {fname:<15} {sign}{abs(c):.3f}  {bar}")

    return {
        "name": name,
        "auroc": float(auroc),
        "avg_precision": float(ap),
        "base_rate": float(y_test_correct.mean()),
        "coef": {fn: float(c) for fn, c in zip(feature_names, coef)},
        "precision_coverage": [
            {
                "threshold": float(t),
                "coverage":  float((y_prob>=t).mean()),
                "precision": float(y_test_correct[y_prob>=t].mean()) if (y_prob>=t).sum()>0 else None,
                "n_trusted": int((y_prob>=t).sum()),
            }
            for t in [0.5, 0.6, 0.7, 0.8, 0.9]
        ],
        "prob_correct": y_prob.tolist(),
    }


def plot_precision_coverage(correctness_results, out_dir):
    """
    Precision vs. coverage curves for all feature sets.
    Shows the selective classification tradeoff:
    higher threshold = fewer cases trusted = higher precision.
    """
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Selective Classification: Can Uncertainty Signals Identify Trustworthy Predictions?\n"
        "Higher precision at same coverage = better uncertainty signal",
        fontsize=12
    )

    colors = {"Sigmoid only": "#718096", "BD probs only": "#A0AEC0",
              "BD full": "#4A5568", "BD full + CPSL": "#1B8A8F"}

    ax1, ax2 = axes

    for r in correctness_results:
        name  = r["name"]
        probs = np.array(r["prob_correct"])
        true  = r["_y_test"]
        col   = colors.get(name, "#333333")

        # Precision-recall curve (recall = coverage here)
        from sklearn.metrics import precision_recall_curve
        prec, rec, thresh = precision_recall_curve(true, probs)

        ax1.plot(rec, prec, label=f"{name} (AUROC={r['auroc']:.3f})", color=col, linewidth=2)
        ax1.axhline(r["base_rate"], color=col, ls=":", alpha=0.4)

        # Coverage vs precision at fixed thresholds
        pc = [(d["coverage"], d["precision"]) for d in r["precision_coverage"]
              if d["precision"] is not None]
        if pc:
            covs, precs = zip(*pc)
            ax2.plot(covs, precs, marker="o", label=name, color=col, linewidth=2)

    ax1.set_xlabel("Coverage (fraction of cases trusted)", fontsize=10)
    ax1.set_ylabel("Precision (accuracy on trusted cases)", fontsize=10)
    ax1.set_title("Precision-Coverage Curve", fontsize=11)
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax1.set_xlim(0,1); ax1.set_ylim(0.5,1.05)
    ax1.axhline(0.98, color="green", ls="--", alpha=0.5, label="98% target")

    ax2.set_xlabel("Coverage (fraction trusted)", fontsize=10)
    ax2.set_ylabel("Precision (accuracy on trusted)", fontsize=10)
    ax2.set_title("Precision at Fixed Thresholds", fontsize=11)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    ax2.set_xlim(0,1); ax2.set_ylim(0.5,1.05)
    ax2.axhline(0.98, color="green", ls="--", alpha=0.5, label="98% target")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "correctness_probe_curves.png"), dpi=150)
    plt.close()
    print("  Saved: correctness_probe_curves.png")


def plot_coefficients(results, feature_sets, out_dir):
    """
    Plot logistic regression coefficients for the full model
    (BD full + CPSL) to show which features drive each class prediction.
    """
    full_result = next((r for r in results if "CPSL" in r["name"]), None)
    if full_result is None:
        return

    coef   = np.array(full_result["coef"])   # (n_classes, n_features)
    n_feat = coef.shape[1]
    # Feature names for the full model
    feat_names = feature_sets["BD full + CPSL"]

    fig, axes = plt.subplots(1, len(CLASS4_NAMES), figsize=(14, 5))
    fig.suptitle("Linear Probe Coefficients (BD full + CPSL)\n"
                 "Which features drive each class prediction?", fontsize=13)

    for i, (ax, cls_name) in enumerate(zip(axes, CLASS4_NAMES)):
        c = coef[i] if i < len(coef) else np.zeros(n_feat)
        colors = ["#2D9E6B" if v > 0 else "#E05252" for v in c]
        bars = ax.barh(feat_names, c, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(cls_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Coefficient", fontsize=9)
        ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "probe_coefficients.png"), dpi=150)
    plt.close()
    print("  Saved: probe_coefficients.png")


def plot_accuracy_comparison(results, out_dir):
    """Bar chart comparing accuracy across all probe configurations."""
    names = [r["name"] for r in results]
    accs  = [r["accuracy"] * 100 for r in results]
    aurocs = [r["macro_auroc"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Linear Probe Comparison: Does combining signals help?", fontsize=13)

    colors = ["#718096"] * len(names)
    colors[-1] = "#1B8A8F"  # highlight full model

    for ax, vals, label, fmt in zip(
        axes,
        [accs, aurocs],
        ["Accuracy (%)", "Macro AUROC"],
        [".1f", ".3f"]
    ):
        bars = ax.bar(range(len(names)), vals, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(label)
        ax.set_ylim(min(vals) * 0.95, max(vals) * 1.05)
        ax.grid(alpha=0.3, axis="y")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.002 * max(vals),
                    f"{val:{fmt}}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "probe_accuracy_comparison.png"), dpi=150)
    plt.close()
    print("  Saved: probe_accuracy_comparison.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.out_dir is None:
        args.out_dir = os.path.join(
            os.path.dirname(args.bd_weights), "linear_probe"
        )
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output directory: {args.out_dir}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- Data ---
    df_test, train_idx, test_probe_idx = load_test_set(
        args.csv, args.img_root, seed=args.seed
    )
    loader = DataLoader(
        KeratitisDataset(df_test, val_transforms),
        batch_size=8, shuffle=False, num_workers=4, pin_memory=True
    )

    # --- Inference (with caching) ---
    cache_dir = args.cache_dir or args.out_dir
    bd_cache  = os.path.join(cache_dir, "cache_bd.npz")
    sig_cache = os.path.join(cache_dir, "cache_sig.npz")

    print("Loading Beta-Dirichlet model...")
    bd_model = load_bd(args.bd_weights, device)
    bd_params, bd_labels = run_inference(
        bd_model, loader, device, is_bd=True, cache_path=bd_cache
    )

    print("Loading Sigmoid model...")
    sig_model = load_sig(args.sig_weights, device)
    sig_probs, _ = run_inference(
        sig_model, loader, device, is_bd=False, cache_path=sig_cache
    )

    # --- Derived BD quantities ---
    alpha_bac  = bd_params[:, 0]; beta_bac   = bd_params[:, 1]
    alpha_fung = bd_params[:, 2]; beta_fung  = bd_params[:, 3]
    e_bac  = alpha_bac + beta_bac
    e_fung = alpha_fung + beta_fung
    p_bac  = alpha_bac / e_bac
    p_fung = alpha_fung / e_fung

    # --- 4-class labels ---
    bac_bin  = (bd_labels[:, 0] > 0.5).astype(int)
    fung_bin = (bd_labels[:, 1] > 0.5).astype(int)
    labels_4 = bac_bin * 1 + fung_bin * 2

    # --- Conformal set sizes on full test set ---
    # Use train_idx as conformal calibration, test_probe_idx as eval
    # (different from the conformal.py split to avoid leakage)
    cpsl, _ = compute_cpsl(sig_probs, bd_labels, train_idx, test_probe_idx, args.alpha)

    # --- Build feature matrices ---
    # Only on probe test indices
    idx = test_probe_idx
    y = labels_4[idx]

    feature_sets = {
        "Sigmoid only":    ["p_bac_sig", "p_fung_sig"],
        "BD probs only":   ["p_bac_bd", "p_fung_bd"],
        "BD full":         ["p_bac_bd", "p_fung_bd", "e_bac", "e_fung"],
        "BD full + CPSL":  ["p_bac_bd", "p_fung_bd", "e_bac", "e_fung", "CPSL"],
    }

    # All features on full test set
    all_features = np.column_stack([
        sig_probs[:, 0], sig_probs[:, 1],   # sigmoid probs
        p_bac, p_fung,                       # BD probs
        e_bac, e_fung,                       # BD evidence
    ])
    # CPSL only available on test_probe_idx subset
    cpsl_full = np.full(len(labels_4), np.nan)
    cpsl_full[test_probe_idx] = cpsl

    # Probe train/test split (use train_idx for training the probe)
    train_features = np.column_stack([
        sig_probs[train_idx, 0], sig_probs[train_idx, 1],
        p_bac[train_idx], p_fung[train_idx],
        e_bac[train_idx], e_fung[train_idx],
    ])
    test_features = np.column_stack([
        sig_probs[idx, 0], sig_probs[idx, 1],
        p_bac[idx], p_fung[idx],
        e_bac[idx], e_fung[idx],
    ])
    y_train = labels_4[train_idx]

    # CPSL for train (use conformal on train_idx — self-calibration, pessimistic)
    cpsl_train, _ = compute_cpsl(sig_probs, bd_labels,
                                  test_probe_idx, train_idx, args.alpha)

    X_sets = {
        "Sigmoid only":    (train_features[:, :2],        test_features[:, :2]),
        "BD probs only":   (train_features[:, 2:4],       test_features[:, 2:4]),
        "BD full":         (train_features[:, 2:6],       test_features[:, 2:6]),
        "BD full + CPSL":  (
            np.column_stack([train_features[:, 2:6], cpsl_train.reshape(-1,1)]),
            np.column_stack([test_features[:,  2:6], cpsl.reshape(-1,1)]),
        ),
    }

    # --- Run probes ---
    print_header = lambda t: print(f"\n{'='*55}\n  {t}\n{'='*55}")
    print_header("LINEAR PROBE RESULTS (4-class classification)")
    print(f"  Probe train: {len(train_idx):,}  |  Probe test: {len(idx):,}")
    print(f"  Classes: {CLASS4_NAMES}")
    print(f"  Label distribution (test): "
          f"{ {CLASS4_NAMES[k]: int((y==k).sum()) for k in range(4)} }")

    results = []
    for name, (X_tr, X_te) in X_sets.items():
        r = run_probe(X_tr, y_train, X_te, y, name)
        results.append(r)

    # --- Plots ---
    plot_accuracy_comparison(results, args.out_dir)
    plot_coefficients(results, feature_sets, args.out_dir)

    # --- Correctness probe ---
    print("\n" + "="*55)
    print("  CORRECTNESS PROBE: Can we predict when the model is right?")
    print("="*55)

    # True correctness on probe test set
    sig4_test  = np.stack([
        (1-sig_probs[idx,0])*(1-sig_probs[idx,1]),
         sig_probs[idx,0]   *(1-sig_probs[idx,1]),
        (1-sig_probs[idx,0])* sig_probs[idx,1],
         sig_probs[idx,0]   * sig_probs[idx,1],
    ], axis=1)
    bac_bin_te  = (bd_labels[idx,0]>0.5).astype(int)
    fng_bin_te  = (bd_labels[idx,1]>0.5).astype(int)
    lab4_te     = bac_bin_te + fng_bin_te*2
    correct_te  = (sig4_test.argmax(axis=1) == lab4_te).astype(int)

    sig4_train  = np.stack([
        (1-sig_probs[train_idx,0])*(1-sig_probs[train_idx,1]),
         sig_probs[train_idx,0]   *(1-sig_probs[train_idx,1]),
        (1-sig_probs[train_idx,0])* sig_probs[train_idx,1],
         sig_probs[train_idx,0]   * sig_probs[train_idx,1],
    ], axis=1)
    bac_bin_tr  = (bd_labels[train_idx,0]>0.5).astype(int)
    fng_bin_tr  = (bd_labels[train_idx,1]>0.5).astype(int)
    lab4_tr     = bac_bin_tr + fng_bin_tr*2
    correct_tr  = (sig4_train.argmax(axis=1) == lab4_tr).astype(int)

    print(f"  Base rate (overall accuracy): {correct_te.mean()*100:.1f}%")

    correctness_results = []
    feat_configs = [
        ("Sigmoid only",   ["p_bac_sig","p_fung_sig"],
         train_features[:,:2], test_features[:,:2]),
        ("BD probs only",  ["p_bac_bd","p_fung_bd"],
         train_features[:,2:4], test_features[:,2:4]),
        ("BD full",        ["p_bac_bd","p_fung_bd","e_bac","e_fung"],
         train_features[:,2:6], test_features[:,2:6]),
        ("BD full + CPSL", ["p_bac_bd","p_fung_bd","e_bac","e_fung","CPSL"],
         np.column_stack([train_features[:,2:6], cpsl_train.reshape(-1,1)]),
         np.column_stack([test_features[:,2:6],  cpsl.reshape(-1,1)])),
    ]
    for name, fnames, X_tr, X_te in feat_configs:
        r = run_correctness_probe(X_tr, correct_tr, X_te, correct_te, name, fnames)
        r["_y_test"] = correct_te   # store for plotting
        correctness_results.append(r)

    plot_precision_coverage(correctness_results, args.out_dir)

    # Save correctness results (strip non-serializable arrays)
    save_corr = [{k:v for k,v in r.items() if k not in ("_y_test","prob_correct")}
                 for r in correctness_results]
    with open(os.path.join(args.out_dir, "correctness_probe_results.json"), "w") as f:
        json.dump(save_corr, f, indent=2)
    print("  Saved: correctness_probe_results.json")

    # --- Save results ---
    out = {r["name"]: {k: v for k, v in r.items()
                       if k not in ("coef", "scaler_mean", "scaler_scale")}
           for r in results}
    with open(os.path.join(args.out_dir, "probe_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nAll outputs saved to: {args.out_dir}/")
    print("  probe_results.json")
    print("  probe_accuracy_comparison.png")
    print("  probe_coefficients.png")


if __name__ == "__main__":
    main()