"""
Conformal Prediction Analysis — Infectious Keratitis
=====================================================
Applies split conformal prediction (LAC only) to the trained sigmoid EfficientNetB7.

Three LAC variants:
    1. Vanilla LAC      — marginal coverage guarantee (standard)
    2. Group-balanced   — per-study coverage guarantee (Section 4.1, Angelopoulos & Bates)
    3. Class-conditional— per-label coverage guarantee (Section 4.2, Angelopoulos & Bates)

Two framings:
    2-class: independent binary LAC for bacterial and fungal separately
    4-class: joint label space {neither=0, bac-only=1, fung-only=2, both=3}
             built from factored joint distribution:
                P(neither)  = (1-bac) * (1-fung)
                P(bac-only) = bac     * (1-fung)
                P(fung-only)= (1-bac) * fung
                P(both)     = bac     * fung

Calibration: 20% of test set (subject-level split)
Evaluation:  remaining 80% of test set

Analyses (run for all variant x framing combinations):
    1. Empirical coverage vs. theoretical guarantee
    2. Prediction set size distribution
    3. Coverage rate vs. set size
    4. Set size by study
    5. Set size vs. LCA posterior probability
    6. Set size vs. LCA bin (MADURAI vs. single-etiology)

Install:
    pip install torch torchvision timm pillow numpy pandas scikit-learn tqdm matplotlib

Run:
    python conformal.py
        --csv      /path/to/data.csv
        --img_root /path/to/images
        --weights  runs/sigmoid_.../best_model.pt
        --alpha    0.1
        --out_dir  runs/sigmoid_.../conformal
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
    parser.add_argument("--csv",      type=str,
                        default="/persist/QTIM/Active/26-0096/Corneal_Ulcer_Image_Sets/Data/corneal_ulcer_study_data_7_24_2025_lambda.csv")
    parser.add_argument("--img_root", type=str,
                        default="/pacs/OSHU/Corneal_Ulcer_Image_Sets")
    parser.add_argument("--weights",  type=str, required=True)
    parser.add_argument("--alpha",    type=float, default=0.1)
    parser.add_argument("--head",     type=str,   default="sigmoid",
                        choices=["sigmoid", "beta_dirichlet"],
                        help="sigmoid: model outputs [p_bac, p_fung] directly. "
                             "beta_dirichlet: model outputs [a_bac, b_bac, a_fung, b_fung]; "
                             "converted to [p_bac, p_fung] via a/(a+b) before conformal.")
    parser.add_argument("--batch",    type=int, default=8)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--out_dir",  type=str, default=None)
    return parser.parse_args()


# =============================================================================
# DATA
# =============================================================================

PACS_PREFIX = "/pacs/OSHU/Corneal_Ulcer_Image_Sets"

def remap_path(p, img_root):
    return p.replace(PACS_PREFIX, img_root)


def load_and_split(csv_path, img_root, seed=42):
    """Reproduce exact 70/15/15 split, then split test 20/80 cal/eval."""
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

    gss3 = GroupShuffleSplit(n_splits=1, test_size=0.80, random_state=seed)
    cal_idx, eval_idx = next(gss3.split(df_test, groups=df_test["subject"].values))
    df_cal  = df_test.iloc[cal_idx].reset_index(drop=True)
    df_eval = df_test.iloc[eval_idx].reset_index(drop=True)

    print(f"Calibration: {len(df_cal):,} images ({df_cal['subject'].nunique():,} subjects)")
    print(f"Evaluation:  {len(df_eval):,} images ({df_eval['subject'].nunique():,} subjects)\n")
    return df_cal, df_eval


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
# MODEL
# =============================================================================

class SigmoidHead(nn.Module):
    def __init__(self, in_features, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(512, 32),          nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(32, 2), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)


class EfficientNetB7Sigmoid(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b7.ra_in1k",
            pretrained=False, num_classes=0, global_pool="avg"
        )
        self.head = SigmoidHead(self.backbone.num_features, dropout)

    def forward(self, x):
        return self.head(self.backbone(x))


class BetaDirichletModel(nn.Module):
    """
    Beta-Dirichlet head — outputs [alpha_bac, beta_bac, alpha_fung, beta_fung].
    Converted to [p_bac, p_fung] via p = alpha/(alpha+beta) before conformal.
    """
    def __init__(self, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b7.ra_in1k", pretrained=False,
            num_classes=0, global_pool="avg")
        f = self.backbone.num_features
        self.trunk = nn.Sequential(
            nn.BatchNorm1d(f), nn.Linear(f, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 32), nn.ReLU(), nn.Dropout(dropout))
        self.bac_head  = nn.Sequential(nn.Linear(32, 2), nn.Softplus())
        self.fung_head = nn.Sequential(nn.Linear(32, 2), nn.Softplus())

    def forward(self, x):
        f = self.trunk(self.backbone(x))
        return torch.cat([self.bac_head(f), self.fung_head(f)], dim=-1)


@torch.no_grad()
def get_predictions(model, loader, device, head="sigmoid"):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Inference", unit="batch"):
            all_probs.append(model(images.to(device)).cpu())
            all_labels.append(labels)
    raw    = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    if head == "beta_dirichlet":
        # Convert [alpha_bac, beta_bac, alpha_fung, beta_fung] -> [p_bac, p_fung]
        # p = alpha / (alpha + beta) — expected value of Beta distribution
        a_bac, b_bac   = raw[:, 0], raw[:, 1]
        a_fung, b_fung = raw[:, 2], raw[:, 3]
        probs = np.stack([a_bac  / (a_bac  + b_bac),
                          a_fung / (a_fung + b_fung)], axis=1)
        print(f"  BD->sigmoid: p_bac=[{probs[:,0].min():.3f},{probs[:,0].max():.3f}]  "
              f"p_fung=[{probs[:,1].min():.3f},{probs[:,1].max():.3f}]")
        return probs, labels

    return raw, labels


# =============================================================================
# LABEL HANDLING
# =============================================================================

CLASS2_NAMES  = ["Absent", "Present"]
CLASS4_NAMES  = ["Neither", "Bac only", "Fung only", "Both"]
STUDY_SHORT   = {
    "MADURAI photos":    "MADURAI",
    "MUTT2-Pictures":    "MUTT2",
    "MUTT1-Pictures":    "MUTT1",
    "SCUT photos":       "SCUT",
    "MUTT Pilot Photos": "MUTT Pilot",
    "CLAIR II photos":   "CLAIR II",
    "CLAIR I photos":    "CLAIR I",
}

def binarize(col, threshold=0.5):
    return (col > threshold).astype(int)

def to_4class_labels(bac_bin, fung_bin):
    """0=neither, 1=bac-only, 2=fung-only, 3=both"""
    return bac_bin * 1 + fung_bin * 2

def to_4class_probs(bac_prob, fung_prob):
    """Factored joint distribution assuming independence."""
    return np.stack([
        (1 - bac_prob) * (1 - fung_prob),   # neither
        bac_prob       * (1 - fung_prob),    # bac-only
        (1 - bac_prob) * fung_prob,          # fung-only
        bac_prob       * fung_prob,          # both
    ], axis=1)

def make_binary_probs(col):
    return np.stack([1 - col, col], axis=1)


# =============================================================================
# LAC CORE
# =============================================================================

def lac_calibration_scores(labels, probs):
    """s_i = 1 - p_hat[true class]"""
    n = len(labels)
    return 1 - probs[np.arange(n), labels]


def lac_quantile(scores, alpha, group_label=None):
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    if q_level > 1.0:
        label = f" [{group_label}]" if group_label else ""
        print(f"  WARNING{label}: n={n} too small for alpha={alpha} "
              f"(q_level={q_level:.3f} > 1.0). "
              f"Clamping to 1.0 — coverage guarantee does NOT hold for this group.")
        q_level = 1.0
    return float(np.quantile(scores, q_level, method="higher"))


def lac_prediction_sets(probs, qhat):
    """Include all classes with probability >= 1 - qhat."""
    return probs >= (1 - qhat)


def lac_metrics(pred_sets, labels, set_sizes=None):
    n = len(labels)
    covered   = pred_sets[np.arange(n), labels]
    set_sizes = pred_sets.sum(axis=1) if set_sizes is None else set_sizes
    K         = pred_sets.shape[1]
    return {
        "empirical_coverage": float(covered.mean()),
        "mean_set_size":      float(set_sizes.mean()),
        "set_size_dist":      {str(k): int((set_sizes == k).sum()) for k in range(K + 1)},
        "pred_sets":          pred_sets,
        "set_sizes":          set_sizes,
        "covered":            covered,
    }


# =============================================================================
# THREE LAC VARIANTS
# =============================================================================

def vanilla_lac(cal_probs, cal_labels, eval_probs, eval_labels, alpha):
    """
    Standard split conformal — one qhat for all samples.
    Guarantees: P(Y in C(X)) >= 1 - alpha   (marginal)
    """
    scores = lac_calibration_scores(cal_labels, cal_probs)
    qhat   = lac_quantile(scores, alpha)
    psets  = lac_prediction_sets(eval_probs, qhat)
    m      = lac_metrics(psets, eval_labels)
    m["qhat"]    = qhat
    m["variant"] = "Vanilla LAC"
    return m


def group_balanced_lac(cal_probs, cal_labels, cal_groups,
                       eval_probs, eval_labels, eval_groups,
                       alpha, group_names=None):
    """
    Group-balanced conformal (Section 4.1, Angelopoulos & Bates).
    Calibrate separately within each study group.
    Guarantees: P(Y in C(X) | study=g) >= 1 - alpha   for each group g.

    At test time the group is known (we know which study a patient comes from),
    so we apply the group-specific qhat.
    """
    unique_groups = np.unique(np.concatenate([cal_groups, eval_groups]))
    group_qhats   = {}

    for g in unique_groups:
        mask = cal_groups == g
        if mask.sum() == 0:
            # No calibration data for this group — fall back to global qhat
            global_scores  = lac_calibration_scores(cal_labels, cal_probs)
            group_qhats[g] = lac_quantile(global_scores, alpha)
        else:
            scores         = lac_calibration_scores(cal_labels[mask], cal_probs[mask])
            group_qhats[g] = lac_quantile(scores, alpha,
                                          group_label=group_names.get(g, g) if group_names else g)

    # Apply group-specific qhat to each eval sample
    K = cal_probs.shape[1]
    psets = np.zeros((len(eval_labels), K), dtype=bool)
    for g in unique_groups:
        mask      = eval_groups == g
        psets[mask] = lac_prediction_sets(eval_probs[mask], group_qhats[g])

    m = lac_metrics(psets, eval_labels)
    m["qhat"]        = group_qhats   # dict: group -> qhat
    m["variant"]     = "Group-Balanced LAC"
    m["group_qhats"] = {str(g): float(v) for g, v in group_qhats.items()}

    # Per-group coverage
    per_group = {}
    for g in unique_groups:
        mask = eval_groups == g
        if mask.sum() == 0:
            continue
        n_g   = mask.sum()
        label = group_names.get(g, g) if group_names else g
        covered_g = psets[mask][np.arange(n_g), eval_labels[mask]]
        per_group[str(label)] = {
            "n":        int(n_g),
            "coverage": float(covered_g.mean()),
            "qhat":     float(group_qhats[g]),
        }
    m["per_group_coverage"] = per_group
    return m


def class_conditional_lac(cal_probs, cal_labels, eval_probs, eval_labels, alpha):
    """
    Class-conditional conformal (Section 4.2, Angelopoulos & Bates).
    Calibrate separately within each true class.
    Guarantees: P(Y in C(X) | Y=y) >= 1 - alpha   for each class y.

    At test time the true class is unknown, so we include class k if its
    class-specific score is below its class-specific qhat.
    C(X) = {y : 1 - p_hat[y] <= qhat^(y)}
         = {y : p_hat[y] >= 1 - qhat^(y)}
    """
    K = cal_probs.shape[1]
    class_qhats = {}

    for k in range(K):
        mask = cal_labels == k
        if mask.sum() == 0:
            # No calibration samples of this class — use global fallback
            global_scores    = lac_calibration_scores(cal_labels, cal_probs)
            class_qhats[k]   = lac_quantile(global_scores, alpha)
        else:
            # Score for class k: 1 - p_hat[k] for samples where true label is k
            scores         = 1 - cal_probs[mask, k]
            class_qhats[k] = lac_quantile(scores, alpha)

    # Build prediction sets: include class k if p_hat[k] >= 1 - qhat^(k)
    psets = np.zeros((len(eval_probs), K), dtype=bool)
    for k in range(K):
        psets[:, k] = eval_probs[:, k] >= (1 - class_qhats[k])

    m = lac_metrics(psets, eval_labels)
    m["qhat"]         = class_qhats
    m["variant"]      = "Class-Conditional LAC"
    m["class_qhats"]  = {str(k): float(v) for k, v in class_qhats.items()}

    # Per-class coverage
    per_class = {}
    for k in range(K):
        mask = eval_labels == k
        if mask.sum() == 0:
            continue
        n_k      = mask.sum()
        covered_k = psets[mask][np.arange(n_k), eval_labels[mask]]
        per_class[str(k)] = {
            "n":        int(n_k),
            "coverage": float(covered_k.mean()),
            "qhat":     float(class_qhats[k]),
        }
    m["per_class_coverage"] = per_class
    return m


def run_all_variants(cal_probs, cal_labels, cal_groups,
                     eval_probs, eval_labels, eval_groups,
                     alpha, group_names=None):
    return {
        "Vanilla":           vanilla_lac(cal_probs, cal_labels,
                                         eval_probs, eval_labels, alpha),
        "Group-Balanced":    group_balanced_lac(cal_probs, cal_labels, cal_groups,
                                                eval_probs, eval_labels, eval_groups,
                                                alpha, group_names),
        "Class-Conditional": class_conditional_lac(cal_probs, cal_labels,
                                                    eval_probs, eval_labels, alpha),
    }


# =============================================================================
# ANALYSES
# =============================================================================

def print_header(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def analysis_coverage(all_results, alpha, out_dir):
    """Analysis 1: Empirical coverage vs. theoretical guarantee."""
    print_header(f"COVERAGE VALIDITY CHECK  (target: {(1-alpha)*100:.0f}%)")
    rows = []
    for framing, variants in all_results.items():
        for variant, m in variants.items():
            ec  = m["empirical_coverage"]
            gap = ec - (1 - alpha)
            print(f"  {framing:12s}  {variant:22s}: "
                  f"{ec*100:.1f}%  (gap {gap*100:+.1f}%)")
            rows.append({
                "Framing": framing, "Variant": variant,
                "Target":   f"{(1-alpha)*100:.1f}%",
                "Empirical": f"{ec*100:.1f}%",
                "Gap":       f"{gap*100:+.1f}%",
            })

            # Print per-group / per-class detail for balanced variants
            if "per_group_coverage" in m:
                for g, gm in m["per_group_coverage"].items():
                    gap_g = gm["coverage"] - (1 - alpha)
                    print(f"    [{g}]  n={gm['n']:>4}  "
                          f"coverage={gm['coverage']*100:.1f}%  "
                          f"qhat={gm['qhat']:.4f}  gap={gap_g*100:+.1f}%")
            if "per_class_coverage" in m:
                class_names = (CLASS4_NAMES if framing == "4-class"
                               else ["Absent", "Present"])
                for k, km in m["per_class_coverage"].items():
                    name  = class_names[int(k)] if int(k) < len(class_names) else k
                    gap_k = km["coverage"] - (1 - alpha)
                    print(f"    [Class {k}: {name}]  n={km['n']:>4}  "
                          f"coverage={km['coverage']*100:.1f}%  "
                          f"qhat={km['qhat']:.4f}  gap={gap_k*100:+.1f}%")

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "coverage.csv"), index=False)


def analysis_set_size_distribution(all_results, out_dir):
    """Analysis 2: Prediction set size distribution."""
    framings  = list(all_results.keys())
    variants  = list(next(iter(all_results.values())).keys())
    n_rows, n_cols = len(framings), len(variants)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 5 * n_rows))
    fig.suptitle("Prediction Set Size Distribution (LAC)", fontsize=14)

    for r, framing in enumerate(framings):
        for c, variant in enumerate(variants):
            ax = axes[r][c] if n_rows > 1 else axes[c]
            m  = all_results[framing][variant]
            K  = max(int(k) for k in m["set_size_dist"])
            sizes  = list(range(K + 1))
            counts = [m["set_size_dist"].get(str(s), 0) for s in sizes]
            total  = sum(counts)

            colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(sizes)))
            bars   = ax.bar(sizes, [c / total * 100 for c in counts], color=colors)
            ax.set_title(f"{framing} — {variant}", fontsize=10)
            ax.set_xlabel("Set Size")
            ax.set_ylabel("% samples")
            ax.set_xticks(sizes)
            for bar, cnt in zip(bars, counts):
                if cnt > 0:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.3,
                            f"n={cnt}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "set_size_distribution.png"), dpi=150)
    plt.close()
    print("  Saved: set_size_distribution.png")


def analysis_coverage_vs_set_size(all_results, out_dir):
    """Analysis 3: Coverage rate vs. prediction set size."""
    framings = list(all_results.keys())
    variants = list(next(iter(all_results.values())).keys())
    colors   = {"Vanilla": "#1f77b4",
                "Group-Balanced": "#ff7f0e",
                "Class-Conditional": "#2ca02c"}

    fig, axes = plt.subplots(1, len(framings),
                             figsize=(7 * len(framings), 5))
    if len(framings) == 1:
        axes = [axes]
    fig.suptitle("Coverage Rate vs. Prediction Set Size (LAC)", fontsize=14)

    for ax, framing in zip(axes, framings):
        for variant in variants:
            m         = all_results[framing][variant]
            set_sizes = m["set_sizes"]
            covered   = m["covered"]
            K         = max(int(k) for k in m["set_size_dist"])

            xs, ys, ns = [], [], []
            for s in range(K + 1):
                mask = set_sizes == s
                if mask.sum() == 0:
                    continue
                xs.append(s)
                ys.append(covered[mask].mean())
                ns.append(mask.sum())

            ax.plot(xs, ys, marker="o",
                    label=variant, color=colors[variant])
            for x, y, n in zip(xs, ys, ns):
                ax.annotate(f"n={n}", (x, y),
                            textcoords="offset points",
                            xytext=(5, 5), fontsize=7)

        ax.set_title(framing)
        ax.set_xlabel("Prediction Set Size")
        ax.set_ylabel("Coverage Rate")
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.4, label="Target (90%)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "coverage_vs_set_size.png"), dpi=150)
    plt.close()
    print("  Saved: coverage_vs_set_size.png")


def analysis_set_size_by_study(all_results, df_eval, out_dir):
    """Analysis 4: Set size by study."""
    studies   = df_eval["study"].values
    framings  = list(all_results.keys())
    variants  = list(next(iter(all_results.values())).keys())
    rows_out  = []

    fig, axes = plt.subplots(len(framings), len(variants),
                             figsize=(6 * len(variants), 5 * len(framings)))
    fig.suptitle("Mean Prediction Set Size by Study (LAC)", fontsize=14)

    for r, framing in enumerate(framings):
        for c, variant in enumerate(variants):
            ax = axes[r][c] if len(framings) > 1 else axes[c]
            m  = all_results[framing][variant]
            set_sizes = m["set_sizes"]
            K  = max(int(k) for k in m["set_size_dist"])

            study_means = {}
            for study in np.unique(studies):
                mask = studies == study
                mean = set_sizes[mask].mean()
                study_means[study] = mean
                rows_out.append({"Framing": framing, "Variant": variant,
                                 "Study": study, "Mean_set_size": mean})

            sorted_items = sorted(study_means.items(),
                                  key=lambda x: x[1], reverse=True)
            labels = [STUDY_SHORT.get(s, s) for s, _ in sorted_items]
            values = [v for _, v in sorted_items]

            bars = ax.barh(labels, values, color="#1f77b4")
            ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.5)
            ax.set_title(f"{framing} — {variant}", fontsize=10)
            ax.set_xlabel("Mean Set Size")
            ax.set_xlim(0, K + 0.3)
            for bar, val in zip(bars, values):
                ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                        f"{val:.2f}", va="center", fontsize=8)

    pd.DataFrame(rows_out).to_csv(
        os.path.join(out_dir, "set_size_by_study.csv"), index=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "set_size_by_study.png"), dpi=150)
    plt.close()
    print("  Saved: set_size_by_study.png")


def analysis_set_size_vs_posterior(all_results, df_eval, out_dir):
    """Analysis 5: Set size vs. mean LCA posterior probability."""
    framings  = list(all_results.keys())
    variants  = list(next(iter(all_results.values())).keys())
    bins      = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    bin_labels= ["0-0.1","0.1-0.3","0.3-0.5","0.5-0.7","0.7-0.9","0.9-1.0"]

    mean_post = (df_eval["bac_post_prob"].values +
                 df_eval["fung_post_prob"].values) / 2
    bin_idx   = np.clip(np.digitize(mean_post, bins) - 1,
                        0, len(bin_labels) - 1)

    fig, axes = plt.subplots(len(framings), len(variants),
                             figsize=(6 * len(variants), 5 * len(framings)))
    fig.suptitle("Set Size vs. LCA Posterior Probability (LAC)\n"
                 "(Uncertainty Alignment Check)", fontsize=14)

    for r, framing in enumerate(framings):
        for c, variant in enumerate(variants):
            ax = axes[r][c] if len(framings) > 1 else axes[c]
            m  = all_results[framing][variant]
            set_sizes = m["set_sizes"]
            K  = max(int(k) for k in m["set_size_dist"])

            means, stds, ns = [], [], []
            for b in range(len(bin_labels)):
                mask = bin_idx == b
                if mask.sum() > 0:
                    means.append(set_sizes[mask].mean())
                    stds.append(set_sizes[mask].std())
                    ns.append(mask.sum())
                else:
                    means.append(0); stds.append(0); ns.append(0)

            x = np.arange(len(bin_labels))
            ax.bar(x, means, yerr=stds, capsize=4, color="#1f77b4", alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=8)
            ax.set_title(f"{framing} — {variant}", fontsize=10)
            ax.set_xlabel("Mean Posterior Probability Bin")
            ax.set_ylabel("Mean Set Size")
            ax.set_ylim(0, K + 0.3)
            for xi, n in zip(x, ns):
                if n > 0:
                    ax.text(xi, 0.05, f"n={n}", ha="center",
                            va="bottom", fontsize=7, rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "set_size_vs_posterior.png"), dpi=150)
    plt.close()
    print("  Saved: set_size_vs_posterior.png")


def analysis_accuracy_vs_set_size(all_results, eval_probs_dict, eval_labels_dict, out_dir):
    """
    Point prediction accuracy vs. prediction set size.

    Accuracy = argmax of model probabilities matches true label.
    Stratified by prediction set size for each variant x framing.

    A well-calibrated model should show a clear downward trend:
    small sets (confident) -> high accuracy
    large sets (uncertain) -> lower accuracy
    """
    framings = list(all_results.keys())
    variants = list(next(iter(all_results.values())).keys())
    colors   = {"Vanilla": "#1f77b4",
                "Group-Balanced": "#ff7f0e",
                "Class-Conditional": "#2ca02c"}

    fig, axes = plt.subplots(1, len(framings),
                             figsize=(7 * len(framings), 5))
    if len(framings) == 1:
        axes = [axes]
    fig.suptitle("Point Prediction Accuracy vs. Prediction Set Size (LAC)\n"
                 "Downward trend = conformal sets are adaptive", fontsize=13)

    rows_out = []

    for ax, framing in zip(axes, framings):
        eval_probs  = eval_probs_dict[framing]
        eval_labels = eval_labels_dict[framing]

        # Point prediction: argmax of model probabilities
        point_preds = eval_probs.argmax(axis=1)
        correct     = (point_preds == eval_labels)

        for variant in variants:
            m         = all_results[framing][variant]
            set_sizes = m["set_sizes"]
            K         = max(int(k) for k in m["set_size_dist"])

            xs, ys, ns = [], [], []
            for s in range(K + 1):
                mask = set_sizes == s
                if mask.sum() == 0:
                    continue
                acc = correct[mask].mean()
                xs.append(s)
                ys.append(float(acc))
                ns.append(int(mask.sum()))
                rows_out.append({
                    "Framing":  framing,
                    "Variant":  variant,
                    "Set_size": s,
                    "Accuracy": float(acc),
                    "N":        int(mask.sum()),
                })

            ax.plot(xs, ys, marker="o",
                    label=variant, color=colors[variant])
            for x, y, n in zip(xs, ys, ns):
                ax.annotate(f"n={n}", (x, y),
                            textcoords="offset points",
                            xytext=(5, 5), fontsize=7)

        ax.set_title(framing)
        ax.set_xlabel("Prediction Set Size")
        ax.set_ylabel("Point Prediction Accuracy")
        ax.set_ylim(-0.05, 1.1)
        ax.axhline(y=0.5, color="grey", linestyle=":", alpha=0.5, label="Chance")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    pd.DataFrame(rows_out).to_csv(
        os.path.join(out_dir, "accuracy_vs_set_size.csv"), index=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_vs_set_size.png"), dpi=150)
    plt.close()
    print("  Saved: accuracy_vs_set_size.png")


def analysis_set_size_vs_lca_bin(all_results, df_eval, out_dir):
    """Analysis 6: Set size — MADURAI vs. single-etiology studies."""
    is_madurai = (df_eval["study"] == "MADURAI photos").values
    framings   = list(all_results.keys())
    variants   = list(next(iter(all_results.values())).keys())
    rows_out   = []

    fig, axes = plt.subplots(len(framings), len(variants),
                             figsize=(6 * len(variants), 5 * len(framings)))
    fig.suptitle("Set Size: MADURAI vs. Single-Etiology (LAC)", fontsize=14)

    for r, framing in enumerate(framings):
        for c, variant in enumerate(variants):
            ax = axes[r][c] if len(framings) > 1 else axes[c]
            m  = all_results[framing][variant]
            set_sizes = m["set_sizes"]
            K  = max(int(k) for k in m["set_size_dist"])

            madurai_sizes = set_sizes[is_madurai]
            other_sizes   = set_sizes[~is_madurai]

            bp = ax.boxplot(
                [madurai_sizes, other_sizes],
                tick_labels=[f"MADURAI\n(n={len(madurai_sizes)})",
                             f"Single-etiology\n(n={len(other_sizes)})"],
                patch_artist=True
            )
            bp["boxes"][0].set_facecolor("#1f77b4")
            bp["boxes"][1].set_facecolor("#ff7f0e")
            ax.set_title(f"{framing} — {variant}", fontsize=10)
            ax.set_ylabel("Prediction Set Size")
            ax.set_ylim(-0.1, K + 0.5)

            rows_out.append({
                "Framing": framing, "Variant": variant,
                "MADURAI_mean":        float(madurai_sizes.mean()),
                "SingleEtiology_mean": float(other_sizes.mean()),
            })

    pd.DataFrame(rows_out).to_csv(
        os.path.join(out_dir, "set_size_vs_lca_bin.csv"), index=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "set_size_vs_lca_bin.png"), dpi=150)
    plt.close()
    print("  Saved: set_size_vs_lca_bin.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(args.weights), "conformal")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output directory: {args.out_dir}\n")

    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- Data ---
    df_cal, df_eval = load_and_split(args.csv, args.img_root, seed=args.seed)

    cal_loader  = DataLoader(KeratitisDataset(df_cal,  val_transforms),
                             batch_size=args.batch, shuffle=False,
                             num_workers=4, pin_memory=True)
    eval_loader = DataLoader(KeratitisDataset(df_eval, val_transforms),
                             batch_size=args.batch, shuffle=False,
                             num_workers=4, pin_memory=True)

    # --- Model ---
    print("Loading model...")
    if args.head == "beta_dirichlet":
        model = BetaDirichletModel()
        raw   = torch.load(args.weights, map_location=device)
        fixed = {(k[len("head."):] if k.startswith("head.") else k): v
                 for k, v in raw.items()}
        model.load_state_dict(fixed)
        print("  Loaded Beta-Dirichlet model")
    else:
        model = EfficientNetB7Sigmoid()
        model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)

    # --- Inference ---
    print("Running inference on calibration set...")
    cal_probs,  cal_labels_soft  = get_predictions(model, cal_loader,  device,
                                                    head=args.head)
    print("Running inference on evaluation set...")
    eval_probs, eval_labels_soft = get_predictions(model, eval_loader, device,
                                                    head=args.head)

    # --- Groups (studies) ---
    cal_groups  = df_cal["study"].values
    eval_groups = df_eval["study"].values
    group_names = STUDY_SHORT  # short display names

    # --- 2-class label arrays ---
    cal_bac_bin   = binarize(cal_labels_soft[:,  0])
    cal_fung_bin  = binarize(cal_labels_soft[:,  1])
    eval_bac_bin  = binarize(eval_labels_soft[:, 0])
    eval_fung_bin = binarize(eval_labels_soft[:, 1])

    # --- 4-class label arrays ---
    cal_labels_4  = to_4class_labels(cal_bac_bin,  cal_fung_bin)
    eval_labels_4 = to_4class_labels(eval_bac_bin, eval_fung_bin)

    # --- 2-class probability arrays ---
    cal_probs_bac   = make_binary_probs(cal_probs[:,  0])
    cal_probs_fung  = make_binary_probs(cal_probs[:,  1])
    eval_probs_bac  = make_binary_probs(eval_probs[:, 0])
    eval_probs_fung = make_binary_probs(eval_probs[:, 1])

    # --- 4-class probability arrays ---
    cal_probs_4  = to_4class_probs(cal_probs[:,  0], cal_probs[:,  1])
    eval_probs_4 = to_4class_probs(eval_probs[:, 0], eval_probs[:, 1])

    # --- Run all variants x framings ---
    print(f"\nRunning conformal prediction (alpha={args.alpha})...\n")

    all_results = {
        "2-class BAC": run_all_variants(
            cal_probs_bac,  cal_bac_bin,  cal_groups,
            eval_probs_bac, eval_bac_bin, eval_groups,
            args.alpha, group_names
        ),
        "2-class FUNG": run_all_variants(
            cal_probs_fung,  cal_fung_bin,  cal_groups,
            eval_probs_fung, eval_fung_bin, eval_groups,
            args.alpha, group_names
        ),
        "4-class": run_all_variants(
            cal_probs_4,  cal_labels_4,  cal_groups,
            eval_probs_4, eval_labels_4, eval_groups,
            args.alpha, group_names
        ),
    }

    # Lookup dicts for accuracy analysis — maps framing -> eval probs/labels
    eval_probs_dict = {
        "2-class BAC":  eval_probs_bac,
        "2-class FUNG": eval_probs_fung,
        "4-class":      eval_probs_4,
    }
    eval_labels_dict = {
        "2-class BAC":  eval_bac_bin,
        "2-class FUNG": eval_fung_bin,
        "4-class":      eval_labels_4,
    }

    # --- Analyses ---
    print("\nRunning analyses...")
    analysis_coverage(all_results, args.alpha, args.out_dir)
    analysis_set_size_distribution(all_results, args.out_dir)
    analysis_coverage_vs_set_size(all_results, args.out_dir)
    analysis_accuracy_vs_set_size(all_results, eval_probs_dict,
                                  eval_labels_dict, args.out_dir)
    analysis_set_size_by_study(all_results, df_eval, args.out_dir)
    analysis_set_size_vs_posterior(all_results, df_eval, args.out_dir)
    analysis_set_size_vs_lca_bin(all_results, df_eval, args.out_dir)

    # --- Save summary JSON (strip arrays) ---
    summary = {}
    for framing, variants in all_results.items():
        summary[framing] = {}
        for variant, m in variants.items():
            entry = {k: v for k, v in m.items()
                     if not isinstance(v, np.ndarray)}
            # qhat may be a dict (group/class variants) — already serializable
            summary[framing][variant] = entry

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs saved to: {args.out_dir}/")
    print("  results.json | coverage.csv")
    print("  set_size_by_study.csv | set_size_vs_lca_bin.csv")
    print("  set_size_distribution.png | coverage_vs_set_size.png")
    print("  set_size_by_study.png | set_size_vs_posterior.png")
    print("  set_size_vs_lca_bin.png | config.json")


if __name__ == "__main__":
    main()