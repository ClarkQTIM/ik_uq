"""
Disagreement Cases: Conformal vs. Beta-Dirichlet
=================================================
Finds and plots images where the two uncertainty frameworks disagree most:

    Type A: HIGH conformal uncertainty (set size 4) + HIGH BD evidence
            → Frequentist says uncertain, but model is confident
            → Model learned something the calibration set didn't capture?

    Type B: LOW conformal uncertainty (set size 1) + LOW BD evidence
            → Frequentist says certain, but model has little evidence
            → Model got lucky? Or conformal is miscalibrated here?

    Type C: MATCHED LCA posterior (±0.05), HIGH vs LOW BD evidence pairs
            → Same diagnostic uncertainty from tests, different image clarity
            → Demonstrates: BD evidence captures image-level epistemic
              uncertainty independent of label-level aleatoric uncertainty
            → The visual demonstration of: "two images can have identical
              LCA posteriors but different BD evidence"

Run:
    python disagreement_cases.py
        --csv        /path/to/data.csv
        --img_root   /path/to/images
        --bd_weights runs/beta_dirichlet_.../best_model.pt
        --sig_weights runs/sigmoid_.../best_model.pt
        --n_cases    6
        --out_dir    runs/beta_dirichlet_.../uncertainty_analysis
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
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import GroupShuffleSplit
import timm
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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
    parser.add_argument("--n_cases",     type=int, default=6,
                        help="Number of cases to show per disagreement type")
    parser.add_argument("--n_pairs",     type=int, default=6,
                        help="Number of Type C matched-posterior pairs to plot")
    parser.add_argument("--posterior_tol", type=float, default=0.05,
                        help="Tolerance for LCA posterior matching in Type C (default +-0.05)")
    parser.add_argument("--alpha",       type=float, default=0.1)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--out_dir",     type=str, default=None)
    return parser.parse_args()


# =============================================================================
# DATA
# =============================================================================

PACS_PREFIX = "/pacs/OSHU/Corneal_Ulcer_Image_Sets"

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

    # Further split test into cal/eval to match conformal.py
    gss3 = GroupShuffleSplit(n_splits=1, test_size=0.80, random_state=seed)
    cal_idx, eval_idx = next(gss3.split(df_test, groups=df_test["subject"].values))

    print(f"Test set: {len(df_test):,}  |  Cal: {len(cal_idx)}  |  Eval: {len(eval_idx)}\n")
    return df_test, cal_idx, eval_idx


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
            image = Image.new("RGB", (600, 600), color=(30, 30, 30))
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

raw_transforms = transforms.Compose([
    transforms.Resize((300, 300)),   # smaller for display
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
        features = self.trunk(self.backbone(x))
        return torch.cat([self.bac_head(features),
                          self.fung_head(features)], dim=-1)


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


# =============================================================================
# INFERENCE
# =============================================================================

@torch.no_grad()
def run_inference(bd_model, sig_model, loader, device):
    bd_params, sig_probs, all_labels = [], [], []
    for images, labels in tqdm(loader, desc="  Inference", leave=False):
        images = images.to(device)
        bd_params.append(bd_model(images).cpu())
        sig_probs.append(sig_model(images).cpu())
        all_labels.append(labels)

    bd_params  = torch.cat(bd_params).numpy()
    sig_probs  = torch.cat(sig_probs).numpy()
    labels     = torch.cat(all_labels).numpy()

    alpha_bac  = bd_params[:, 0];  beta_bac   = bd_params[:, 1]
    alpha_fung = bd_params[:, 2];  beta_fung  = bd_params[:, 3]
    e_bac  = alpha_bac  + beta_bac
    e_fung = alpha_fung + beta_fung
    p_bac  = alpha_bac  / e_bac
    p_fung = alpha_fung / e_fung

    return {
        "e_bac": e_bac, "e_fung": e_fung,
        "p_bac": p_bac, "p_fung": p_fung,
        "alpha_bac":  alpha_bac,  "alpha_fung": alpha_fung,
        "beta_bac":   beta_bac,   "beta_fung":  beta_fung,
        "sig_bac": sig_probs[:, 0], "sig_fung": sig_probs[:, 1],
        "labels": labels,
    }


# =============================================================================
# CONFORMAL
# =============================================================================

def build_conformal_sets(sig_bac, sig_fung, labels, cal_idx, eval_idx, alpha):
    """Vanilla LAC 4-class on eval subset."""
    def to_4class(b, f):
        return np.stack([(1-b)*(1-f), b*(1-f), (1-b)*f, b*f], axis=1)

    probs_4  = to_4class(sig_bac, sig_fung)
    bac_bin  = (labels[:, 0] > 0.5).astype(int)
    fung_bin = (labels[:, 1] > 0.5).astype(int)
    labels_4 = bac_bin * 1 + fung_bin * 2

    cal_p = probs_4[cal_idx];  cal_l = labels_4[cal_idx]
    eval_p = probs_4[eval_idx]

    n = len(cal_l)
    scores  = 1 - cal_p[np.arange(n), cal_l]
    q_level = min(np.ceil((n+1)*(1-alpha))/n, 1.0)
    qhat    = float(np.quantile(scores, q_level, method="higher"))
    psets   = eval_p >= (1 - qhat)
    return psets.sum(axis=1), labels_4[eval_idx], qhat


# =============================================================================
# PLOT DISAGREEMENT CASES
# =============================================================================

CLASS4_NAMES = ["Neither", "Bac only", "Fung only", "Both"]
DENORM_MEAN  = np.array([0.485, 0.456, 0.406])
DENORM_STD   = np.array([0.229, 0.224, 0.225])


def denormalize(tensor):
    """Convert normalized tensor back to displayable uint8 image."""
    img = tensor.numpy().transpose(1, 2, 0)
    img = img * DENORM_STD + DENORM_MEAN
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def plot_cases(cases, df_eval, results_eval, set_sizes, title, filename, out_dir):
    """
    Plot a grid of images with their uncertainty stats.
    cases: list of eval-set indices to plot
    """
    n = len(cases)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 9))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    for col, idx in enumerate(cases):
        row_meta = df_eval.iloc[idx]

        # Load raw image for display (not normalized)
        try:
            raw_img = Image.open(row_meta["local_path"]).convert("RGB")
            raw_img = raw_img.resize((300, 300))
        except Exception:
            raw_img = Image.new("RGB", (300, 300), (50, 50, 50))

        # Top row: image
        ax_img = axes[0][col]
        ax_img.imshow(raw_img)
        ax_img.axis("off")

        study = row_meta["study"].replace(" photos","").replace("-Pictures","")
        true_bac  = row_meta["bac_post_prob"]
        true_fung = row_meta["fung_post_prob"]
        ax_img.set_title(
            f"{study}\nBAC label={true_bac:.2f}  FUNG label={true_fung:.2f}",
            fontsize=8
        )

        # Bottom row: uncertainty panel
        ax_stats = axes[1][col]
        ax_stats.axis("off")

        e_bac  = results_eval["e_bac"][idx]
        e_fung = results_eval["e_fung"][idx]
        p_bac  = results_eval["p_bac"][idx]
        p_fung = results_eval["p_fung"][idx]
        sb     = results_eval["sig_bac"][idx]
        sf     = results_eval["sig_fung"][idx]
        ss     = set_sizes[idx]

        label_4 = CLASS4_NAMES[
            int(true_bac > 0.5) * 1 + int(true_fung > 0.5) * 2
        ]

        lines = [
            f"True label: {label_4}",
            "",
            "── Beta-Dirichlet ──",
            f"p_bac  = {p_bac:.3f}",
            f"p_fung = {p_fung:.3f}",
            f"e_bac  = {e_bac:.2f}",
            f"e_fung = {e_fung:.2f}",
            f"e_mean = {(e_bac+e_fung)/2:.2f}",
            "",
            "── Sigmoid (conformal) ──",
            f"p_bac  = {sb:.3f}",
            f"p_fung = {sf:.3f}",
            f"conf set size = {ss}",
        ]

        # Colour background by agreement type
        bg_color = "#fff3cd" if ss == 4 else "#d4edda"
        ax_stats.set_facecolor(bg_color)
        ax_stats.patch.set_visible(True)

        for i, line in enumerate(lines):
            bold = line.startswith("──") or line.startswith("True")
            ax_stats.text(
                0.05, 0.97 - i * 0.075, line,
                transform=ax_stats.transAxes,
                fontsize=8.5,
                fontweight="bold" if bold else "normal",
                verticalalignment="top",
                fontfamily="monospace",
            )

    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


# =============================================================================
# HELPERS: FIND MATCHED PAIRS AND PLOT
# =============================================================================

def find_matched_pairs(df_eval, results_eval, set_sizes_eval,
                       n_pairs=6, posterior_tol=0.05, pathogen="bac",
                       match_on="evidence"):
    """
    Find pairs of eval images where:
      - LCA posterior for the given pathogen is similar (within posterior_tol)
      - The comparison metric differs substantially:
          match_on="evidence": directional alpha for that pathogen (high vs low)
          match_on="cpsl":     conformal set size (CPSL=1 vs CPSL=4)

    For Type C: same LCA posterior, wildly different BD alpha for that pathogen.
    For Type D: same LCA posterior, CPSL=1 vs CPSL=4.
    """
    col   = "bac_post_prob"  if pathogen == "bac" else "fung_post_prob"
    a_key = "alpha_bac"      if pathogen == "bac" else "alpha_fung"
    b_key = "beta_bac"       if pathogen == "bac" else "beta_fung"
    e_key = "e_bac"          if pathogen == "bac" else "e_fung"

    posterior = df_eval[col].values
    valid = posterior >= 0
    if valid.sum() < 10:
        print(f"  Not enough valid {pathogen} posteriors")
        return []

    if match_on == "evidence":
        # Use directional alpha parameter specifically for this pathogen.
        # alpha_bac = how much visual evidence FOR bacterial the model accumulated.
        # Two images with the same LCA posterior but different alpha_bac:
        # one where the image clearly looks bacterial, one where it gives nothing.
        a_key = "alpha_bac" if pathogen == "bac" else "alpha_fung"
        values = results_eval[a_key]
        t33 = np.percentile(values[valid], 33)
        t67 = np.percentile(values[valid], 67)
        high_mask = valid & (values >= t67)   # high alpha = strong positive visual evidence
        low_mask  = valid & (values <= t33)   # low alpha  = little visual evidence
    else:  # cpsl
        values = set_sizes_eval.astype(float)
        high_mask = valid & (values >= 4)   # CPSL=4 most uncertain
        low_mask  = valid & (values == 1)   # CPSL=1 most certain

    pairs = []
    high_idx = np.where(high_mask)[0]
    low_idx  = np.where(low_mask)[0]

    for hi in high_idx:
        for lo in low_idx:
            if abs(posterior[hi] - posterior[lo]) <= posterior_tol:
                gap = abs(values[hi] - values[lo])
                pairs.append((hi, lo, gap,
                               posterior[hi], posterior[lo],
                               values[hi], values[lo]))

    pairs.sort(key=lambda x: -x[2])
    seen_hi, seen_lo = set(), set()
    unique = []
    for pair in pairs:
        hi, lo = pair[0], pair[1]
        if hi not in seen_hi and lo not in seen_lo:
            unique.append(pair)
            seen_hi.add(hi); seen_lo.add(lo)
        if len(unique) >= n_pairs:
            break

    print(f"  Found {len(unique)} pairs from {len(pairs)} candidates")
    return unique


def overlay_text(img, lines, position="top", bg_color=(0,0,0,180), text_color=(255,255,255)):
    """Draw text overlay directly on PIL image."""
    from PIL import ImageDraw, ImageFont
    img = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    font_size = max(16, img.width // 30)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size - 2)
    except:
        font = ImageFont.load_default()
        font_sm = font

    # Measure total text block height
    line_h = font_size + 6
    total_h = line_h * len(lines) + 12
    w = img.width

    y0 = 8 if position == "top" else img.height - total_h - 8
    draw.rectangle([0, y0 - 4, w, y0 + total_h], fill=bg_color)

    for i, (text, bold) in enumerate(lines):
        f = font if bold else font_sm
        draw.text((10, y0 + i * line_h + 4), text, font=f, fill=text_color)

    img = Image.alpha_composite(img, overlay)
    return img.convert("RGB")


def plot_pair(idx_a, idx_b, val_a, val_b, post_a, post_b,
              df_eval, results_eval, set_sizes_eval,
              label_a, label_b, col_a, col_b,
              pathogen, metric_name, pathogen_label, out_dir, fname):
    """
    Save a single clean pair-plot: two images side by side.
    Key numbers are overlaid DIRECTLY on each image so they are clearly readable.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor("#0D1B2A")

    e_key     = "e_bac"      if pathogen == "bac" else "e_fung"
    p_key     = "p_bac"      if pathogen == "bac" else "p_fung"
    a_key     = "alpha_bac"  if pathogen == "bac" else "alpha_fung"
    b_key     = "beta_bac"   if pathogen == "bac" else "beta_fung"
    pathogen_str = "BAC" if pathogen == "bac" else "FUNG"
    post_col  = "bac_post_prob" if pathogen == "bac" else "fung_post_prob"

    for ax, idx, label, col in [
        (axes[0], idx_a, label_a, col_a),
        (axes[1], idx_b, label_b, col_b),
    ]:
        row = df_eval.iloc[idx]
        post_val  = row[post_col] if post_col in row.index else 0.0
        alpha_val = results_eval[a_key][idx]
        beta_val  = results_eval[b_key][idx]
        e_val     = results_eval[e_key][idx]
        p_val     = results_eval[p_key][idx]
        cpsl_val  = int(set_sizes_eval[idx])
        study     = str(row.get("study","?")).replace(" photos","").replace("-Pictures","")

        try:
            img = Image.open(row["local_path"]).convert("RGB")
            img = img.resize((500, 500))
        except:
            img = Image.new("RGB", (500, 500), color=(40,40,40))

        lines_top = [
            (label, True),
            (f"LCA posterior ({pathogen_str}): {post_val:.3f}", False),
        ]
        lines_bot = [
            (f"alpha_{pathogen_str} = {alpha_val:.2f}   beta_{pathogen_str} = {beta_val:.2f}", True),
            (f"Total evidence (a+b) = {e_val:.2f}   BD prob = {p_val:.3f}", False),
            (f"Conformal CPSL = {cpsl_val}", True),
            (f"Study: {study}", False),
        ]
        img = overlay_text(img, lines_top, position="top",
                           bg_color=(0,0,0,200), text_color=(255,255,255))
        img = overlay_text(img, lines_bot, position="bottom",
                           bg_color=(0,0,0,200), text_color=(255,255,255))

        ax.imshow(img)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(col)
            spine.set_linewidth(6)
            spine.set_visible(True)

    post_mid = (post_a + post_b) / 2
    plt.suptitle(
        f"{pathogen_label} Keratitis  |  Matched LCA posterior ~{post_mid:.3f} (tol +/-{0.05:.2f})  |  Varying: {metric_name}",
        fontsize=12, color="white", y=1.01
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


# =============================================================================
# TYPE C AND D RUNNERS
# =============================================================================

def run_type_c(results_eval, set_sizes_eval, df_eval, args, base_out):
    """Type C: matched posterior, high vs low BD evidence. One file per pair."""
    out_dir = os.path.join(base_out, "type_c_matched_posterior")
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "="*55)
    print("  TYPE C: MATCHED POSTERIOR, DIFFERENT BD EVIDENCE")
    print("="*55)

    for pathogen in ["bac", "fung"]:
        pathogen_label = "Bacterial" if pathogen == "bac" else "Fungal"
        e_key = "e_bac" if pathogen == "bac" else "e_fung"
        print(f"\n  {pathogen_label}...")

        pairs = find_matched_pairs(df_eval, results_eval, set_sizes_eval,
                                    n_pairs=args.n_pairs,
                                    posterior_tol=args.posterior_tol,
                                    pathogen=pathogen, match_on="evidence")

        for i, (hi, lo, gap, post_hi, post_lo, e_hi, e_lo) in enumerate(pairs):
            fname = f"pair_{pathogen}_{i+1:02d}.png"
            plot_pair(hi, lo, e_hi, e_lo, post_hi, post_lo,
                      df_eval, results_eval, set_sizes_eval,
                      label_a="HIGH alpha (strong visual evidence)",
                      label_b="LOW alpha (weak visual evidence)",
                      col_a="#2D9E6B", col_b="#E05252",
                      pathogen=pathogen,
                      metric_name="alpha (directional BD evidence)",
                      pathogen_label=pathogen_label,
                      out_dir=out_dir, fname=fname)
            print(f"    {fname}  e={e_hi:.1f} vs {e_lo:.1f}  post={post_hi:.3f} vs {post_lo:.3f}")

    print(f"\n  Type C saved to: {out_dir}/")


def run_type_d(results_eval, set_sizes_eval, df_eval, args, base_out):
    """Type D: matched posterior, CPSL=1 vs CPSL=4. One file per pair."""
    out_dir = os.path.join(base_out, "type_d_matched_cpsl")
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "="*55)
    print("  TYPE D: MATCHED POSTERIOR, DIFFERENT CPSL")
    print("="*55)

    for pathogen in ["bac", "fung"]:
        pathogen_label = "Bacterial" if pathogen == "bac" else "Fungal"
        print(f"\n  {pathogen_label}...")

        pairs = find_matched_pairs(df_eval, results_eval, set_sizes_eval,
                                    n_pairs=args.n_pairs,
                                    posterior_tol=args.posterior_tol,
                                    pathogen=pathogen, match_on="cpsl")

        for i, (hi, lo, gap, post_hi, post_lo, cpsl_hi, cpsl_lo) in enumerate(pairs):
            fname = f"pair_{pathogen}_{i+1:02d}.png"
            # lo = CPSL=1 (certain), hi = CPSL=4 (uncertain)
            plot_pair(lo, hi, cpsl_lo, cpsl_hi, post_lo, post_hi,
                      df_eval, results_eval, set_sizes_eval,
                      label_a="CPSL=1 (confident)",
                      label_b="CPSL=4 (uncertain)",
                      col_a="#2D9E6B", col_b="#E05252",
                      pathogen=pathogen,
                      metric_name="Conformal CPSL",
                      pathogen_label=pathogen_label,
                      out_dir=out_dir, fname=fname)
            print(f"    {fname}  CPSL=1 vs 4  post={post_lo:.3f} vs {post_hi:.3f}")

    print(f"\n  Type D saved to: {out_dir}/")



# =============================================================================
# MAIN
# =============================================================================

def main():
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.out_dir is None:
        args.out_dir = os.path.join(
            os.path.dirname(args.bd_weights), "uncertainty_analysis"
        )
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output directory: {args.out_dir}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- Data ---
    df_test, cal_idx, eval_idx = load_test_set(
        args.csv, args.img_root, seed=args.seed
    )
    loader = DataLoader(
        KeratitisDataset(df_test, val_transforms),
        batch_size=args.batch if hasattr(args, "batch") else 8,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # --- Models ---
    print("Loading models...")
    bd_model  = load_bd(args.bd_weights,  device)
    sig_model = load_sig(args.sig_weights, device)

    # --- Inference on full test set ---
    results = run_inference(bd_model, sig_model, loader, device)

    # --- Conformal sets on eval subset ---
    set_sizes_eval, labels_4_eval, qhat = build_conformal_sets(
        results["sig_bac"], results["sig_fung"],
        results["labels"], cal_idx, eval_idx, args.alpha
    )
    print(f"  Conformal qhat: {qhat:.4f}")
    print(f"  Set size distribution: "
          f"{ {s: int((set_sizes_eval==s).sum()) for s in range(5)} }\n")

    # Results indexed to eval subset
    results_eval = {k: v[eval_idx] for k, v in results.items()}
    df_eval = df_test.iloc[eval_idx].reset_index(drop=True)

    e_mean_eval = (results_eval["e_bac"] + results_eval["e_fung"]) / 2

    # -------------------------------------------------------------------------
    # TYPE A: HIGH conformal uncertainty (set size 4) + HIGH BD evidence
    # Model is confident but conformal says uncertain
    # Pick cases with set_size==4 and highest e_mean
    # -------------------------------------------------------------------------
    mask_a = set_sizes_eval == 4
    if mask_a.sum() > 0:
        candidates_a = np.where(mask_a)[0]
        # Sort by descending evidence (most confident BD, most uncertain conformal)
        sorted_a = candidates_a[np.argsort(e_mean_eval[candidates_a])[::-1]]
        cases_a  = sorted_a[:args.n_cases].tolist()

        print(f"TYPE A: Conformal uncertain (size=4), BD confident")
        for idx in cases_a:
            print(f"  idx={idx}  e_mean={e_mean_eval[idx]:.2f}  "
                  f"e_bac={results_eval['e_bac'][idx]:.2f}  "
                  f"e_fung={results_eval['e_fung'][idx]:.2f}  "
                  f"p_bac={results_eval['p_bac'][idx]:.3f}  "
                  f"p_fung={results_eval['p_fung'][idx]:.3f}  "
                  f"study={df_eval.iloc[idx]['study']}")

        plot_cases(
            cases_a, df_eval, results_eval, set_sizes_eval,
            title=(
                "TYPE A: Conformal UNCERTAIN (set size=4)  +  "
                "Beta-Dirichlet CONFIDENT (high evidence)\n"
                "Frequentist says uncertain — model disagrees"
            ),
            filename="disagreement_type_A.png",
            out_dir=args.out_dir,
        )
    else:
        print("No Type A cases found (no set size=4 cases in eval set)")

    # -------------------------------------------------------------------------
    # TYPE B: LOW conformal uncertainty (set size 1) + LOW BD evidence
    # Conformal says certain but model has little evidence
    # Pick cases with set_size==1 and lowest e_mean
    # -------------------------------------------------------------------------
    mask_b = set_sizes_eval == 1
    if mask_b.sum() > 0:
        candidates_b = np.where(mask_b)[0]
        # Sort by ascending evidence (least confident BD, most certain conformal)
        sorted_b = candidates_b[np.argsort(e_mean_eval[candidates_b])]
        cases_b  = sorted_b[:args.n_cases].tolist()

        print(f"\nTYPE B: Conformal certain (size=1), BD uncertain")
        for idx in cases_b:
            print(f"  idx={idx}  e_mean={e_mean_eval[idx]:.2f}  "
                  f"e_bac={results_eval['e_bac'][idx]:.2f}  "
                  f"e_fung={results_eval['e_fung'][idx]:.2f}  "
                  f"p_bac={results_eval['p_bac'][idx]:.3f}  "
                  f"p_fung={results_eval['p_fung'][idx]:.3f}  "
                  f"study={df_eval.iloc[idx]['study']}")

        plot_cases(
            cases_b, df_eval, results_eval, set_sizes_eval,
            title=(
                "TYPE B: Conformal CERTAIN (set size=1)  +  "
                "Beta-Dirichlet UNCERTAIN (low evidence)\n"
                "Model says uncertain — conformal disagrees"
            ),
            filename="disagreement_type_B.png",
            out_dir=args.out_dir,
        )
    else:
        print("No Type B cases found (no set size=1 cases in eval set)")

    # ── Type C and D ──────────────────────────────────────────────────────────
    run_type_c(results_eval, set_sizes_eval, df_eval, args, args.out_dir)
    run_type_d(results_eval, set_sizes_eval, df_eval, args, args.out_dir)

    print(f"\nDone. Outputs in: {args.out_dir}/")


if __name__ == "__main__":
    main()