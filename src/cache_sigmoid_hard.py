"""
Generate cache_sig.npz for a sigmoid model (Cell C: sigmoid + hard labels).
Uses the exact architecture and weight-loading logic from linear_probe.py.

Saves cache_sig.npz in the same format as dirichlet_analysis.py:
    probs  : (785, 2)  — [p_bac, p_fung] after sigmoid
    labels : (785, 2)  — soft LCA posteriors from CSV

Run:
    python src/cache_sigmoid_hard.py \
        --weights runs/sigmoid_hard_2026-08-19_14-24-09/best_model.pt \
        --out_dir runs/sigmoid_hard_2026-08-19_14-24-09/uncertainty_analysis/ \
        --csv /persist/QTIM/Active/.../corneal_ulcer_study_data_7_24_2025_lambda.csv
"""

import os, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit


IMG_ROOT = "/pacs/OSHU/Corneal_Ulcer_Image_Sets"
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",  type=str, required=True)
    p.add_argument("--out_dir",  type=str, required=True)
    p.add_argument("--csv",      type=str,
        default="/persist/QTIM/Active/26-0096/Corneal_Ulcer_Image_Sets/Data/"
                "corneal_ulcer_study_data_7_24_2025_lambda.csv")
    p.add_argument("--img_root", type=str, default=IMG_ROOT)
    p.add_argument("--batch",    type=int, default=16)
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


# ── Model — exact copy from linear_probe.py ───────────────────────────────────
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


def load_sig(path, device):
    """Exact weight-loading logic from linear_probe.py."""
    model = SigmoidModel()
    raw = torch.load(path, map_location=device)

    # Debug: show first 5 keys so we can verify checkpoint structure
    sample_keys = list(raw.keys())[:5]
    print(f"  Checkpoint sample keys: {sample_keys}")

    fixed = {}
    for k, v in raw.items():
        if k.startswith("head.net."):
            fixed["head." + k[len("head.net."):]] = v
        elif k.startswith("backbone."):
            fixed[k] = v
        # else: skip any other top-level keys

    print(f"  Remapped {len(raw)} → {len(fixed)} keys")
    head_keys = [k for k in fixed if k.startswith("head.")]
    print(f"  Head keys sample: {head_keys[:5]}")

    model.load_state_dict(fixed, strict=True)
    return model.to(device).eval()


# ── Dataset ───────────────────────────────────────────────────────────────────
class KerDataset(Dataset):
    def __init__(self, df, img_root):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.tfm = transforms.Compose([
            transforms.Resize((600, 600)),
            transforms.CenterCrop(600),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = str(row["file_path"]).replace(
            "/pacs/OSHU/Corneal_Ulcer_Image_Sets", self.img_root)
        img = Image.open(path).convert("RGB")
        bac  = float(row.get("bac_post_prob",  row.get("bac_LCA_bin",  0)))
        fung = float(row.get("fung_post_prob", row.get("fung_LCA_bin", 0)))
        return self.tfm(img), torch.tensor([bac, fung], dtype=torch.float32)


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


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_sig(args.weights, device)
    print(f"Loaded: {args.weights}")

    df_test = load_test_set(args.csv, args.seed)
    print(f"Test set: {len(df_test):,} images")

    ds = KerDataset(df_test, args.img_root)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=4, pin_memory=(device.type == "cuda"))

    all_probs, all_labels = [], []
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(dl):
            imgs = imgs.to(device)
            probs = model(imgs).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
            if (i + 1) % 10 == 0:
                print(f"  batch {i+1}/{len(dl)}")

    probs  = np.concatenate(all_probs,  axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"Done: probs {probs.shape}, labels {labels.shape}")

    out_path = os.path.join(args.out_dir, "cache_sig.npz")
    np.savez(out_path, probs=probs, labels=labels)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()