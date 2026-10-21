"""
Latent Supervision — Infectious Keratitis Classifier
=====================================================
Faithful reproduction of Yazbeck et al. (2026) with two novel extensions.

Three head types:
    sigmoid        — faithful R script reproduction (rescaled MSE loss)
    dirichlet      — joint 2-class Dirichlet head (ICMLA paper, eqs 3-7)
                     outputs α = [α_bac, α_fung]; classes compete via simplex
    beta_dirichlet — TWO INDEPENDENT K=2 Dirichlet (Beta) heads, one per pathogen
                     outputs [α_bac, β_bac, α_fung, β_fung]
                     p̂_bac  = α_bac  / (α_bac  + β_bac)
                     p̂_fung = α_fung / (α_fung + β_fung)
                     Evidence per pathogen is INDEPENDENT — fixes the zero-sum
                     competition problem of the joint Dirichlet head, which is
                     mathematically wrong for a multilabel task.

Also adds:
    --patience     early stopping patience (default 5, 0 = disabled)
    --batch_phase2 separate batch size for Phase 2 (default = same as --batch)
    --lr_phase2    separate learning rate for Phase 2 (default = same as --lr)

Install:
    pip install torch torchvision timm pillow numpy pandas scikit-learn tqdm

Run:
    python model.py --head beta_dirichlet --epochs 30 --patience 5
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from datetime import datetime
import timm
from tqdm import tqdm


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def get_args():
    parser = argparse.ArgumentParser(description="Latent Supervision Keratitis Classifier")
    parser.add_argument("--csv",         type=str,
                        default="/persist/QTIM/Active/26-0096/Corneal_Ulcer_Image_Sets/Data/corneal_ulcer_study_data_7_24_2025_lambda.csv")
    parser.add_argument("--img_root",    type=str,
                        default="/pacs/OSHU/Corneal_Ulcer_Image_Sets")
    parser.add_argument("--head",        type=str, default="sigmoid",
                        choices=["sigmoid", "dirichlet", "beta_dirichlet"])
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch",       type=int,   default=8)
    parser.add_argument("--batch_phase2",type=int,   default=None,
                        help="Batch size for Phase 2. Defaults to --batch.")
    parser.add_argument("--lr",          type=float, default=1e-5)
    parser.add_argument("--lr_phase2",   type=float, default=None,
                        help="Learning rate for Phase 2. Defaults to --lr.")
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--patience",    type=int,   default=5,
                        help="Early stopping patience (epochs without val loss improvement). "
                             "0 = disabled.")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--label_type", type=str,   default="soft",
                        choices=["soft", "hard"],
                        help="soft = LCA posteriors (bac_post_prob, fung_post_prob); "
                             "hard = binarized LCA bins (bac_LCA_bin, fung_LCA_bin)")
    parser.add_argument("--out_dir",     type=str,   default=None)
    parser.add_argument("--device",      type=str,   default=None,
                        help="Device to use, e.g. 'cuda:0', 'cuda:1', 'cpu'. "
                             "Defaults to cuda if available, else cpu.")
    return parser.parse_args()


def make_output_dir(args):
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = os.path.join("runs", f"{args.head}_{args.label_type}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Output directory: {out_dir}\n")
    return out_dir


# =============================================================================
# DATA
# =============================================================================

PACS_PREFIX = "/pacs/OSHU/Corneal_Ulcer_Image_Sets"

def remap_path(file_path, img_root):
    return file_path.replace(PACS_PREFIX, img_root)


def load_dataframe(csv_path, img_root):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["file_path", "subject"])
    unlabelable = (df["fung_LCA_bin"] == -1) & (df["bac_LCA_bin"] == -1)
    df = df[~unlabelable].reset_index(drop=True)
    df["local_path"] = df["file_path"].apply(lambda p: remap_path(p, img_root))
    print(f"Loaded {len(df):,} rows after filtering")
    print(f"  Unique subjects: {df['subject'].nunique():,}")
    print(f"  Studies: {df['study'].value_counts().to_dict()}\n")
    return df


def split_dataframe(df, seed=42):
    subjects = df["subject"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(gss.split(df, groups=subjects))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_temp  = df.iloc[temp_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(gss2.split(df_temp, groups=df_temp["subject"].values))
    df_val  = df_temp.iloc[val_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)

    print(f"Split sizes:")
    print(f"  Train: {len(df_train):,} ({df_train['subject'].nunique():,} subjects)")
    print(f"  Val:   {len(df_val):,} ({df_val['subject'].nunique():,} subjects)")
    print(f"  Test:  {len(df_test):,} ({df_test['subject'].nunique():,} subjects)\n")
    return df_train, df_val, df_test


class KeratitisDataset(Dataset):
    def __init__(self, df, transform=None, label_type="soft"):
        self.df = df
        self.transform = transform
        self.label_type = label_type

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            image = Image.open(row["local_path"]).convert("RGB")
        except Exception as e:
            print(f"Warning: could not load {row['local_path']}: {e}")
            image = Image.new("RGB", (600, 600), color=0)
        if self.transform:
            image = self.transform(image)
        if self.label_type == "soft":
            label = torch.tensor(
                [row["bac_post_prob"], row["fung_post_prob"]],
                dtype=torch.float32
            )
        else:  # hard — binarized LCA bins (0 or 1, no -1 after standard filtering)
            label = torch.tensor(
                [float(row["bac_LCA_bin"]), float(row["fung_LCA_bin"])],
                dtype=torch.float32
            )
        return image, label


train_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(contrast=0.1),
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# =============================================================================
# HEADS
# =============================================================================

class SigmoidHead(nn.Module):
    """R script reproduction — sigmoid output, rescaled MSE loss."""
    def __init__(self, in_features, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(512, 32),          nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)   # shape: (batch, 2)  values in (0, 1)


class DirichletHead(nn.Module):
    """
    Joint 2-class Dirichlet head (ICMLA paper).
    Outputs α = [α_bac, α_fung] > 0.
    p̂ = α / sum(α)  — classes compete on the simplex.
    Limitation: wrong geometry for multilabel (bac and fung are not mutually exclusive).
    """
    def __init__(self, in_features, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(512, 32),          nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(32, 2),
            nn.Softplus()
        )

    def forward(self, x):
        return self.net(x)   # shape: (batch, 2)  values > 0  (α_bac, α_fung)


class BetaDirichletHead(nn.Module):
    """
    Two independent K=2 Dirichlet (Beta) heads — one per pathogen.
    Outputs [α_bac, β_bac, α_fung, β_fung] > 0.

    Per pathogen:
        p̂_bac  = α_bac  / (α_bac  + β_bac)
        p̂_fung = α_fung / (α_fung + β_fung)

    Evidence per pathogen:
        E_bac  = α_bac  + β_bac    (independent of fungal evidence)
        E_fung = α_fung + β_fung   (independent of bacterial evidence)

    Key difference from DirichletHead:
        - α_bac and α_fung are NOT competing. High evidence for one does not
          suppress evidence for the other. This is mathematically correct for
          a multilabel task where polymicrobial infection is possible.
        - Each pathogen has its own uncertainty estimate (β encodes absence
          evidence; high β_bac = strong evidence patient is NOT bacterial).

    Shared trunk (512 → 32) to keep parameter count comparable to other heads,
    then two independent output layers (one per pathogen, each → 2 params).
    """
    def __init__(self, in_features, dropout=0.3):
        super().__init__()
        # Shared feature extraction
        self.trunk = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(512, 32),          nn.ReLU(), nn.Dropout(p=dropout),
        )
        # Independent per-pathogen output heads
        self.bac_head  = nn.Sequential(nn.Linear(32, 2), nn.Softplus())
        self.fung_head = nn.Sequential(nn.Linear(32, 2), nn.Softplus())

    def forward(self, x):
        features = self.trunk(x)
        bac_params  = self.bac_head(features)   # [α_bac,  β_bac]
        fung_params = self.fung_head(features)  # [α_fung, β_fung]
        return torch.cat([bac_params, fung_params], dim=-1)  # shape: (batch, 4)


# =============================================================================
# MODEL
# =============================================================================

class EfficientNetB7(nn.Module):
    def __init__(self, head_type="sigmoid", dropout=0.3,
                 freeze_backbone=True, pretrained=True):
        super().__init__()
        assert head_type in ("sigmoid", "dirichlet", "beta_dirichlet")
        self.head_type = head_type

        self.backbone = timm.create_model(
            "tf_efficientnet_b7.ra_in1k",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )
        backbone_features = self.backbone.num_features  # 2560 for B7

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if head_type == "sigmoid":
            self.head = SigmoidHead(backbone_features, dropout)
        elif head_type == "dirichlet":
            self.head = DirichletHead(backbone_features, dropout)
        else:
            self.head = BetaDirichletHead(backbone_features, dropout)

    def forward(self, x):
        return self.head(self.backbone(x))

    def predict(self, x):
        """
        Returns class probabilities and per-pathogen evidence for all head types.

        sigmoid:        p_hat = sigmoid outputs directly
        dirichlet:      p_hat = α / α₀,  total_evidence = α₀ (scalar, both classes)
        beta_dirichlet: p_hat = [α_bac/(α+β), α_fung/(α+β)]
                        evidence_bac  = α_bac + β_bac   (per-pathogen)
                        evidence_fung = α_fung + β_fung  (per-pathogen)
        """
        out = self.forward(x)

        if self.head_type == "sigmoid":
            return {
                "p_hat":          out,
                "total_evidence": None,
                "evidence_bac":   None,
                "evidence_fung":  None,
            }
        elif self.head_type == "dirichlet":
            alpha_0 = out.sum(dim=-1, keepdim=True)
            return {
                "p_hat":          out / alpha_0,
                "total_evidence": alpha_0.squeeze(-1),
                "evidence_bac":   None,
                "evidence_fung":  None,
            }
        else:  # beta_dirichlet
            alpha_bac  = out[:, 0]
            beta_bac   = out[:, 1]
            alpha_fung = out[:, 2]
            beta_fung  = out[:, 3]
            e_bac  = alpha_bac  + beta_bac
            e_fung = alpha_fung + beta_fung
            p_bac  = alpha_bac  / e_bac
            p_fung = alpha_fung / e_fung
            return {
                "p_hat":          torch.stack([p_bac, p_fung], dim=-1),
                "total_evidence": (e_bac + e_fung) / 2,  # mean for display
                "evidence_bac":   e_bac,
                "evidence_fung":  e_fung,
            }

    def unfreeze(self, from_block=54):
        """
        Progressive unfreezing — mirrors unfreeze_weak_model() in R script.
        ENB7_blocks[2] = 54 (the R script default).
        BatchNorm layers stay frozen throughout.
        """
        for i, (name, module) in enumerate(self.backbone.named_modules()):
            if i >= from_block:
                if not isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    for param in module.parameters():
                        param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Unfrozen from block {from_block}. Trainable params: {trainable:,}")


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def rescaled_mse_loss(y_true, y_pred):
    """R script loss — MSE * 10."""
    return ((y_true - y_pred) ** 2 * 10).mean()


def dirichlet_loss(y_true, alpha, lambda_kl):
    """
    ICMLA paper equation 5 — joint Dirichlet MSE + annealed KL.
    y_true: soft labels [bac_prob, fung_prob], shape (batch, 2)
    alpha:  concentration params [α_bac, α_fung], shape (batch, 2)
    """
    alpha_0 = alpha.sum(dim=-1, keepdim=True)
    p_hat   = alpha / alpha_0

    mse_term = (
        (y_true - p_hat) ** 2 +
        p_hat * (1 - p_hat) / (alpha_0 + 1)
    ).sum(dim=-1)

    alpha_tilde   = y_true + (1.0 - y_true) * alpha
    alpha_tilde_0 = alpha_tilde.sum(dim=-1)
    K = alpha.shape[-1]

    kl_term = (
        torch.lgamma(alpha_tilde_0)
        - torch.lgamma(torch.tensor(float(K), device=alpha.device))
        - torch.lgamma(alpha_tilde).sum(dim=-1)
        + ((alpha_tilde - 1.0) * (
            torch.digamma(alpha_tilde) -
            torch.digamma(alpha_tilde_0.unsqueeze(-1))
        )).sum(dim=-1)
    )
    return (mse_term + lambda_kl * kl_term).mean()


def beta_dirichlet_loss(y_true, params, lambda_kl):
    """
    Two independent K=2 Dirichlet (Beta) losses, one per pathogen.

    params:  [α_bac, β_bac, α_fung, β_fung], shape (batch, 4)
    y_true:  [bac_prob, fung_prob], shape (batch, 2)

    Each pathogen is treated as a binary Dirichlet (Beta distribution):
        y_bac  = [bac_prob,      1 - bac_prob]    — soft label as (present, absent)
        y_fung = [fung_prob,     1 - fung_prob]
        α_bac  = [params[:,0],   params[:,1]]
        α_fung = [params[:,2],   params[:,3]]

    Loss = dirichlet_loss(y_bac, α_bac) + dirichlet_loss(y_fung, α_fung)
    The two losses are completely independent — no zero-sum competition.
    """
    # Bacterial: binary soft label [p_present, p_absent]
    y_bac   = torch.stack([y_true[:, 0], 1 - y_true[:, 0]], dim=-1)
    alpha_bac = params[:, :2]   # [α_bac, β_bac]

    # Fungal: binary soft label [p_present, p_absent]
    y_fung   = torch.stack([y_true[:, 1], 1 - y_true[:, 1]], dim=-1)
    alpha_fung = params[:, 2:]  # [α_fung, β_fung]

    loss_bac  = dirichlet_loss(y_bac,  alpha_bac,  lambda_kl)
    loss_fung = dirichlet_loss(y_fung, alpha_fung, lambda_kl)

    return loss_bac + loss_fung


def kl_schedule(epoch, total_epochs, lambda_0=1e-4, num_classes=2):
    """ICMLA paper equation 7."""
    return (lambda_0 / num_classes) * (epoch / max(total_epochs, 1))


def compute_loss(head_type, y_true, output, epoch=0, total_epochs=1):
    if head_type == "sigmoid":
        return rescaled_mse_loss(y_true, output)
    elif head_type == "dirichlet":
        lam = kl_schedule(epoch, total_epochs, num_classes=2)
        return dirichlet_loss(y_true, output, lam)
    else:  # beta_dirichlet
        lam = kl_schedule(epoch, total_epochs, num_classes=2)
        return beta_dirichlet_loss(y_true, output, lam)


# =============================================================================
# METRICS
# =============================================================================

@torch.no_grad()
def compute_metrics(model, loader, device, threshold=0.5):
    model.eval()
    all_preds, all_labels = [], []

    bar = tqdm(loader, desc="  Evaluating", unit="batch", leave=False)
    for images, labels in bar:
        result = model.predict(images.to(device))
        all_preds.append(result["p_hat"].cpu())
        all_labels.append(labels)

    preds  = torch.cat(all_preds,  dim=0)
    labels = torch.cat(all_labels, dim=0)

    binary_preds  = (preds  > threshold).float()
    binary_labels = (labels > threshold).float()

    metrics = {}
    for i, name in enumerate(["bac", "fung"]):
        tp = ((binary_preds[:, i] == 1) & (binary_labels[:, i] == 1)).sum().item()
        fp = ((binary_preds[:, i] == 1) & (binary_labels[:, i] == 0)).sum().item()
        fn = ((binary_preds[:, i] == 0) & (binary_labels[:, i] == 1)).sum().item()
        tn = ((binary_preds[:, i] == 0) & (binary_labels[:, i] == 0)).sum().item()

        acc  = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        metrics[name] = {"accuracy": acc, "f1": f1,
                         "precision": prec, "recall": rec}

    return metrics, preds.numpy(), labels.numpy()


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_one_epoch(model, loader, optimizer, epoch, total_epochs, device):
    model.train()
    total_loss = 0.0
    bar = tqdm(loader, desc="  Train", unit="batch", leave=False)
    for images, labels in bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = compute_loss(model.head_type, labels, output, epoch, total_epochs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, epoch, total_epochs, device):
    model.eval()
    total_loss     = 0.0
    total_evidence = 0.0
    bar = tqdm(loader, desc="  Val  ", unit="batch", leave=False)
    for images, labels in bar:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = compute_loss(model.head_type, labels, output, epoch, total_epochs)
        total_loss += loss.item()
        if model.head_type in ("dirichlet", "beta_dirichlet"):
            result = model.predict(images)
            total_evidence += result["total_evidence"].mean().item()
        bar.set_postfix(loss=f"{loss.item():.4f}")

    out = {"loss": total_loss / len(loader)}
    if model.head_type in ("dirichlet", "beta_dirichlet"):
        out["mean_evidence"] = total_evidence / len(loader)
    return out


def train(model, train_loader, val_loader, args, device, out_dir,
          phase_label="", lr=None, loader_override=None):
    """
    Training loop with early stopping.

    Early stopping: if val loss doesn't improve for `args.patience` epochs,
    stop and restore best weights. Set args.patience=0 to disable.

    lr: override learning rate (used for Phase 2 if --lr_phase2 is set)
    loader_override: override train_loader (used for Phase 2 if --batch_phase2 is set)
    """
    if lr is None:
        lr = args.lr
    if loader_override is not None:
        train_loader = loader_override

    optimizer    = torch.optim.Adam(model.parameters(), lr=lr)
    weights_path = os.path.join(out_dir, "best_model.pt")
    log_path     = os.path.join(out_dir, "train_log.csv")

    best_val_loss    = float("inf")
    best_epoch       = 0
    epochs_no_improve = 0

    epoch_bar = tqdm(range(args.epochs), desc=f"[{phase_label}]", unit="epoch")
    for epoch in epoch_bar:
        train_loss = train_one_epoch(
            model, train_loader, optimizer, epoch, args.epochs, device
        )
        val = evaluate(model, val_loader, epoch, args.epochs, device)

        improved = val["loss"] < best_val_loss
        if improved:
            best_val_loss     = val["loss"]
            best_epoch        = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), weights_path)
        else:
            epochs_no_improve += 1

        # Log
        row = {
            "phase":      phase_label,
            "epoch":      epoch + 1,
            "train_loss": train_loss,
            "val_loss":   val["loss"],
            "improved":   improved,
        }
        if "mean_evidence" in val:
            row["mean_evidence"] = val["mean_evidence"]

        pd.DataFrame([row]).to_csv(
            log_path, mode="a",
            header=not os.path.exists(log_path) or epoch == 0,
            index=False
        )

        postfix = {
            "train": f"{train_loss:.4f}",
            "val":   f"{val['loss']:.4f}",
            "best":  f"ep{best_epoch}",
        }
        if "mean_evidence" in val:
            postfix["α₀"] = f"{val['mean_evidence']:.3f}"
        if not improved:
            postfix["no_improv"] = f"{epochs_no_improve}"
        epoch_bar.set_postfix(postfix)

        # Early stopping
        if args.patience > 0 and epochs_no_improve >= args.patience:
            tqdm.write(
                f"  [{phase_label}] Early stopping at epoch {epoch+1} "
                f"(no improvement for {args.patience} epochs)"
            )
            break

    tqdm.write(
        f"  [{phase_label}] Best val loss: {best_val_loss:.4f} "
        f"at epoch {best_epoch} — saved to {weights_path}"
    )
    # Restore best weights before returning
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model


# =============================================================================
# MAIN
# =============================================================================

def main():
    args    = get_args()
    out_dir = make_output_dir(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- Data ---
    df = load_dataframe(args.csv, args.img_root)
    df_train, df_val, df_test = split_dataframe(df, seed=args.seed)

    train_ds = KeratitisDataset(df_train, transform=train_transforms, label_type=args.label_type)
    val_ds   = KeratitisDataset(df_val,   transform=val_transforms,   label_type=args.label_type)
    test_ds  = KeratitisDataset(df_test,  transform=val_transforms,   label_type=args.label_type)

    batch_p2 = args.batch_phase2 if args.batch_phase2 else args.batch
    lr_p2    = args.lr_phase2    if args.lr_phase2    else args.lr

    train_loader    = DataLoader(train_ds, batch_size=args.batch,
                                 shuffle=True,  num_workers=4, pin_memory=True)
    train_loader_p2 = DataLoader(train_ds, batch_size=batch_p2,
                                 shuffle=True,  num_workers=4, pin_memory=True)
    val_loader      = DataLoader(val_ds,   batch_size=args.batch,
                                 shuffle=False, num_workers=4, pin_memory=True)
    test_loader     = DataLoader(test_ds,  batch_size=args.batch,
                                 shuffle=False, num_workers=4, pin_memory=True)

    # --- Model ---
    print(f"Building EfficientNetB7 with {args.head} head + {args.label_type} labels...")
    model = EfficientNetB7(
        head_type=args.head,
        dropout=args.dropout,
        freeze_backbone=True,
        pretrained=True
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}  (head only)")
    if args.patience > 0:
        print(f"Early stopping:   patience={args.patience}\n")
    else:
        print(f"Early stopping:   disabled\n")

    # --- Phase 1: Train head only ---
    print("=== Phase 1: Train head (backbone frozen) ===")
    model = train(model, train_loader, val_loader, args, device, out_dir,
                  phase_label="Phase 1", lr=args.lr)

    # --- Phase 2: Unfreeze and fine-tune ---
    print("\n=== Phase 2: Unfreeze backbone and fine-tune ===")
    model.unfreeze(from_block=54)
    if batch_p2 != args.batch:
        print(f"  Phase 2 batch size: {batch_p2} (was {args.batch})")
    if lr_p2 != args.lr:
        print(f"  Phase 2 learning rate: {lr_p2} (was {args.lr})")
    model = train(model, train_loader, val_loader, args, device, out_dir,
                  phase_label="Phase 2", lr=lr_p2,
                  loader_override=train_loader_p2 if batch_p2 != args.batch else None)

    # --- Test set evaluation ---
    print("\n=== Test Set Evaluation ===")
    metrics, preds, labels = compute_metrics(model, test_loader, device)

    try:
        from sklearn.metrics import roc_auc_score
        for i, cls in enumerate(["bac", "fung"]):
            metrics[cls]["auroc"] = roc_auc_score(labels[:, i] > 0.5, preds[:, i])
    except ImportError:
        print("  (install scikit-learn for AUROC)")

    for cls in ["bac", "fung"]:
        m = metrics[cls]
        print(f"\n  {cls.upper()} keratitis:")
        for k, v in m.items():
            print(f"    {k:<12} {v:.4f}")

    metrics_path = os.path.join(out_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAll outputs saved to: {out_dir}/")
    print(f"  best_model.pt     — best weights (restored after early stopping)")
    print(f"  train_log.csv     — per-epoch metrics")
    print(f"  test_metrics.json — final test results")
    print(f"  config.json       — run configuration")


if __name__ == "__main__":
    main()