"""
Latent Supervision - Infectious Keratitis Classifier (v2)
=========================================================
Trains one cell of the 2x2 study: head (sigmoid | beta_dirichlet) x labels (soft | hard).

v2 changes (2026-10-21; see the audit log, findings C1-C4, M1-M2, M9, R1-R5):
  * Labels come from labels.py: soft target = posterior where one exists, otherwise
    the LCA bin; reference = LCA bin; scored as binary bacterial vs fungal.
  * --recipe paper (default) follows Yazbeck et al., supplement section 3, latent
    supervision keratitis model, and the R demo where the paper is silent:
        EfficientNetB7 (ImageNet weights), 600x600 input.
        Trainable from Keras layer 753 = start of the last stage (timm backbone.blocks[6])
        plus the top conv and the head; every backbone BatchNorm frozen (inference mode).
        Adam, learning rate 1e-5, gradient clipnorm 1 (per weight, as Keras does),
        dropout 0.3, batch 32, 15 epochs, one phase, no early stopping.
        The final-epoch model is the model (the demo saves no checkpoints).
        Augmentation (demo): rotation +-54 deg, translation +-10%, horizontal and vertical
        flips, contrast +-10%, reflect fill.
      Keras defaults matched where cheap (--keras_parity 1, default for paper):
        stochastic depth 0.2 in the backbone (Keras drop_connect_rate), head BatchNorm
        eps 1e-3 / momentum 0.99, Adam epsilon 1e-7, Glorot-uniform head initialisation.
  * --recipe legacy keeps the previous two-phase schedule, for comparison only.
  * Fails loudly: missing images stop the run at start-up, unreadable images raise,
    a non-finite loss raises.
  * Split grouped by pID (was subject).
  * Outputs go under --out_root and never into a folder called runs/.
  * Writes test_predictions.npz (no identifiers) for the analysis.

Typical use (paper recipe, Cell A):
    python src/model.py --head sigmoid --label_type soft \
        --out_root /scratch90/chris_/ik_uq_runs --device cuda:0
Check labels and images only:
    python src/model.py --dry_run
"""

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import timm
import torch
from timm.data import resolve_data_config
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import (STUDY_SHORT, label_summary, load_labelled,  # noqa: E402
                    participant_codes, split_by_participant, targets)

DEFAULT_CSV = ("/persist/QTIM/Active/26-0096/Corneal_Ulcer_Image_Sets/Data/"
               "corneal_ulcer_study_data_7_24_2025_lambda.csv")
DEFAULT_IMG_ROOT = "/pacs/OSHU/Corneal_Ulcer_Image_Sets"

RECIPES = {
    #            epochs  batch  patience  keras_parity
    "paper":  dict(epochs=15, batch=32, patience=0, keras_parity=1),
    "legacy": dict(epochs=30, batch=8,  patience=5, keras_parity=0),
}


# =============================================================================
# ARGUMENTS
# =============================================================================

def get_args(argv=None):
    p = argparse.ArgumentParser(description="Latent Supervision Keratitis Classifier (v2)")
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--img_root", default=DEFAULT_IMG_ROOT)
    p.add_argument("--head", default="sigmoid", choices=["sigmoid", "dirichlet", "beta_dirichlet"])
    p.add_argument("--label_type", default="soft", choices=["soft", "hard"])
    p.add_argument("--recipe", default="paper", choices=list(RECIPES))
    p.add_argument("--epochs", type=int, default=None, help="paper 15; legacy 30 per phase")
    p.add_argument("--batch", type=int, default=None, help="paper 32; legacy 8")
    p.add_argument("--accum", type=int, default=1,
                   help="gradient accumulation steps (effective batch = batch x accum); "
                        "use e.g. --batch 16 --accum 2 if batch 32 does not fit")
    p.add_argument("--batch_phase2", type=int, default=None, help="legacy only")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr_phase2", type=float, default=None, help="legacy only")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=None, help="early stopping; paper 0 (off)")
    p.add_argument("--keras_parity", type=int, default=None, choices=[0, 1])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--group_col", default="pID", help="column the split is grouped by")
    p.add_argument("--exclude_followups_train", action="store_true",
                   help="drop follow-up photos (3 weeks, 3 months, 12 months) from training")
    p.add_argument("--backbone", default="tf_efficientnet_b7.ra_in1k")
    p.add_argument("--no_pretrained", action="store_true", help="testing only")
    p.add_argument("--img_size", type=int, default=600)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_root", default=None, help="parent folder for run folders (not runs/)")
    p.add_argument("--out_dir", default=None, help="exact run folder (overrides --out_root)")
    p.add_argument("--run_name", default=None)
    p.add_argument("--device", default=None, help="e.g. cuda:0")
    p.add_argument("--dry_run", action="store_true", help="check labels and images, then exit")
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end run to test the pipeline")
    args = p.parse_args(argv)
    for k, v in RECIPES[args.recipe].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.smoke:
        args.epochs = 1
    return args


def git_info():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True,
                                text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=here,
                                    capture_output=True, text=True, timeout=10).stdout.strip())
        return {"commit": commit or None, "uncommitted_changes": dirty}
    except Exception:
        return {"commit": None, "uncommitted_changes": None}


def make_output_dir(args):
    if args.out_dir:
        out_dir = args.out_dir
    else:
        if not args.out_root:
            raise SystemExit("Give --out_root (a folder outside runs/) or --out_dir.")
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = args.run_name or f"{args.head}_{args.label_type}_{args.recipe}_{ts}"
        if args.smoke:
            name = "SMOKE_" + name
        out_dir = os.path.join(args.out_root, name)
    if "runs" in os.path.normpath(os.path.abspath(out_dir)).split(os.sep):
        raise SystemExit(f"Refusing to write inside a folder called 'runs' ({out_dir}): "
                         f"that is where the old results live. Use a different --out_root.")
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise SystemExit(f"{out_dir} already exists and is not empty.")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}\n")
    return out_dir


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


# =============================================================================
# DATA
# =============================================================================

class KeratitisDataset(Dataset):
    def __init__(self, paths, y, transform):
        self.paths = list(paths)
        self.y = torch.as_tensor(np.asarray(y, dtype=np.float32))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
        except Exception as e:  # fail loudly (audit M9): never substitute a blank image
            raise RuntimeError(f"Could not read image {self.paths[i]}: {e}") from e
        return self.transform(img), self.y[i]


class KerasStyleAugment:
    """
    The R demo's augmentation, applied to the 600x600 input as Keras does:
      layer_random_rotation(0.15)            -> angle uniform in +-0.15 * 360 deg = +-54 deg
      layer_random_translation(0.1, 0.1)     -> shift uniform in +-10% of height / width
      layer_random_flip()                    -> horizontal and vertical, each with p = 0.5
      layer_random_contrast(0.1)             -> (x - channel mean) * U(0.9, 1.1) + mean, clipped
    Keras fills with reflection (fill_mode="reflect"); here: reflect-pad, transform, crop.
    """

    def __init__(self, size, mean, std, rotation=0.15, translation=0.1, contrast=0.1):
        self.size, self.mean, self.std = size, list(mean), list(std)
        self.rotation, self.translation, self.contrast = rotation, translation, contrast

    def __call__(self, img):
        s = self.size
        img = TF.resize(img, [s, s], interpolation=InterpolationMode.BILINEAR)
        pad = s // 2
        img = TF.pad(img, [pad, pad, pad, pad], padding_mode="reflect")
        angle = random.uniform(-self.rotation * 360.0, self.rotation * 360.0)
        tx = int(round(random.uniform(-self.translation, self.translation) * s))
        ty = int(round(random.uniform(-self.translation, self.translation) * s))
        img = TF.affine(img, angle=angle, translate=[tx, ty], scale=1.0, shear=[0.0, 0.0],
                        interpolation=InterpolationMode.BILINEAR)
        img = TF.center_crop(img, [s, s])
        if random.random() < 0.5:
            img = TF.hflip(img)
        if random.random() < 0.5:
            img = TF.vflip(img)
        x = TF.to_tensor(img)
        f = random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
        m = x.mean(dim=(1, 2), keepdim=True)
        x = ((x - m) * f + m).clamp(0.0, 1.0)
        return TF.normalize(x, self.mean, self.std)


def eval_transform(size, mean, std):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(list(mean), list(std)),
    ])


def legacy_train_transform(size):
    return transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(contrast=0.1),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# =============================================================================
# HEADS
# =============================================================================

def _head_bn(n, keras):
    # Keras BatchNormalization: epsilon 1e-3, momentum 0.99 (= PyTorch momentum 0.01)
    return nn.BatchNorm1d(n, eps=1e-3, momentum=0.01) if keras else nn.BatchNorm1d(n)


def _glorot(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)


class SigmoidHead(nn.Module):
    """The demo's head: GAP -> BN -> 512 ReLU -> dropout -> 32 ReLU -> dropout -> 2 sigmoid."""

    def __init__(self, in_features, dropout=0.3, keras=False):
        super().__init__()
        self.net = nn.Sequential(
            _head_bn(in_features, keras),
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(512, 32), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(32, 2),
            nn.Sigmoid(),
        )
        if keras:
            _glorot(self)

    def forward(self, x):
        return self.net(x)


class DirichletHead(nn.Module):
    """Joint 2-class Dirichlet head (ICMLA). Outputs alpha = [alpha_bac, alpha_fung] > 0."""

    def __init__(self, in_features, dropout=0.3, keras=False):
        super().__init__()
        self.net = nn.Sequential(
            _head_bn(in_features, keras),
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(512, 32), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(32, 2),
            nn.Softplus(),
        )
        if keras:
            _glorot(self)

    def forward(self, x):
        return self.net(x)


class BetaDirichletHead(nn.Module):
    """
    Two K=2 Dirichlet (Beta) outputs, one per pathogen, on a shared trunk.
    Outputs [alpha_bac, beta_bac, alpha_fung, beta_fung] > 0.
    With binary labels the two Betas describe the same event, so expect them to mirror
    each other; mean evidence is (alpha_bac + beta_bac + alpha_fung + beta_fung) / 2.
    """

    def __init__(self, in_features, dropout=0.3, keras=False):
        super().__init__()
        self.trunk = nn.Sequential(
            _head_bn(in_features, keras),
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(512, 32), nn.ReLU(), nn.Dropout(p=dropout),
        )
        self.bac_head = nn.Sequential(nn.Linear(32, 2), nn.Softplus())
        self.fung_head = nn.Sequential(nn.Linear(32, 2), nn.Softplus())
        if keras:
            _glorot(self)

    def forward(self, x):
        f = self.trunk(x)
        return torch.cat([self.bac_head(f), self.fung_head(f)], dim=-1)


# =============================================================================
# MODEL
# =============================================================================

BN_TYPES = nn.modules.batchnorm._BatchNorm


class EfficientNetB7(nn.Module):
    def __init__(self, head_type="sigmoid", dropout=0.3, pretrained=True,
                 backbone_name="tf_efficientnet_b7.ra_in1k", drop_path_rate=0.0, keras=False):
        super().__init__()
        assert head_type in ("sigmoid", "dirichlet", "beta_dirichlet")
        self.head_type = head_type
        self.freeze_backbone_bn = False
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0,
                                          global_pool="avg", drop_path_rate=drop_path_rate)
        for prm in self.backbone.parameters():
            prm.requires_grad = False
        nf = self.backbone.num_features
        head_cls = {"sigmoid": SigmoidHead, "dirichlet": DirichletHead,
                    "beta_dirichlet": BetaDirichletHead}[head_type]
        self.head = head_cls(nf, dropout, keras)

    def forward(self, x):
        return self.head(self.backbone(x))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone_bn:  # Keras: frozen BatchNorm runs in inference mode
            for m in self.backbone.modules():
                if isinstance(m, BN_TYPES):
                    m.eval()
        return self

    def set_trainable_last_stage(self):
        """
        Paper recipe: train from Keras layer 753 (start of block 7, the last stage) onward,
        i.e. timm backbone.blocks[-1] and conv_head, plus the head. Every backbone
        BatchNorm stays frozen and in inference mode, as in the demo's unfreeze loop.
        Parameters are set per module (recurse=False) so containers cannot re-enable BN.
        """
        self.freeze_backbone_bn = True
        for prm in self.backbone.parameters():
            prm.requires_grad = False
        last = len(self.backbone.blocks) - 1
        prefixes = (f"blocks.{last}.", "conv_head")
        for name, m in self.backbone.named_modules():
            if not name.startswith(prefixes) or isinstance(m, BN_TYPES):
                continue
            for prm in m.parameters(recurse=False):
                prm.requires_grad = True
        for prm in self.head.parameters():
            prm.requires_grad = True
        self.train(self.training)
        return last

    def unfreeze(self, from_block=54):
        """Legacy recipe only: the old Phase 2 unfreeze, unchanged (see audit M1)."""
        for i, (name, module) in enumerate(self.backbone.named_modules()):
            if i >= from_block and not isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                for prm in module.parameters():
                    prm.requires_grad = True

    def p_hat(self, out):
        """Per-pathogen probabilities [p_bac, p_fung] from raw outputs."""
        if self.head_type == "sigmoid":
            return out
        if self.head_type == "dirichlet":
            return out / out.sum(dim=-1, keepdim=True)
        return torch.stack([out[:, 0] / (out[:, 0] + out[:, 1]),
                            out[:, 2] / (out[:, 2] + out[:, 3])], dim=-1)


def count_trainable(model):
    return sum(prm.numel() for prm in model.parameters() if prm.requires_grad)


# =============================================================================
# LOSSES (unchanged)
# =============================================================================

def rescaled_mse_loss(y_true, y_pred):
    """The paper's loss: MSE x 10."""
    return ((y_true - y_pred) ** 2 * 10).mean()


def dirichlet_loss(y_true, alpha, lambda_kl):
    """Dirichlet MSE + variance term (ICMLA eq. 5) + KL on alpha_tilde (Sensoy et al.)."""
    alpha_0 = alpha.sum(dim=-1, keepdim=True)
    p_hat = alpha / alpha_0
    mse_term = ((y_true - p_hat) ** 2 + p_hat * (1 - p_hat) / (alpha_0 + 1)).sum(dim=-1)
    alpha_tilde = y_true + (1.0 - y_true) * alpha
    alpha_tilde_0 = alpha_tilde.sum(dim=-1)
    K = alpha.shape[-1]
    kl_term = (torch.lgamma(alpha_tilde_0)
               - torch.lgamma(torch.tensor(float(K), device=alpha.device))
               - torch.lgamma(alpha_tilde).sum(dim=-1)
               + ((alpha_tilde - 1.0) * (torch.digamma(alpha_tilde)
                                         - torch.digamma(alpha_tilde_0.unsqueeze(-1)))).sum(dim=-1))
    return (mse_term + lambda_kl * kl_term).mean()


def beta_dirichlet_loss(y_true, params, lambda_kl):
    y_bac = torch.stack([y_true[:, 0], 1 - y_true[:, 0]], dim=-1)
    y_fung = torch.stack([y_true[:, 1], 1 - y_true[:, 1]], dim=-1)
    return (dirichlet_loss(y_bac, params[:, :2], lambda_kl)
            + dirichlet_loss(y_fung, params[:, 2:], lambda_kl))


def kl_schedule(epoch, total_epochs, lambda_0=1e-4, num_classes=2):
    """ICMLA eq. 7. Negligible at K = 2 (max about 5e-5); kept as published."""
    return (lambda_0 / num_classes) * (epoch / max(total_epochs, 1))


def compute_loss(head_type, y_true, output, epoch=0, total_epochs=1):
    if head_type == "sigmoid":
        return rescaled_mse_loss(y_true, output)
    lam = kl_schedule(epoch, total_epochs, num_classes=2)
    if head_type == "dirichlet":
        return dirichlet_loss(y_true, output, lam)
    return beta_dirichlet_loss(y_true, output, lam)


# =============================================================================
# TRAINING
# =============================================================================

def make_optimizer(model, lr, keras):
    params = [prm for prm in model.parameters() if prm.requires_grad]
    return torch.optim.Adam(params, lr=lr, eps=1e-7 if keras else 1e-8)


def clip_gradients(model, keras):
    if keras:  # Keras clipnorm=1 clips each weight's gradient separately
        for prm in model.parameters():
            if prm.grad is not None:
                torch.nn.utils.clip_grad_norm_([prm], max_norm=1.0)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


def train_one_epoch(model, loader, optimizer, epoch, total_epochs, device, accum, keras):
    model.train()
    total, n = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    bar = tqdm(loader, desc="  Train", unit="batch", leave=False)
    for step, (x, y) in enumerate(bar):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        loss = compute_loss(model.head_type, y, model(x), epoch, total_epochs)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss ({loss.item()}) at epoch {epoch + 1}, "
                                     f"step {step}. Stopping instead of training on NaN.")
        (loss / accum).backward()
        if (step + 1) % accum == 0 or (step + 1) == len(loader):
            clip_gradients(model, keras)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total += loss.item()
        n += 1
        bar.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, epoch, total_epochs, device):
    model.eval()
    total, ev, n = 0.0, 0.0, 0
    for x, y in tqdm(loader, desc="  Val  ", unit="batch", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        total += compute_loss(model.head_type, y, out, epoch, total_epochs).item()
        if model.head_type == "beta_dirichlet":
            ev += ((out[:, 0] + out[:, 1] + out[:, 2] + out[:, 3]) / 2).mean().item()
        elif model.head_type == "dirichlet":
            ev += out.sum(dim=-1).mean().item()
        n += 1
    res = {"val_loss": total / max(n, 1)}
    if model.head_type != "sigmoid":
        res["mean_evidence"] = ev / max(n, 1)
    if not np.isfinite(res["val_loss"]):
        raise FloatingPointError(f"Non-finite validation loss at epoch {epoch + 1}.")
    return res


def log_epoch(log_path, row):
    pd.DataFrame([row]).to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)


def train_paper(model, train_loader, val_loader, args, device, out_dir, keras):
    """One phase, fixed epochs, no early stopping; the final-epoch model is kept."""
    optimizer = make_optimizer(model, args.lr, keras)
    log_path = os.path.join(out_dir, "train_log.csv")
    best = float("inf")
    for epoch in range(args.epochs):
        tl = train_one_epoch(model, train_loader, optimizer, epoch, args.epochs, device,
                             args.accum, keras)
        if epoch == 0 and device.type == "cuda":
            gb = torch.cuda.max_memory_allocated(device) / 1e9
            tqdm.write(f"  Peak GPU memory after epoch 1: {gb:.1f} GB")
        val = evaluate(model, val_loader, epoch, args.epochs, device)
        improved = val["val_loss"] < best
        if improved:
            best = val["val_loss"]
            torch.save(model.state_dict(), os.path.join(out_dir, "best_val_model.pt"))
        log_epoch(log_path, {"phase": "paper", "epoch": epoch + 1, "train_loss": tl,
                             **val, "best_so_far": improved})
        tqdm.write(f"  epoch {epoch + 1}/{args.epochs}  train {tl:.4f}  val {val['val_loss']:.4f}"
                   + (f"  evidence {val['mean_evidence']:.3f}" if "mean_evidence" in val else ""))
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    return model


def train_legacy(model, loaders, args, device, out_dir):
    """The previous two-phase schedule, with a checkpoint per phase and the best carried across."""
    train_loader, train_loader_p2, val_loader = loaders
    log_path = os.path.join(out_dir, "train_log.csv")
    best_overall, best_file = float("inf"), None
    for phase in (1, 2):
        if phase == 2:
            model.unfreeze(from_block=54)
        lr = args.lr if phase == 1 else (args.lr_phase2 or args.lr)
        loader = train_loader if phase == 1 else train_loader_p2
        optimizer = make_optimizer(model, lr, keras=False)
        ckpt = os.path.join(out_dir, f"best_phase{phase}.pt")
        best, no_imp = float("inf"), 0
        for epoch in range(args.epochs):
            tl = train_one_epoch(model, loader, optimizer, epoch, args.epochs, device, args.accum, False)
            val = evaluate(model, val_loader, epoch, args.epochs, device)
            improved = val["val_loss"] < best
            if improved:
                best, no_imp = val["val_loss"], 0
                torch.save(model.state_dict(), ckpt)
            else:
                no_imp += 1
            log_epoch(log_path, {"phase": phase, "epoch": epoch + 1, "train_loss": tl, **val,
                                 "best_so_far": improved})
            if args.patience > 0 and no_imp >= args.patience:
                break
        model.load_state_dict(torch.load(ckpt, map_location=device))
        if best < best_overall:
            best_overall, best_file = best, ckpt
    model.load_state_dict(torch.load(best_file, map_location=device))
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    return model


# =============================================================================
# TEST PREDICTIONS AND METRICS
# =============================================================================

@torch.no_grad()
def predict(model, paths, transform, batch, device, num_workers):
    ds = KeratitisDataset(paths, np.zeros((len(paths), 2)), transform)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda")
    model.eval()
    outs = [model(x.to(device)).cpu() for x, _ in tqdm(loader, desc="  Predict", leave=False)]
    out = torch.cat(outs)
    return out.numpy(), model.p_hat(out).numpy()


def binary_metrics(y, score, p):
    """AUROC of the bacterial-vs-fungal score and of each pathogen output, accuracy at 0.5."""
    if len(y) == 0:
        return {"n_images": 0}
    m = {"n_images": int(len(y)), "bacterial_share": float(np.mean(y))}
    if len(np.unique(y)) == 2:
        m["auroc_binary"] = float(roc_auc_score(y, score))
        m["auroc_bac_output"] = float(roc_auc_score(y, p[:, 0]))
        m["auroc_fung_output"] = float(roc_auc_score(1 - y, p[:, 1]))
    m["accuracy"] = float(np.mean((score > 0.5) == (y == 1)))
    return m


# =============================================================================
# MAIN
# =============================================================================

def main(argv=None):
    args = get_args(argv)
    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    keras = bool(args.keras_parity)

    # --- Labels and split -----------------------------------------------------
    df = load_labelled(args.csv, args.img_root, args.group_col)
    parts = split_by_participant(df, seed=args.seed, group_col=args.group_col)
    if args.smoke:  # a small random subset of each part
        rng = np.random.default_rng(args.seed)
        pick = lambda a, n: np.sort(rng.permutation(a)[:n])
        parts = {"train": pick(parts["train"], 48), "val": pick(parts["val"], 16),
                 "cal": pick(parts["cal"], 8), "eval": pick(parts["eval"], 24)}
        parts["test"] = np.concatenate([parts["cal"], parts["eval"]])
    tr = parts["train"]
    if args.exclude_followups_train:
        tr = tr[~df["is_followup"].to_numpy()[tr]]
    test = np.concatenate([parts["cal"], parts["eval"]])
    part_flag = np.r_[np.zeros(len(parts["cal"]), int), np.ones(len(parts["eval"]), int)]

    print(label_summary(df, args.group_col).to_string(), "\n")
    print(f"Split (grouped by {args.group_col}): train {len(tr):,}, val {len(parts['val']):,}, "
          f"test {len(test):,} (cal {len(parts['cal']):,} / eval {len(parts['eval']):,})\n")

    # --- Images must all exist before anything trains -------------------------
    used = np.concatenate([tr, parts["val"], test])
    missing = [p for p in df["local_path"].to_numpy()[used] if not os.path.exists(p)]
    if missing:
        print(f"{len(missing)} image files are missing, e.g.:")
        for p in missing[:10]:
            print("  ", p)
        raise SystemExit("Stopping: fix the image paths before training.")
    print(f"All {len(used):,} image files found.")

    cfg = resolve_data_config({}, model=timm.create_model(args.backbone, pretrained=False,
                                                            num_classes=0))
    mean, std = cfg["mean"], cfg["std"]
    if args.recipe == "paper":
        train_tf = KerasStyleAugment(args.img_size, mean, std)
        eval_tf = eval_transform(args.img_size, mean, std)
    else:
        train_tf = legacy_train_transform(args.img_size)
        eval_tf = eval_transform(args.img_size, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    if args.dry_run:
        bad = 0
        for s, g in df.iloc[used].groupby("study"):
            for p in g["local_path"].head(3):
                try:
                    Image.open(p).convert("RGB")
                except Exception as e:
                    bad += 1
                    print(f"  cannot read {p}: {e}")
        print("Dry run finished: labels OK, all files present"
              + ("." if not bad else f", but {bad} sample images could not be read."))
        return

    out_dir = make_output_dir(args)

    # --- Model ----------------------------------------------------------------
    model = EfficientNetB7(head_type=args.head, dropout=args.dropout,
                           pretrained=not args.no_pretrained, backbone_name=args.backbone,
                           drop_path_rate=0.2 if keras else 0.0, keras=keras).to(device)
    stage = model.set_trainable_last_stage() if args.recipe == "paper" else None
    n_train_params = count_trainable(model)
    print(f"Recipe {args.recipe}: trainable parameters {n_train_params:,}"
          + (f" (backbone.blocks[{stage}] + conv_head + head; backbone BatchNorm frozen)"
             if stage is not None else " (Phase 1: head only)"))

    # --- Record everything about the run --------------------------------------
    config = {**vars(args), "git": git_info(), "trainable_params": n_train_params,
              "last_stage_index": stage, "norm_mean": list(mean), "norm_std": list(std),
              "n_train": int(len(tr)), "n_val": int(len(parts["val"])), "n_test": int(len(test)),
              "started": datetime.now().isoformat(timespec="seconds"),
              "torch": torch.__version__, "timm": timm.__version__}
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    split_tab = pd.DataFrame({"csv_row": df["csv_row"].to_numpy()})
    split_tab["part"] = "unused"
    for k in ("train", "val", "cal", "eval"):
        split_tab.loc[parts[k], "part"] = k
    if args.exclude_followups_train:
        dropped = np.setdiff1d(parts["train"], tr)
        split_tab.loc[dropped, "part"] = "train_excluded_followup"
    split_tab.to_csv(os.path.join(out_dir, "split.csv"), index=False)
    label_summary(df, args.group_col).to_csv(os.path.join(out_dir, "label_summary.csv"))

    # --- Loaders ----------------------------------------------------------------
    y_all = targets(df, args.label_type)
    paths = df["local_path"].to_numpy()
    g = torch.Generator().manual_seed(args.seed)
    nw, pin = args.num_workers, device.type == "cuda"

    def loader(idx, tf, batch, shuffle):
        ds = KeratitisDataset(paths[idx], y_all[idx], tf)
        drop_last = shuffle and len(ds) % batch == 1  # BatchNorm cannot train on a batch of 1
        return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=nw, pin_memory=pin,
                          worker_init_fn=worker_init, generator=g, drop_last=drop_last)

    train_loader = loader(tr, train_tf, args.batch, True)
    val_loader = loader(parts["val"], eval_tf, args.batch, False)

    # --- Train ------------------------------------------------------------------
    print(f"\nTraining {args.head} / {args.label_type} / {args.recipe}: {args.epochs} epochs, "
          f"batch {args.batch} x accum {args.accum}, lr {args.lr}, device {device}\n")
    if args.recipe == "paper":
        model = train_paper(model, train_loader, val_loader, args, device, out_dir, keras)
    else:
        p2 = loader(tr, train_tf, args.batch_phase2 or args.batch, True)
        model = train_legacy(model, (train_loader, p2, val_loader), args, device, out_dir)

    # --- Test predictions (no identifiers) ---------------------------------------
    out, p = predict(model, paths[test], eval_tf, args.batch, device, nw)
    score = (p[:, 0] + 1.0 - p[:, 1]) / 2.0  # one bacterial probability from the two outputs
    sub = df.iloc[test]
    np.savez(os.path.join(out_dir, "test_predictions.npz"),
             head=args.head, label_type=args.label_type, recipe=args.recipe,
             output=out, p_hat=p, p_bac_binary=score,
             y_bac=sub["y_bac"].to_numpy(), soft_bac=sub["soft_bac"].to_numpy(),
             soft_fung=sub["soft_fung"].to_numpy(), has_posterior=sub["has_posterior"].to_numpy(),
             is_followup=sub["is_followup"].to_numpy(), visit=sub["visit"].to_numpy().astype(str),
             study=np.array([STUDY_SHORT.get(s, s) for s in sub["study"]]),
             csv_row=sub["csv_row"].to_numpy(),
             participant=participant_codes(df, args.group_col)[test], part=part_flag)

    y = sub["y_bac"].to_numpy()
    mad = sub["study"].eq("MADURAI photos").to_numpy()
    ev = part_flag == 1
    metrics = {"reference": "LCA bin (binary, bacterial vs fungal)",
               "score": "(p_bac + 1 - p_fung) / 2",
               "test_all": binary_metrics(y, score, p),
               "eval": binary_metrics(y[ev], score[ev], p[ev]),
               "eval_MADURAI": binary_metrics(y[ev & mad], score[ev & mad], p[ev & mad]),
               "eval_other_studies": binary_metrics(y[ev & ~mad], score[ev & ~mad], p[ev & ~mad])}
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    for k in ("eval", "eval_MADURAI", "eval_other_studies"):
        m = metrics[k]
        print(f"  {k:20s} n={m['n_images']:4d}  AUROC {m.get('auroc_binary', float('nan')):.3f}  "
              f"accuracy {m.get('accuracy', float('nan')):.3f}")
    with open(os.path.join(out_dir, "config.json")) as f:
        config = json.load(f)
    config["finished"] = datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nDone. Outputs in {out_dir}: model.pt (final epoch), best_val_model.pt (reference only), "
          f"train_log.csv, test_predictions.npz, test_metrics.json, split.csv, config.json")


if __name__ == "__main__":
    main()
