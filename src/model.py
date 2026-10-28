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

v2.1 (2026-10-23):
  * --n_folds K --fold k: K-fold cross-validation grouped by pID and stratified by study x label
    (labels.split_cv). The same seed gives the same folds for every model; pool the K runs of a
    model in the analysis. Without --n_folds the single split is unchanged.
  * --label_type smear | culture: the paper's test-supervised baselines (supplement section 3).
    Two sigmoid outputs, each trained on its own test (Gram and KOH, or blood agar and PDA),
    binary cross-entropy over the tests that were done; results carried across a participant's
    images. Supplement settings: smear 11 epochs, trainable from Keras layer 753 (last stage);
    culture 22 epochs, from layer 559 (= start of block 6, timm blocks[5], so two stages train).
    Only images with at least one of the two tests train; every test image is predicted.
  * test_predictions.npz also stores the fold and each image's participant-level test results.

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
from labels import (CULTURES, PAPER_LABEL_TYPES, STUDY_SHORT, TEST_TARGETS, TESTS,  # noqa: E402
                    has_culture, label_summary, load_labelled, load_paper_table, participant_codes,
                    split_by_participant, split_cv, split_paper, targets, test_results)

DEFAULT_CSV = ("/persist/QTIM/Active/26-0096/Corneal_Ulcer_Image_Sets/Data/"
               "corneal_ulcer_study_data_7_24_2025_lambda.csv")
DEFAULT_IMG_ROOT = "/pacs/OSHU/Corneal_Ulcer_Image_Sets"

RECIPES = {
    #            epochs  batch  patience  keras_parity
    "paper":  dict(epochs=15, batch=32, patience=0, keras_parity=1),
    "legacy": dict(epochs=30, batch=8,  patience=5, keras_parity=0),
}
# Supplement section 3, test-supervised baselines (paper recipe). Keras layer 753 = start of the
# last stage (timm blocks[6]); layer 559 = start of block 6 (timm blocks[5]).
TEST_RECIPES = {"smear": dict(epochs=11, trainable_stages=1),
                "culture": dict(epochs=22, trainable_stages=2)}


# =============================================================================
# ARGUMENTS
# =============================================================================

def get_args(argv=None):
    p = argparse.ArgumentParser(description="Latent Supervision Keratitis Classifier (v2)")
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--img_root", default=DEFAULT_IMG_ROOT)
    p.add_argument("--head", default="sigmoid", choices=["sigmoid", "dirichlet", "beta_dirichlet"])
    p.add_argument("--label_type", default="soft",
                   choices=["soft", "hard"] + list(TEST_TARGETS) + PAPER_LABEL_TYPES,
                   help="soft / hard: our CSV's labels; smear / culture: test-supervised baselines; "
                        "paper_soft / paper_hard / twoclass_soft / twoclass_hard: the redo table (--labels_csv)")
    p.add_argument("--labels_csv", default=None,
                   help="redo: paper_labels.csv from src/paper_labels.py (the paper's images, split and labels)")
    p.add_argument("--subset", default="all", choices=["all", "culture"],
                   help="redo: 'culture' keeps only images with at least one culture (blood agar or PDA) in every "
                        "split: the culture analysis")
    p.add_argument("--val_share", type=float, default=0.0,
                   help="redo, no validation file: share of training subjects held out to monitor "
                        "validation loss (default 0: none; the paper recipe never uses it to pick a model)")
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
    p.add_argument("--kl_schedule", default="icmla", choices=["icmla", "sensoy"],
                   help="BD/Dirichlet heads: icmla = ICMLA eq. 7 (default, about 5e-5 at most); "
                        "sensoy = kl_max * min(1, epoch / 10)")
    p.add_argument("--kl_max", type=float, default=None, help="full KL weight for --kl_schedule sensoy")
    p.add_argument("--kl_form", default="sensoy", choices=["sensoy", "icmla_eq5"],
                   help="sensoy = KL on alpha_tilde (default); icmla_eq5 = KL on alpha")
    p.add_argument("--trainable_stages", type=int, default=None,
                   help="paper recipe: number of final backbone stages that train (1 = Keras layer "
                        "753 onward, the default; culture baseline 2 = layer 559 onward)")
    p.add_argument("--n_folds", type=int, default=0, help="cross-validation folds (0 = single split)")
    p.add_argument("--fold", type=int, default=None, help="which fold is the test set (0-based)")
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
    if args.labels_csv:
        if args.label_type in ("soft", "hard"):
            p.error("with --labels_csv use paper_soft, paper_hard, culture_hard, twoclass_soft, twoclass_hard, "
                    "smear or culture")
        if args.label_type == "culture_hard" and args.subset != "culture":
            p.error("--label_type culture_hard needs --subset culture (images with a culture)")
        if args.group_col == "pID":
            args.group_col = "subject"
    elif args.subset != "all":
        p.error("--subset needs --labels_csv")
    elif args.label_type in PAPER_LABEL_TYPES:
        p.error(f"--label_type {args.label_type} needs --labels_csv")
    if args.label_type in TEST_TARGETS:
        if args.head != "sigmoid":
            p.error("--label_type smear / culture needs --head sigmoid (the paper's baselines)")
        if args.recipe == "paper":
            for k, v in TEST_RECIPES[args.label_type].items():
                if getattr(args, k) is None:
                    setattr(args, k, v)
    if args.trainable_stages is None:
        args.trainable_stages = 1
    if args.n_folds:
        if args.n_folds < 2:
            p.error("--n_folds must be at least 2 (or 0 for the single split)")
        if args.fold is None or not 0 <= args.fold < args.n_folds:
            p.error(f"--n_folds {args.n_folds} needs --fold between 0 and {args.n_folds - 1}")
    elif args.fold is not None:
        p.error("--fold needs --n_folds")
    for k, v in RECIPES[args.recipe].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.smoke:
        args.epochs = 1
    if args.kl_schedule == "sensoy" and args.kl_max is None:
        p.error("--kl_schedule sensoy needs --kl_max")
    if args.kl_schedule == "icmla" and args.kl_max is not None:
        p.error("--kl_max only applies to --kl_schedule sensoy")
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
        kl = ("" if (args.kl_schedule, args.kl_form) == ("icmla", "sensoy")
              else f"_kl-{args.kl_schedule}{args.kl_max if args.kl_max is not None else ''}-{args.kl_form}")
        fold = f"_fold{args.fold}of{args.n_folds}" if args.n_folds else ""
        sub = "_culture-subset" if args.subset == "culture" else ""
        name = args.run_name or f"{args.head}_{args.label_type}{sub}_{args.recipe}{kl}{fold}_{ts}"
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

    def set_trainable_last_stage(self, n_stages=1):
        """
        Paper recipe: train from Keras layer 753 (start of block 7, the last stage) onward,
        i.e. timm backbone.blocks[-1] and conv_head, plus the head (n_stages = 1). The culture
        baseline trains from layer 559, the start of block 6 (n_stages = 2). Every backbone
        BatchNorm stays frozen and in inference mode, as in the demo's unfreeze loop.
        Parameters are set per module (recurse=False) so containers cannot re-enable BN.
        Returns the index of the first trainable stage.
        """
        self.freeze_backbone_bn = True
        for prm in self.backbone.parameters():
            prm.requires_grad = False
        n_blocks = len(self.backbone.blocks)
        if not 1 <= n_stages <= n_blocks:
            raise ValueError(f"n_stages must be in 1..{n_blocks}")
        first = n_blocks - n_stages
        prefixes = tuple(f"blocks.{i}." for i in range(first, n_blocks)) + ("conv_head",)
        for name, m in self.backbone.named_modules():
            if not name.startswith(prefixes) or isinstance(m, BN_TYPES):
                continue
            for prm in m.parameters(recurse=False):
                prm.requires_grad = True
        for prm in self.head.parameters():
            prm.requires_grad = True
        self.train(self.training)
        return first

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


def masked_bce_loss(y_true, y_pred):
    """Binary cross-entropy per output over the tests that were done (NaN = not done).
    Probabilities clipped at 1e-7 as Keras does."""
    obs = ~torch.isnan(y_true)
    if not obs.any():
        return y_pred.sum() * 0.0
    p = y_pred.clamp(1e-7, 1 - 1e-7)
    y = torch.nan_to_num(y_true)
    bce = -(y * torch.log(p) + (1 - y) * torch.log(1 - p))
    return bce[obs].mean()


# Loss for sigmoid heads; set in main(): "mse" (latent supervision) or "bce" (test baselines).
LOSS = {"kind": "mse"}


def dirichlet_kl_to_uniform(a):
    """KL[Dir(a) || Dir(1, ..., 1)] for each row of a."""
    a0 = a.sum(dim=-1)
    K = a.shape[-1]
    return (torch.lgamma(a0)
            - torch.lgamma(torch.tensor(float(K), device=a.device))
            - torch.lgamma(a).sum(dim=-1)
            + ((a - 1.0) * (torch.digamma(a) - torch.digamma(a0.unsqueeze(-1)))).sum(dim=-1))


def dirichlet_loss(y_true, alpha, lambda_kl, kl_form="sensoy"):
    """
    Dirichlet MSE + variance term (ICMLA eq. 5) + lambda * KL regulariser.
      kl_form "sensoy":    KL[Dir(alpha_tilde) || Dir(1)], alpha_tilde = y + (1 - y) * alpha
                           (Sensoy et al. 2018: evidence for the true class is removed first,
                           so only misleading evidence is pulled towards uniform). Default.
      kl_form "icmla_eq5": KL[Dir(alpha) || Dir(1)] (ICMLA eq. 5): all evidence is pulled
                           towards uniform.
    alpha comes from a softplus (ICMLA eq. 3), so it can be below 1.
    """
    alpha_0 = alpha.sum(dim=-1, keepdim=True)
    p_hat = alpha / alpha_0
    mse_term = ((y_true - p_hat) ** 2 + p_hat * (1 - p_hat) / (alpha_0 + 1)).sum(dim=-1)
    if kl_form == "sensoy":
        kl_term = dirichlet_kl_to_uniform(y_true + (1.0 - y_true) * alpha)
    elif kl_form == "icmla_eq5":
        kl_term = dirichlet_kl_to_uniform(alpha)
    else:
        raise ValueError(kl_form)
    return (mse_term + lambda_kl * kl_term).mean()


def beta_dirichlet_loss(y_true, params, lambda_kl, kl_form="sensoy"):
    y_bac = torch.stack([y_true[:, 0], 1 - y_true[:, 0]], dim=-1)
    y_fung = torch.stack([y_true[:, 1], 1 - y_true[:, 1]], dim=-1)
    return (dirichlet_loss(y_bac, params[:, :2], lambda_kl, kl_form)
            + dirichlet_loss(y_fung, params[:, 2:], lambda_kl, kl_form))


# KL settings for this run; set from the command line in main().
KL = {"schedule": "icmla", "max": None, "form": "sensoy"}


def kl_schedule(epoch, total_epochs, lambda_0=1e-4, num_classes=2):
    """ICMLA eq. 7. Negligible at K = 2 (max about 5e-5); kept as published."""
    return (lambda_0 / num_classes) * (epoch / max(total_epochs, 1))


def kl_weight(epoch, total_epochs):
    """
    "icmla":  ICMLA eq. 7, (1e-4 / K) * (epoch / N)  -- the published default.
    "sensoy": kl_max * min(1, epoch / 10) -- Sensoy et al.'s annealing (kl_max = 1 is theirs);
              epochs counted from 0, so full weight from the 11th epoch.
    """
    if KL["schedule"] == "icmla":
        return kl_schedule(epoch, total_epochs, num_classes=2)
    return KL["max"] * min(1.0, epoch / 10.0)


def compute_loss(head_type, y_true, output, epoch=0, total_epochs=1):
    if head_type == "sigmoid":
        return masked_bce_loss(y_true, output) if LOSS["kind"] == "bce" else rescaled_mse_loss(y_true, output)
    lam = kl_weight(epoch, total_epochs)
    if head_type == "dirichlet":
        return dirichlet_loss(y_true, output, lam, KL["form"])
    return beta_dirichlet_loss(y_true, output, lam, KL["form"])


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
        if val_loader is not None:
            val = evaluate(model, val_loader, epoch, args.epochs, device)
            improved = val["val_loss"] < best
            if improved:
                best = val["val_loss"]
                torch.save(model.state_dict(), os.path.join(out_dir, "best_val_model.pt"))
        else:  # no validation set: the final-epoch model is the model anyway
            val, improved = {}, None
        row = {"phase": "paper", "epoch": epoch + 1, "train_loss": tl, **val, "best_so_far": improved}
        if model.head_type != "sigmoid":
            row["kl_weight"] = kl_weight(epoch, args.epochs)
        log_epoch(log_path, row)
        tqdm.write(f"  epoch {epoch + 1}/{args.epochs}  train {tl:.4f}"
                   + (f"  val {val['val_loss']:.4f}" if "val_loss" in val else "")
                   + (f"  evidence {val['mean_evidence']:.3f}" if "mean_evidence" in val else "")
                   + (f"  kl_weight {row['kl_weight']:.3g}" if "kl_weight" in row else ""))
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


def per_pathogen_uq(p, out, head, ref):
    """For each output against a per-pathogen reference (NaN = no label): AUROC, accuracy at 0.5, and how well
    uncertainty flags the errors (AUROC of entropy, and of low BD evidence = -(alpha + beta), for wrong vs right).
    Used on the validation set to choose BD's KL setting."""
    res = {}
    for j, name in enumerate(("bacterial", "fungal")):
        m = ~np.isnan(ref[:, j])
        t, pj = ref[m, j], np.clip(p[m, j], 1e-7, 1 - 1e-7)
        r = {"n_images": int(m.sum())}
        if m.any():
            err = (pj > 0.5) != (t == 1)
            ent = -(pj * np.log(pj) + (1 - pj) * np.log(1 - pj))
            r["accuracy"] = float(1 - err.mean())
            r["auroc"] = float(roc_auc_score(t, pj)) if len(np.unique(t)) == 2 else None
            two = len(np.unique(err)) == 2
            r["error_detection_entropy"] = float(roc_auc_score(err, ent)) if two else None
            if head == "beta_dirichlet":
                ev = out[m, 2 * j] + out[m, 2 * j + 1]
                r["mean_evidence"] = float(ev.mean())
                r["error_detection_evidence"] = float(roc_auc_score(err, -ev)) if two else None
        res[name] = r
    return res


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
    KL.update(schedule=args.kl_schedule, max=args.kl_max, form=args.kl_form)
    test_supervised = args.label_type in TEST_TARGETS
    LOSS["kind"] = "bce" if test_supervised else "mse"
    if args.head != "sigmoid":
        print(f"KL regulariser: schedule {args.kl_schedule}"
              + (f", max weight {args.kl_max}" if args.kl_max is not None else "")
              + f", form {args.kl_form}\n")

    # --- Labels and split -----------------------------------------------------
    if args.labels_csv:  # redo: the paper's images, split and labels
        df = load_paper_table(args.labels_csv)
        if args.subset == "culture":  # the culture analysis: images with at least one culture, in every split
            keep = has_culture(df)
            print(f"Culture subset: {int(keep.sum()):,} of {len(df):,} images "
                  f"({df.loc[keep, 'subject'].nunique():,} patients) have a culture.")
            df = df[keep].reset_index(drop=True)
    else:
        df = load_labelled(args.csv, args.img_root, args.group_col)
    if args.n_folds:
        parts = split_cv(df, args.n_folds, args.fold, seed=args.seed, group_col=args.group_col)
        split_desc = f"fold {args.fold} of {args.n_folds}, stratified by study x label"
    elif args.labels_csv:
        parts = split_paper(df, seed=args.seed, group_col=args.group_col, val_share=args.val_share)
        split_desc = "the paper's split" + (" + validation file" if (df["split"] == "val").any() else
                                            (f" (validation: {args.val_share:.0%} of training subjects)"
                                             if args.val_share else " (no validation set)"))
    else:
        parts = split_by_participant(df, seed=args.seed, group_col=args.group_col)
        split_desc = "single split"
    y_all = targets(df, args.label_type, args.group_col)
    no_target = np.isnan(y_all).all(axis=1)  # test-supervised: neither test done
    tr, va = parts["train"], parts["val"]
    if not test_supervised and np.isnan(y_all).any():  # rescaled MSE / evidential losses need both targets
        raise SystemExit(f"{int(np.isnan(y_all).any(axis=1).sum())} images lack a {args.label_type} target for one "
                         "pathogen (e.g. one culture only); this loss needs both.")
    if test_supervised:  # only images with at least one of the two tests can train
        tr, va = tr[~no_target[tr]], va[~no_target[va]]
    if args.smoke:  # a small random subset of each part
        rng = np.random.default_rng(args.seed)
        pick = lambda a, n: np.sort(rng.permutation(a)[:n])
        tr, va = pick(tr, 48), pick(va, 16)
        parts = {**parts, "cal": pick(parts["cal"], 8), "eval": pick(parts["eval"], 24)}
    if args.exclude_followups_train:
        tr = tr[~df["is_followup"].to_numpy()[tr]]
    test = np.concatenate([parts["cal"], parts["eval"]])
    part_flag = np.r_[np.zeros(len(parts["cal"]), int), np.ones(len(parts["eval"]), int)]

    print(label_summary(df, args.group_col).to_string(), "\n")
    print(f"Split ({split_desc}, grouped by {args.group_col}): train {len(tr):,}, val {len(va):,}, "
          f"test {len(test):,} (cal {len(parts['cal']):,} / eval {len(parts['eval']):,})")
    if test_supervised:
        cols = TEST_TARGETS[args.label_type]
        print(f"Test-supervised ({args.label_type}: {cols[0]}, {cols[1]}): training uses images with at "
              f"least one of the two tests; {int((~no_target[parts['train']]).sum()):,} of "
              f"{len(parts['train']):,} training images. Every test image is predicted.")
    print()

    # --- Images must all exist before anything trains -------------------------
    used = np.concatenate([tr, va, test])
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
    stage = model.set_trainable_last_stage(args.trainable_stages) if args.recipe == "paper" else None
    n_train_params = count_trainable(model)
    print(f"Recipe {args.recipe}: trainable parameters {n_train_params:,}"
          + (f" (backbone.blocks[{stage}:] + conv_head + head; backbone BatchNorm frozen)"
             if stage is not None else " (Phase 1: head only)"))

    # --- Record everything about the run --------------------------------------
    config = {**vars(args), "git": git_info(), "trainable_params": n_train_params,
              "first_trainable_stage": stage, "norm_mean": list(mean), "norm_std": list(std),
              "loss": "binary cross-entropy over tests done" if test_supervised else
                      ("rescaled MSE" if args.head == "sigmoid" else "evidential"),
              "split": split_desc,
              "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(test)),
              "started": datetime.now().isoformat(timespec="seconds"),
              "torch": torch.__version__, "timm": timm.__version__}
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    split_tab = pd.DataFrame({"csv_row": df["csv_row"].to_numpy()})
    split_tab["part"] = "unused"
    for k in ("cal", "eval"):
        split_tab.loc[parts[k], "part"] = k
    for k, used_idx in (("train", tr), ("val", va)):
        split_tab.loc[parts[k], "part"] = f"{k}_not_used"  # no test (baselines), follow-up, smoke
        split_tab.loc[used_idx, "part"] = k
    split_tab.to_csv(os.path.join(out_dir, "split.csv"), index=False)
    label_summary(df, args.group_col).to_csv(os.path.join(out_dir, "label_summary.csv"))

    # --- Loaders ----------------------------------------------------------------
    paths = df["local_path"].to_numpy()
    g = torch.Generator().manual_seed(args.seed)
    nw, pin = args.num_workers, device.type == "cuda"

    def loader(idx, tf, batch, shuffle):
        ds = KeratitisDataset(paths[idx], y_all[idx], tf)
        drop_last = shuffle and len(ds) % batch == 1  # BatchNorm cannot train on a batch of 1
        return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=nw, pin_memory=pin,
                          worker_init_fn=worker_init, generator=g, drop_last=drop_last)

    train_loader = loader(tr, train_tf, args.batch, True)
    val_loader = loader(va, eval_tf, args.batch, False) if len(va) else None

    # --- Train ------------------------------------------------------------------
    print(f"\nTraining {args.head} / {args.label_type} / {args.recipe} ({config['loss']}): {args.epochs} epochs, "
          f"batch {args.batch} x accum {args.accum}, lr {args.lr}, device {device}\n")
    if args.recipe == "paper":
        model = train_paper(model, train_loader, val_loader, args, device, out_dir, keras)
    else:
        p2 = loader(tr, train_tf, args.batch_phase2 or args.batch, True)
        model = train_legacy(model, (train_loader, p2, val_loader), args, device, out_dir)

    # --- Validation predictions (no identifiers): to choose BD's KL setting -------------------
    if len(va) and args.labels_csv:
        vout, vp = predict(model, paths[va], eval_tf, args.batch, device, nw)
        vsub = df.iloc[va]
        vref = {"lca_hard_labels": vsub[["hard_bac", "hard_fung"]].to_numpy(dtype=float),
                "culture": test_results(df, args.group_col)[va][:, [TESTS.index(c) for c in CULTURES]].astype(float)}
        vmad = np.array([STUDY_SHORT.get(x, x) for x in vsub["study"]]) == "MADURAI"
        vmetrics = {"n_images": int(len(va)), "n_patients": int(vsub[args.group_col].nunique()),
                    "head": args.head, "label_type": args.label_type, "subset": args.subset,
                    "kl_schedule": args.kl_schedule, "kl_max": args.kl_max, "kl_form": args.kl_form}
        for rname, ref in vref.items():
            vmetrics[rname] = {"all": per_pathogen_uq(vp, vout, args.head, ref),
                               "MADURAI": per_pathogen_uq(vp[vmad], vout[vmad], args.head, ref[vmad])}
        with open(os.path.join(out_dir, "val_metrics.json"), "w") as f:
            json.dump(vmetrics, f, indent=2)
        np.savez(os.path.join(out_dir, "val_predictions.npz"), output=vout, p_hat=vp,
                 lca_hard_labels=vref["lca_hard_labels"], culture=vref["culture"],
                 study=np.array([STUDY_SHORT.get(x, x) for x in vsub["study"]]),
                 participant=participant_codes(df, args.group_col)[va], head=args.head)
        print("  validation: " + "; ".join(
            f"{r} {k} AUROC {vmetrics[r]['all'][k].get('auroc')}" for r in vref for k in ("bacterial", "fungal")))

    # --- Test predictions (no identifiers) ---------------------------------------
    out, p = predict(model, paths[test], eval_tf, args.batch, device, nw)
    score = (p[:, 0] + 1.0 - p[:, 1]) / 2.0  # one bacterial probability from the two outputs
    sub = df.iloc[test]
    extra = {}
    if args.labels_csv:  # per-pathogen labels for the paper's own scoring
        extra = {k: sub[k].to_numpy(dtype=float) for k in ("hard_bac", "hard_fung", "paper_soft_bac", "paper_soft_fung")}
    np.savez(os.path.join(out_dir, "test_predictions.npz"), **extra,
             head=args.head, label_type=args.label_type, recipe=args.recipe,
             output=out, p_hat=p, p_bac_binary=score,
             y_bac=sub["y_bac"].to_numpy(), soft_bac=sub["soft_bac"].to_numpy(),
             soft_fung=sub["soft_fung"].to_numpy(), has_posterior=sub["has_posterior"].to_numpy(),
             is_followup=sub["is_followup"].to_numpy(), visit=sub["visit"].to_numpy().astype(str),
             study=np.array([STUDY_SHORT.get(s, s) for s in sub["study"]]),
             csv_row=sub["csv_row"].to_numpy(),
             participant=participant_codes(df, args.group_col)[test], part=part_flag,
             fold=-1 if args.fold is None else args.fold, n_folds=args.n_folds,
             train_target=y_all[test], test_results=test_results(df, args.group_col)[test],
             test_cols=np.array(TESTS))

    y = sub["y_bac"].to_numpy()
    mad = np.array([STUDY_SHORT.get(s, s) for s in sub["study"]]) == "MADURAI"
    ev = part_flag == 1
    metrics = {"reference": ("two-class posterior > 0.5 (binary, bacterial vs fungal)" if args.labels_csv
                             else "LCA bin (binary, bacterial vs fungal)"),
               "score": "(p_bac + 1 - p_fung) / 2",
               "test_all": binary_metrics(y, score, p),
               "eval": binary_metrics(y[ev], score[ev], p[ev]),
               "eval_MADURAI": binary_metrics(y[ev & mad], score[ev & mad], p[ev & mad]),
               "eval_other_studies": binary_metrics(y[ev & ~mad], score[ev & ~mad], p[ev & ~mad])}
    if test_supervised:  # each output against its own test, where the test was done
        own = {}
        for j, col in enumerate(TEST_TARGETS[args.label_type]):
            t = y_all[test][:, j]
            m = ~np.isnan(t)
            own[col] = {"n_images": int(m.sum()), "positive_share": float(np.mean(t[m])) if m.any() else None,
                        "auroc": float(roc_auc_score(t[m], p[m, j])) if len(np.unique(t[m])) == 2 else None}
        metrics["own_tests"] = own
    if args.labels_csv:  # the paper's scoring: each output against Travis's hard label for that pathogen
        def per_pathogen(m):
            r = {"n_images": int(m.sum())}
            for j, (name, h) in enumerate((("bacterial", "hard_bac"), ("fungal", "hard_fung"))):
                t = sub[h].to_numpy()[m]
                r[f"{name}_positive_share"] = float(t.mean()) if m.any() else None
                r[f"{name}_auroc"] = float(roc_auc_score(t, p[m, j])) if len(np.unique(t)) == 2 else None
                r[f"{name}_accuracy"] = float(np.mean((p[m, j] > 0.5) == (t == 1))) if m.any() else None
            return r
        metrics["per_pathogen_vs_hard_labels"] = {"test_all": per_pathogen(np.ones(len(test), bool)),
                                                  "test_MADURAI": per_pathogen(mad),
                                                  "test_other_studies": per_pathogen(~mad)}
        cul = test_results(df, args.group_col)[test][:, [TESTS.index(c) for c in CULTURES]]

        def per_pathogen_culture(m):
            r = {}
            for j, name in enumerate(("bacterial", "fungal")):
                k = m & ~np.isnan(cul[:, j])
                t = cul[k, j]
                r[f"{name}_n_images"] = int(k.sum())
                r[f"{name}_positive_share"] = float(t.mean()) if k.any() else None
                r[f"{name}_auroc"] = float(roc_auc_score(t, p[k, j])) if len(np.unique(t)) == 2 else None
                r[f"{name}_accuracy"] = float(np.mean((p[k, j] > 0.5) == (t == 1))) if k.any() else None
            return r
        # the culture analysis's reference: blood agar for the bacterial output, PDA for the fungal one,
        # on the test images where that culture was done (all test images when --subset culture)
        metrics["per_pathogen_vs_culture"] = {"test_all": per_pathogen_culture(np.ones(len(test), bool)),
                                              "test_MADURAI": per_pathogen_culture(mad),
                                              "test_other_studies": per_pathogen_culture(~mad),
                                              "eval": per_pathogen_culture(ev)}
        metrics["per_pathogen_vs_hard_labels"]["eval"] = per_pathogen(ev)
        pc = metrics["per_pathogen_vs_culture"]["test_all"]
        print(f"  per pathogen vs culture, test images with a culture: bacterial AUROC {pc['bacterial_auroc']}, "
              f"fungal AUROC {pc['fungal_auroc']}")
        pp = metrics["per_pathogen_vs_hard_labels"]["test_all"]
        print(f"  per pathogen, all test images (paper's scoring): bacterial AUROC {pp['bacterial_auroc']}, "
              f"fungal AUROC {pp['fungal_auroc']}")
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
