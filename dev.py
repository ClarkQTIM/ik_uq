"""
Latent Supervision + Dirichlet Classifier
==========================================
Combines:
  - Soft probabilistic labels from LCA + Bayesian posterior estimation
    (from the R script / latent supervision paper)
  - Dirichlet output head + annealed KL loss
    (from the ICMLA Dirichlet classifiers paper)

Pipeline overview:
    Step 1: LCA on diagnostic test results → conditional probabilities
    Step 2: Bayesian posterior → per-sample soft labels (e.g. [0.73, 0.27])
    Step 3: Train image classifier with Dirichlet head on those soft labels

Dependencies (to install):
    pip install torch torchvision pandas numpy scikit-learn
    # For LCA you'd either:
    #   (a) run the R script first and export the posteriors as a CSV, OR
    #   (b) use a Python LCA library e.g. `stepmix` (pip install stepmix)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import pandas as pd


# =============================================================================
# STEP 1 & 2: LCA + BAYESIAN POSTERIOR
# (Produces soft labels — run once, save to CSV, load here)
# =============================================================================

# NOTE: The cleanest approach is to run the R script's Steps 1 & 2 as-is,
# then export ulcer_dataset with the posterior columns to a CSV.
# Load it here like:
#
#   df = pd.read_csv("ulcer_posteriors.csv")
#   # df has columns: image_path, bac_post_prob, fung_post_prob
#
# Alternatively, use the `stepmix` Python package to replicate LCA in Python.
# stepmix docs: https://stepmix.readthedocs.io
# Example (not fully implemented — placeholder):
#
#   from stepmix.stepmix import StepMix
#   model = StepMix(n_components=2, measurement="binary", random_state=42)
#   model.fit(test_results_df)
#   soft_labels = model.predict_proba(test_results_df)  # shape: (N, 2)
#
# Then apply Bayes update per sample using the conditional probs,
# same logic as update_posterior_from_lca_vectorized() in the R script.


def load_soft_labels(csv_path):
    """
    Load precomputed soft labels from CSV.
    Expected columns: image_path, bac_post_prob, fung_post_prob
    Returns: list of (image_path, label_tensor) tuples
    """
    df = pd.read_csv(csv_path)
    samples = []
    for _, row in df.iterrows():
        label = torch.tensor(
            [row["bac_post_prob"], row["fung_post_prob"]],
            dtype=torch.float32
        )
        samples.append((row["image_path"], label))
    return samples


# =============================================================================
# DATASET
# =============================================================================

class SoftLabelImageDataset(Dataset):
    """
    Image dataset with soft (probabilistic) labels.
    Labels are 2D tensors like [0.73, 0.27] rather than hard integers.
    """

    def __init__(self, samples, transform=None):
        # samples: list of (image_path, soft_label_tensor)
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# Standard augmentation pipeline (mirrors the R script's Keras augmentation)
train_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(contrast=0.1),
    transforms.Resize((600, 600)),   # EfficientNet-B7 input size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean/std
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# =============================================================================
# STEP 3A: MODEL — EfficientNet backbone + Dirichlet head
# =============================================================================

class DirichletEfficientNet(nn.Module):
    """
    EfficientNet-B7 backbone with a Dirichlet output head.

    Key difference from the R script:
      - Final activation is softplus (not sigmoid)
      - Outputs are Dirichlet concentration parameters α = [α1, α2]
      - Class probabilities: p̂_k = α_k / sum(α)
      - Total evidence: α0 = sum(α) — higher = more confident

    NOTE: Could also use MobileNetV3 for the trachoma use case,
    just swap the backbone. Same head and loss apply.
    """

    def __init__(self, num_classes=2, dropout=0.3, freeze_backbone=True):
        super().__init__()

        # Load pretrained EfficientNet-B7
        # NOTE: torchvision doesn't have B7 natively — options:
        #   (a) pip install timm; then use timm.create_model("efficientnet_b7", pretrained=True)
        #   (b) Use EfficientNet-B0 from torchvision as a stand-in for development
        #
        # Using timm (recommended):
        #   import timm
        #   self.backbone = timm.create_model("efficientnet_b7", pretrained=True, num_classes=0)
        #   backbone_out_features = self.backbone.num_features
        #
        # Using torchvision B0 as placeholder:
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        backbone_out_features = backbone.classifier[1].in_features
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # remove classifier

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Dirichlet head (mirrors the R script's dense layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(backbone_out_features),
            nn.Linear(backbone_out_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
            nn.Softplus()   # <-- KEY CHANGE: ensures α_k > 0 always
        )

    def forward(self, x):
        features = self.backbone(x)
        alpha = self.head(features)     # Dirichlet concentration params
        return alpha

    def predict_proba(self, x):
        """Convenience method: returns class probabilities and total evidence."""
        alpha = self.forward(x)
        alpha_0 = alpha.sum(dim=-1, keepdim=True)
        p_hat = alpha / alpha_0         # Expected class probabilities
        return p_hat, alpha_0           # Also return total evidence


# =============================================================================
# STEP 3B: LOSS — Dirichlet MSE loss with annealed KL
# =============================================================================

def dirichlet_loss(y_true, alpha, lambda_kl, num_classes=2):
    """
    Dirichlet loss from Sensoy et al. (2018), equation 5 in the ICMLA paper.

    Args:
        y_true:      Soft label tensor, shape (batch, num_classes)
                     e.g. [[0.73, 0.27], [0.12, 0.88], ...]
                     NOTE: Unlike the paper's one-hot labels, these are
                     already probabilistic — uncertainty enters from both
                     the labels AND the Dirichlet output. This is the novel
                     combination not present in either source paper.
        alpha:       Dirichlet concentration params from model, shape (batch, num_classes)
        lambda_kl:   Annealed KL weight (scalar, increases over training)
        num_classes: K (2 here)

    Returns:
        Scalar loss value
    """

    alpha_0 = alpha.sum(dim=-1, keepdim=True)       # Total evidence, shape (batch, 1)
    p_hat = alpha / alpha_0                          # Expected class probs

    # --- MSE term ---
    # Squared error between soft label and predicted probability,
    # plus a variance term that penalizes overconfidence
    mse_term = ((y_true - p_hat) ** 2 +
                p_hat * (1 - p_hat) / (alpha_0 + 1)).sum(dim=-1)

    # --- KL divergence term ---
    # Regularize toward uniform Dirichlet prior Dir(1, 1, ..., 1)
    # Use alpha_tilde: zero out evidence for the "true" class
    # NOTE: With soft labels, y_true is not one-hot, so alpha_tilde
    # is a weighted blend rather than a hard zeroing. This is an open
    # design choice — one option is to threshold, another is to use as-is.
    alpha_tilde = y_true + (1 - y_true) * alpha

    alpha_tilde_0 = alpha_tilde.sum(dim=-1, keepdim=True).squeeze()

    kl_term = (
        torch.lgamma(alpha_tilde_0)
        - torch.lgamma(torch.tensor(float(num_classes)))
        - torch.lgamma(alpha_tilde).sum(dim=-1)
        + ((alpha_tilde - 1) * (
            torch.digamma(alpha_tilde) -
            torch.digamma(alpha_tilde_0.unsqueeze(-1))
        )).sum(dim=-1)
    )

    loss = (mse_term + lambda_kl * kl_term).mean()
    return loss


def kl_annealing_schedule(epoch, total_epochs, lambda_0=1e-4, num_classes=2):
    """
    Dataset-agnostic KL annealing from the ICMLA paper, equation 7.
    Linearly increases KL weight from 0 to lambda_0/K over training.
    Critical for stability with high class counts; minor effect for K=2.
    """
    return (lambda_0 / num_classes) * (epoch / total_epochs)


# =============================================================================
# STEP 3C: TRAINING LOOP
# =============================================================================

def train(model, train_loader, val_loader,
          num_epochs=30, lr=1e-5, num_classes=2):
    """
    Custom training loop needed (vs. model.fit) because:
      - KL weight must update each epoch (can't be done in a static loss fn)
      - We want to track total evidence as an extra metric
    """

    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    # NOTE: The R script uses RMSprop for EfficientNet, Adam for MobileNet.
    # Match that choice based on which backbone you use.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(num_epochs):

        # Compute annealed KL weight for this epoch
        lambda_kl = kl_annealing_schedule(epoch, num_epochs, num_classes=num_classes)

        # --- Training phase ---
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            alpha = model(images)
            loss = dirichlet_loss(labels, alpha, lambda_kl, num_classes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        total_evidence_sum = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                alpha = model(images)
                loss = dirichlet_loss(labels, alpha, lambda_kl, num_classes)
                val_loss += loss.item()

                # Accuracy (argmax of soft label vs argmax of predicted probs)
                p_hat = alpha / alpha.sum(dim=-1, keepdim=True)
                pred = p_hat.argmax(dim=-1)
                true = labels.argmax(dim=-1)
                val_correct += (pred == true).sum().item()
                val_total += labels.size(0)

                # Track mean total evidence as a confidence metric
                total_evidence_sum += alpha.sum(dim=-1).mean().item()

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss/len(train_loader):.4f} | "
            f"Val Loss: {val_loss/len(val_loader):.4f} | "
            f"Val Acc: {val_correct/val_total:.4f} | "
            f"Mean Evidence: {total_evidence_sum/len(val_loader):.2f} | "
            f"λ_KL: {lambda_kl:.6f}"
        )

    return model


# =============================================================================
# STEP 3D: INFERENCE
# =============================================================================

def predict(model, image_tensor, threshold=None):
    """
    At inference time you get three things:
      - p_hat:    class probabilities (what you show the clinician)
      - alpha_0:  total evidence (how confident the model is)
      - flag:     whether to refer for human review

    threshold: if alpha_0 < threshold, flag for human review.
               Set empirically on a validation set.
               This is the "selective abstention" from the ICMLA paper.
    """
    model.eval()
    with torch.no_grad():
        alpha = model(image_tensor.unsqueeze(0))
        alpha_0 = alpha.sum(dim=-1)
        p_hat = alpha / alpha_0.unsqueeze(-1)

    result = {
        "bac_probability": p_hat[0, 0].item(),
        "fung_probability": p_hat[0, 1].item(),
        "total_evidence": alpha_0[0].item(),
        "refer_for_review": (threshold is not None and
                             alpha_0[0].item() < threshold)
    }
    return result


# =============================================================================
# MAIN — Wire it all together
# =============================================================================

if __name__ == "__main__":

    # --- Load soft labels (exported from R script Steps 1 & 2) ---
    # TODO: replace with actual CSV path
    # samples = load_soft_labels("ulcer_posteriors.csv")

    # --- Placeholder for development/testing ---
    # Generates random fake data so the code structure can be tested
    # without real images or the R pipeline
    samples = [("fake_path.jpg", torch.tensor([0.7, 0.3])) for _ in range(100)]

    # --- Datasets and loaders ---
    # TODO: proper train/val/test split
    dataset = SoftLabelImageDataset(samples, transform=val_transforms)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # --- Model ---
    model = DirichletEfficientNet(num_classes=2, dropout=0.3, freeze_backbone=True)

    # --- Train ---
    # TODO: unfreeze deeper layers after initial training (mirrors R script's
    # unfreeze_weak_model logic) — implement as a second training phase
    # with a lower LR and freeze_backbone=False
    model = train(model, loader, loader, num_epochs=5)

    # --- Example inference ---
    dummy_image = torch.randn(3, 600, 600)
    result = predict(model, dummy_image, threshold=2.0)
    print("\nExample prediction:", result)