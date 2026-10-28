"""
Can a model tell the study from the photo? If it can, every model trained on the pooled data has a shortcut: in the
trials the label is fixed by the study, so recognising the study scores well on a pooled AUROC.

Takes image features from a backbone (ImageNet-pretrained by default, or the backbone of a trained run with
--checkpoint), then fits a linear probe (standardised features, logistic regression) with 5-fold cross-validation
grouped by patient, on these tasks:
  all studies                 which of the eight studies (study and diagnosis are tangled here)
  fungal-only images          which study, among images whose label is fungal only: same diagnosis, different study
  fungal-only: MADURAI vs trials   can it tell a MADURAI fungal ulcer from a trial fungal ulcer?
  bacterial-only images       MADURAI vs SCUT among bacterial-only images
  site within a trial         which hospital, within MUTT1, MUTT2 and SCUT (same trial, same diagnosis)
Balanced accuracy near chance (1 / classes) and AUROC near 0.5 mean the photos carry no study signal beyond the
diagnosis; near 1 means the study is recognisable and the shortcut is open. Uses the images on our server.
Aggregates only.

    python src/study_probe.py --labels results/labels/paper_labels.csv --out results/study_probe \
        [--backbone tf_efficientnet_b7.aa_in1k] [--checkpoint <run folder>/model.pt]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eda_paper_data import md  # noqa: E402
from labels import load_paper_table  # noqa: E402

MIN_IMAGES = 20  # a class needs this many images to be probed


def features(df, a, device):
    import timm
    from timm.data import resolve_data_config
    from model import EfficientNetB7, KeratitisDataset, eval_transform
    head, backbone = "sigmoid", a.backbone
    if a.checkpoint:
        cfg = json.load(open(os.path.join(os.path.dirname(a.checkpoint), "config.json")))
        head, backbone = cfg["head"], cfg["backbone"]
    model = EfficientNetB7(head_type=head, pretrained=not (a.checkpoint or a.no_pretrained), backbone_name=backbone)
    if a.checkpoint:
        model.load_state_dict(torch.load(a.checkpoint, map_location="cpu"))
    bb = model.backbone.to(device).eval()
    cfg = resolve_data_config({}, model=timm.create_model(backbone, pretrained=False, num_classes=0))
    ds = KeratitisDataset(df["local_path"].to_numpy(), np.zeros((len(df), 2)),
                          eval_transform(a.img_size, cfg["mean"], cfg["std"]))
    out = []
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch_size=a.batch, num_workers=a.num_workers, pin_memory=device.type == "cuda"):
            out.append(bb(x.to(device)).float().cpu().numpy())
    return np.concatenate(out), backbone


def probe(X, y, groups, seed, C):
    """Out-of-fold class probabilities from a grouped, stratified 5-fold linear probe."""
    classes = np.unique(y)
    proba = np.zeros((len(y), len(classes)))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        cols = [list(classes).index(c) for c in clf.classes_]
        proba[np.ix_(te, cols)] = clf.predict_proba(X[te])
    return classes, proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="results/labels/paper_labels.csv")
    ap.add_argument("--out", default="results/study_probe")
    ap.add_argument("--backbone", default="tf_efficientnet_b7.aa_in1k")
    ap.add_argument("--checkpoint", default=None, help="model.pt of a trained run: probe its fine-tuned backbone")
    ap.add_argument("--no_pretrained", action="store_true", help="testing only")
    ap.add_argument("--img_size", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--C", type=float, default=0.1, help="inverse L2 strength of the probe")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    df = load_paper_table(a.labels, verbose=False).reset_index(drop=True)
    cache = os.path.join(a.out, "features_" + ("checkpoint" if a.checkpoint else a.backbone.replace("/", "_")) + ".npz")
    if os.path.exists(cache) and not a.checkpoint:
        z = np.load(cache)
        X, used = z["X"], str(z["backbone"])
        assert len(X) == len(df), "cached features do not match the label table; delete the cache"
    else:
        print(f"Extracting features for {len(df):,} images on {device} ...")
        X, used = features(df, a, device)
        np.savez(cache, X=X, backbone=used)
    fung_only = (df.hard_fung == 1) & (df.hard_bac == 0)
    bac_only = (df.hard_bac == 1) & (df.hard_fung == 0)

    tasks = [("all studies", np.ones(len(df), bool), df.study),
             ("fungal-only images, by study", fung_only.to_numpy(), df.study),
             ("fungal-only images: MADURAI vs trials", fung_only.to_numpy(),
              np.where(df.study == "MADURAI", "MADURAI", "trials")),
             ("bacterial-only images, by study", bac_only.to_numpy(), df.study)]
    for st in ("MUTT1", "MUTT2", "SCUT"):
        tasks.append((f"site within {st}", (df.study == st).to_numpy(), df.subfolder.fillna("").astype(str)))

    rows, recall_rows, conf = [], [], {}
    for name, mask, lab in tasks:
        lab = pd.Series(np.asarray(lab), index=df.index)[mask]
        keep = lab.map(lab.value_counts()) >= MIN_IMAGES
        lab = lab[keep & (lab != "")]
        if lab.nunique() < 2:
            rows.append({"task": name, "note": "fewer than two classes with enough images"})
            continue
        idx = lab.index.to_numpy()
        y, g = lab.to_numpy(), df.subject.to_numpy()[idx]
        classes, proba = probe(X[idx], y, g, a.seed, a.C)
        pred = classes[proba.argmax(1)]
        k = len(classes)
        auc = (roc_auc_score(y == classes[1], proba[:, 1]) if k == 2
               else roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes))
        rows.append({"task": name, "images (patients)": f"{len(idx):,} ({len(np.unique(g)):,})", "classes": k,
                     "chance balanced accuracy": 1 / k, "balanced accuracy": balanced_accuracy_score(y, pred),
                     "AUROC (macro, one vs rest)": auc})
        for c in classes:
            m = y == c
            recall_rows.append({"task": name, "class": c, "images": int(m.sum()), "recall": float((pred[m] == c).mean())})
        conf[name] = pd.DataFrame(confusion_matrix(y, pred, labels=classes), index=classes, columns=classes)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, "study_probe.csv"), index=False)
    rec = pd.DataFrame(recall_rows)
    rec.to_csv(os.path.join(a.out, "study_probe_recall.csv"), index=False)
    src = f"fine-tuned backbone of {a.checkpoint}" if a.checkpoint else f"{used}, ImageNet-pretrained, not fine-tuned"
    with open(os.path.join(a.out, "study_probe.md"), "w") as f:
        f.write("# Can a model tell the study from the photo?\n\nAggregates only. Features: " + src + ". Linear probe, "
                "5-fold cross-validation grouped by patient; images on our server; classes with at least "
                f"{MIN_IMAGES} images.\n\nBalanced accuracy near chance and AUROC near 0.5: no study signal beyond the "
                "diagnosis. Near 1: the study is recognisable from the photo, so a shortcut is open to every model.\n\n"
                "## Tasks\n\n" + md(res) + "\n\n## Recall by class\n\n" + md(rec)
                + "".join(f"\n\n## Confusion matrix: {n} (rows: true; columns: predicted)\n\n"
                          + md(c.reset_index().rename(columns={"index": "true"}), ".0f") for n, c in conf.items()) + "\n")
    print(res.to_string(index=False))
    print(f"\nReport: {a.out}/study_probe.md")


if __name__ == "__main__":
    main()
