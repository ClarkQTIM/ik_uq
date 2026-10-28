"""
Scoring that accounts for the study confound. Every study except MADURAI enrolled one diagnosis, so within a trial
each output has only positives or only negatives and no AUROC exists; a pooled AUROC is then mostly built from
pairs of patients in different studies, which a model can win by recognising the study.

  pair_auroc        the pooled AUROC split into its within-study pairs and its between-study pairs
  study_only        the AUROC of a 'model' that only knows the study (its training-set positive rate)
  per_study         single-class measures that work inside a trial: share called correctly, mean prediction,
                    and the AUROC where the study has both classes
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def _concordance(y, s):
    """Concordant positive-negative pairs (ties count half) and the number of pairs."""
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return 0.0, 0
    return float(roc_auc_score(y, s)) * pos * neg, pos * neg


def pair_auroc(y, s, groups):
    """Pooled AUROC, and the AUROC over within-group pairs only and over between-group pairs only."""
    y, s, groups = np.asarray(y, float), np.asarray(s, float), np.asarray(groups)
    m = ~np.isnan(y)
    y, s, groups = y[m], s[m], groups[m]
    c_all, n_all = _concordance(y, s)
    c_in = n_in = 0
    for g in np.unique(groups):
        k = groups == g
        c, n = _concordance(y[k], s[k])
        c_in += c
        n_in += n
    n_between = n_all - n_in
    return {"auroc_pooled": c_all / n_all if n_all else None,
            "auroc_within_study_pairs": c_in / n_in if n_in else None,
            "auroc_between_study_pairs": (c_all - c_in) / n_between if n_between else None,
            "share_of_pairs_within_study": n_in / n_all if n_all else None}


def study_only(y_train, study_train, y_eval, study_eval):
    """Pair-split AUROC of a score that is just each study's positive rate in training (NaN labels ignored)."""
    y_train = np.asarray(y_train, float)
    rate = {}
    for st in np.unique(study_train):
        v = y_train[(np.asarray(study_train) == st) & ~np.isnan(y_train)]
        rate[st] = float(v.mean()) if len(v) else 0.5
    score = np.array([rate.get(st, 0.5) for st in study_eval])
    return pair_auroc(y_eval, score, study_eval)


def per_study(y, p, study):
    """Per study: images, positive share, mean prediction, share called correctly at 0.5, AUROC if defined."""
    y, p, study = np.asarray(y, float), np.asarray(p, float), np.asarray(study)
    out = {}
    for st in np.unique(study):
        k = (study == st) & ~np.isnan(y)
        t, q = y[k], p[k]
        r = {"n_images": int(k.sum())}
        if k.any():
            r.update(positive_share=float(t.mean()), mean_prediction=float(q.mean()),
                     share_called_correctly=float(np.mean((q > 0.5) == (t == 1))),
                     auroc=float(roc_auc_score(t, q)) if len(np.unique(t)) == 2 else None)
        out[str(st)] = r
    return out


def same_label_study_gap(hard, p, study):
    """Does the model's output separate studies when the diagnosis is the same? Among fungal-only images: AUROC of
    the fungal output for trial vs MADURAI; among bacterial-only images: the same with the bacterial output. Near 0.5
    means the output carries no study signal at a fixed diagnosis; far from 0.5 means it does (shortcut or a real
    difference in how the ulcers look)."""
    hard, p, study = np.asarray(hard, float), np.asarray(p, float), np.asarray(study)
    out = {}
    for name, j, sel in (("fungal-only images, fungal output", 1, (hard[:, 1] == 1) & (hard[:, 0] == 0)),
                         ("bacterial-only images, bacterial output", 0, (hard[:, 0] == 1) & (hard[:, 1] == 0))):
        trial = study[sel] != "MADURAI"
        r = {"n_MADURAI": int((~trial).sum()), "n_trials": int(trial.sum())}
        if trial.any() and (~trial).any():
            q = p[sel, j]
            r.update(mean_output_MADURAI=float(q[~trial].mean()), mean_output_trials=float(q[trial].mean()),
                     auroc_trials_vs_MADURAI=float(roc_auc_score(trial, q)))
        out[name] = r
    return out
