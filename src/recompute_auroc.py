"""
Recompute AUROC on eval subset only (n=639).
Uses existing inference caches — no GPU needed.

This ensures AUROC, CPSL, BD evidence, and all other metrics
are computed on the exact same 639 images.

Run:
    python src/recompute_auroc.py \
        --cache_dir runs/beta_dirichlet_2026-07-17_13-31-38/uncertainty_analysis/ \
        --csv /persist/QTIM/Active/.../corneal_ulcer_study_data_7_24_2025_lambda.csv
"""

import os, json, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--csv", type=str,
        default="/persist/QTIM/Active/26-0096/Corneal_Ulcer_Image_Sets/Data/"
                "corneal_ulcer_study_data_7_24_2025_lambda.csv")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def get_eval_idx(csv_path, seed=42):
    """Reproduce exact eval_idx — same as all analysis scripts."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["file_path", "subject"])
    df = df[(df["fung_LCA_bin"] != -1) | (df["bac_LCA_bin"] != -1)].reset_index(drop=True)
    subj = df["subject"].values

    # 70/30 split
    gss = GroupShuffleSplit(1, test_size=0.30, random_state=seed)
    _, ti = next(gss.split(df, groups=subj))
    df_t = df.iloc[ti].reset_index(drop=True)

    # 50/50 of the 30% → test set
    gss2 = GroupShuffleSplit(1, test_size=0.50, random_state=seed)
    _, ti2 = next(gss2.split(df_t, groups=df_t["subject"].values))
    df_test = df_t.iloc[ti2].reset_index(drop=True)

    # 20/80 cal/eval split
    gss3 = GroupShuffleSplit(1, test_size=0.80, random_state=seed)
    cal_idx, eval_idx = next(gss3.split(df_test, groups=df_test["subject"].values))

    print(f"Test set: {len(df_test):,}  Cal: {len(cal_idx):,}  Eval: {len(eval_idx):,}")
    return eval_idx, df_test


def auroc_on_eval(probs, labels, eval_idx):
    """Compute AUROC for BAC and FUNG on eval subset only."""
    p_eval = probs[eval_idx]
    l_eval = labels[eval_idx]
    results = {}
    for i, name in enumerate(["bac", "fung"]):
        y_true = (l_eval[:, i] > 0.5).astype(int)
        y_score = p_eval[:, i]
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            print(f"  {name}: skipped (no class variation)")
            results[name] = None
        else:
            auc = roc_auc_score(y_true, y_score)
            results[name] = float(auc)
            print(f"  {name}: AUROC = {auc:.4f}  (n={len(y_true)}, "
                  f"pos={y_true.sum()}, neg={(~y_true.astype(bool)).sum()})")
    return results


def main():
    args = get_args()
    eval_idx, df_test = get_eval_idx(args.csv, args.seed)

    # Load caches
    sig_cache = np.load(os.path.join(args.cache_dir, "cache_sig.npz"))
    bd_cache  = np.load(os.path.join(args.cache_dir, "cache_bd.npz"))

    sig_probs  = sig_cache["probs"]    # (785, 2)
    sig_labels = sig_cache["labels"]   # (785, 2)
    bd_params  = bd_cache["params"]    # (785, 4)

    # BD probabilities
    a_bac  = bd_params[:, 0]; b_bac  = bd_params[:, 1]
    a_fung = bd_params[:, 2]; b_fung = bd_params[:, 3]
    bd_probs = np.stack([a_bac / (a_bac + b_bac),
                         a_fung / (a_fung + b_fung)], axis=1)

    print("\n" + "="*55)
    print("  AUROC ON EVAL SUBSET ONLY (n=639)")
    print("="*55)

    print("\nCell A — Sigmoid + Soft labels:")
    auroc_sig = auroc_on_eval(sig_probs, sig_labels, eval_idx)

    print("\nCell B — BD + Soft labels:")
    auroc_bd  = auroc_on_eval(bd_probs, sig_labels, eval_idx)

    results = {
        "eval_n": int(len(eval_idx)),
        "Cell_A_Sigmoid_Soft": auroc_sig,
        "Cell_B_BD_Soft":      auroc_bd,
    }

    # Also check if we have hard-label caches
    hard_sig_path = os.path.join(args.cache_dir, "cache_sig_hard.npz")
    hard_bd_path  = os.path.join(args.cache_dir, "cache_bd_hard.npz")

    if os.path.exists(hard_sig_path):
        hard_sig = np.load(hard_sig_path)
        print("\nCell C — Sigmoid + Hard labels:")
        auroc_sig_hard = auroc_on_eval(hard_sig["probs"], hard_sig["labels"], eval_idx)
        results["Cell_C_Sigmoid_Hard"] = auroc_sig_hard

    if os.path.exists(hard_bd_path):
        hard_bd = np.load(hard_bd_path)
        hard_bd_p = hard_bd["params"]
        hard_bd_probs = np.stack([
            hard_bd_p[:,0]/(hard_bd_p[:,0]+hard_bd_p[:,1]),
            hard_bd_p[:,2]/(hard_bd_p[:,2]+hard_bd_p[:,3])
        ], axis=1)
        print("\nCell D — BD + Hard labels:")
        auroc_bd_hard = auroc_on_eval(hard_bd_probs, hard_bd["labels"], eval_idx)
        results["Cell_D_BD_Hard"] = auroc_bd_hard

    # Save
    out_path = os.path.join(args.cache_dir, "auroc_eval_only.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()