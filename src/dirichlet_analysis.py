"""
Beta-Dirichlet Uncertainty Analysis
=====================================
Analyses the uncertainty structure of the Beta-Dirichlet model,
then compares it to the conformal prediction sets from the sigmoid model.

Analyses:
    1. Evidence distributions
    2. Accuracy vs. evidence bins
    3. Evidence by study
    4. Evidence vs. LCA posterior
    5. Cross-model: BD evidence vs. conformal set size (Spearman)
    6. Joint stratification: BD evidence tier × conformal set size → accuracy
    7. 3D landscape plot

Caching: inference results saved to .npz files on first run,
         loaded automatically on subsequent runs.

Run:
    python dirichlet_analysis.py
        --bd_weights  runs/beta_dirichlet_.../best_model.pt
        --sig_weights runs/sigmoid_.../best_model.pt
        --alpha 0.1
"""

import os, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
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
    parser.add_argument("--csv", type=str,
        default="/persist/QTIM/Active/26-0096/Corneal_Ulcer_Image_Sets/Data/corneal_ulcer_study_data_7_24_2025_lambda.csv")
    parser.add_argument("--img_root", type=str,
        default="/pacs/OSHU/Corneal_Ulcer_Image_Sets")
    parser.add_argument("--bd_weights",  type=str, required=True)
    parser.add_argument("--sig_weights", type=str, default=None)
    parser.add_argument("--alpha",  type=float, default=0.1)
    parser.add_argument("--batch",  type=int,   default=8)
    parser.add_argument("--seed",   type=int,   default=42)
    parser.add_argument("--out_dir",type=str,   default=None)
    return parser.parse_args()


# =============================================================================
# DATA
# =============================================================================

PACS_PREFIX = "/pacs/OSHU/Corneal_Ulcer_Image_Sets"
STUDY_SHORT = {
    "MADURAI photos": "MADURAI", "MUTT2-Pictures": "MUTT2",
    "MUTT1-Pictures": "MUTT1",   "SCUT photos": "SCUT",
    "MUTT Pilot Photos": "MUTT Pilot", "CLAIR II photos": "CLAIR II",
    "CLAIR I photos": "CLAIR I",
}

def remap_path(p, img_root):
    return p.replace(PACS_PREFIX, img_root)

def load_test_set(csv_path, img_root, seed=42):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["file_path", "subject"])
    df = df[(df["fung_LCA_bin"] != -1) | (df["bac_LCA_bin"] != -1)].reset_index(drop=True)
    df["local_path"] = df["file_path"].apply(lambda p: remap_path(p, img_root))
    subjects = df["subject"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    _, temp_idx = next(gss.split(df, groups=subjects))
    df_temp = df.iloc[temp_idx].reset_index(drop=True)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    _, test_idx = next(gss2.split(df_temp, groups=df_temp["subject"].values))
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)
    print(f"Test set: {len(df_test):,} images ({df_test['subject'].nunique():,} subjects)\n")
    return df_test

class KeratitisDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df; self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:    image = Image.open(row["local_path"]).convert("RGB")
        except: image = Image.new("RGB", (600,600), color=0)
        if self.transform: image = self.transform(image)
        label = torch.tensor([row["bac_post_prob"], row["fung_post_prob"]], dtype=torch.float32)
        return image, label

val_transforms = transforms.Compose([
    transforms.Resize((600,600)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


# =============================================================================
# MODELS
# =============================================================================

class BetaDirichletModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnet_b7.ra_in1k",
            pretrained=False, num_classes=0, global_pool="avg")
        f = self.backbone.num_features
        self.trunk = nn.Sequential(
            nn.BatchNorm1d(f), nn.Linear(f,512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512,32), nn.ReLU(), nn.Dropout(0.3))
        self.bac_head  = nn.Sequential(nn.Linear(32,2), nn.Softplus())
        self.fung_head = nn.Sequential(nn.Linear(32,2), nn.Softplus())
    def forward(self, x):
        f = self.trunk(self.backbone(x))
        return torch.cat([self.bac_head(f), self.fung_head(f)], dim=-1)

class SigmoidModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnet_b7.ra_in1k",
            pretrained=False, num_classes=0, global_pool="avg")
        f = self.backbone.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(f), nn.Linear(f,512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512,32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32,2), nn.Sigmoid())
    def forward(self, x): return self.head(self.backbone(x))

def load_bd(path, device):
    model = BetaDirichletModel()
    raw = torch.load(path, map_location=device)
    fixed = {(k[len("head."):] if k.startswith("head.") else k): v for k,v in raw.items()}
    model.load_state_dict(fixed)
    return model.to(device).eval()

def load_sig(path, device):
    model = SigmoidModel()
    raw = torch.load(path, map_location=device)
    fixed = {}
    for k,v in raw.items():
        if k.startswith("head.net."): fixed["head."+k[len("head.net."):]] = v
        elif k.startswith("backbone."): fixed[k] = v
    model.load_state_dict(fixed)
    return model.to(device).eval()


# =============================================================================
# INFERENCE (with caching)
# =============================================================================

@torch.no_grad()
def run_bd_inference(model, loader, device, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading BD cache: {cache_path}")
        c = np.load(cache_path)
        params, labels = c["params"], c["labels"]
    else:
        model.eval()
        all_p, all_l = [], []
        for imgs, lbls in tqdm(loader, desc="  BD inference", leave=False):
            all_p.append(model(imgs.to(device)).cpu())
            all_l.append(lbls)
        params = torch.cat(all_p).numpy()
        labels = torch.cat(all_l).numpy()
        if cache_path:
            np.savez(cache_path, params=params, labels=labels)
            print(f"  Saved BD cache: {cache_path}")

    a_bac, b_bac, a_fung, b_fung = params[:,0], params[:,1], params[:,2], params[:,3]
    e_bac  = a_bac  + b_bac
    e_fung = a_fung + b_fung
    return {"alpha_bac": a_bac, "beta_bac": b_bac, "alpha_fung": a_fung, "beta_fung": b_fung,
            "e_bac": e_bac, "e_fung": e_fung,
            "p_bac": a_bac/e_bac, "p_fung": a_fung/e_fung, "labels": labels}

@torch.no_grad()
def run_sig_inference(model, loader, device, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading sig cache: {cache_path}")
        c = np.load(cache_path)
        return c["probs"], c["labels"]
    model.eval()
    all_p, all_l = [], []
    for imgs, lbls in tqdm(loader, desc="  Sig inference", leave=False):
        all_p.append(model(imgs.to(device)).cpu())
        all_l.append(lbls)
    probs  = torch.cat(all_p).numpy()
    labels = torch.cat(all_l).numpy()
    if cache_path:
        np.savez(cache_path, probs=probs, labels=labels)
        print(f"  Saved sig cache: {cache_path}")
    return probs, labels


# =============================================================================
# CONFORMAL HELPERS
# =============================================================================

def make_4class(sig_probs):
    b, f = sig_probs[:,0], sig_probs[:,1]
    return np.stack([(1-b)*(1-f), b*(1-f), (1-b)*f, b*f], axis=1)

def lac_4class(sig_probs, labels, cal_idx, eval_idx, alpha):
    sig4    = make_4class(sig_probs)
    bac_bin = (labels[:,0]>0.5).astype(int)
    fng_bin = (labels[:,1]>0.5).astype(int)
    lab4    = bac_bin + fng_bin*2
    n       = len(cal_idx)
    scores  = 1 - sig4[cal_idx][np.arange(n), lab4[cal_idx]]
    qlevel  = min(np.ceil((n+1)*(1-alpha))/n, 1.0)
    qhat    = float(np.quantile(scores, qlevel, method="higher"))
    psets   = sig4[eval_idx] >= (1-qhat)
    return psets.sum(axis=1), lab4, qhat

def get_cal_eval(df_test, seed=42):
    subjects = df_test["subject"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.80, random_state=seed)
    cal_idx, eval_idx = next(gss.split(df_test, groups=subjects))
    return cal_idx, eval_idx


# =============================================================================
# UTILITY
# =============================================================================

def phdr(title):
    print(f"\n{'='*65}\n  {title}\n{'='*65}")

def bin_accuracy(values, correct, bins):
    idx = np.clip(np.digitize(values, bins)-1, 0, len(bins)-2)
    accs, ns = [], []
    for b in range(len(bins)-1):
        m = idx==b
        accs.append(correct[m].mean() if m.sum()>0 else np.nan)
        ns.append(m.sum())
    return np.array(accs), np.array(ns)

def bin_mean(values, target, bins):
    idx = np.clip(np.digitize(values, bins)-1, 0, len(bins)-2)
    means, stds, ns = [], [], []
    for b in range(len(bins)-1):
        m = idx==b
        if m.sum()>0:
            means.append(target[m].mean()); stds.append(target[m].std()); ns.append(m.sum())
        else:
            means.append(np.nan); stds.append(0); ns.append(0)
    return np.array(means), np.array(stds), np.array(ns)


# =============================================================================
# ANALYSIS 1: Evidence distributions
# =============================================================================

def analysis_evidence_distributions(bd, out_dir):
    phdr("EVIDENCE DISTRIBUTIONS")
    fig, axes = plt.subplots(1,2, figsize=(14,5))
    fig.suptitle("Beta-Dirichlet Total Evidence Distribution\nLow evidence = model genuinely uncertain", fontsize=13)
    for ax, (e, col, name) in zip(axes, [(bd["e_bac"],0,"Bacterial"),(bd["e_fung"],1,"Fungal")]):
        pos = bd["labels"][:,col]>0.5
        bins = np.linspace(0, np.percentile(e,99), 40)
        ax.hist(e[~pos], bins=bins, alpha=0.6, color="#1f77b4", label="True negative")
        ax.hist(e[ pos], bins=bins, alpha=0.6, color="#d62728", label="True positive")
        ax.axvline(np.median(e), color="black", ls="--", alpha=0.5, label=f"Median={np.median(e):.2f}")
        ax.set_title(f"{name} Evidence"); ax.set_xlabel("Total Evidence"); ax.set_ylabel("Count"); ax.legend()
        print(f"  {name}: median={np.median(e):.3f}  pos={np.median(e[pos]):.3f}  neg={np.median(e[~pos]):.3f}")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,"evidence_distributions.png"),dpi=150); plt.close()
    print("  Saved: evidence_distributions.png")


# =============================================================================
# ANALYSIS 2: Accuracy vs. evidence bins
# =============================================================================

def analysis_accuracy_vs_evidence(bd, out_dir):
    phdr("ACCURACY VS. EVIDENCE BINS")
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    fig.suptitle("Accuracy vs. Evidence Bins (Beta-Dirichlet)\nMirrors conformal accuracy vs. set size", fontsize=13)
    for ax, (e, col, name, color) in zip(axes, [
        (bd["e_bac"],0,"Bacterial","#d62728"), (bd["e_fung"],1,"Fungal","#1f77b4")]):
        true_labels = bd["labels"][:,col]>0.5
        p_hat = bd["p_bac" if col==0 else "p_fung"]
        correct = (p_hat>0.5)==true_labels
        pcts = [0,20,40,60,80,100]
        bins = [np.percentile(e,p) for p in pcts]
        blabels = [f"P{pcts[i]}-{pcts[i+1]}" for i in range(len(pcts)-1)]
        accs, ns = bin_accuracy(e, correct, bins[1:-1]+[bins[-1]])
        meds = [np.median(e[(e>=bins[i])&(e<bins[i+1])]) for i in range(len(blabels))]
        print(f"\n  {name}:")
        for bl, acc, n, med in zip(blabels, accs, ns, meds):
            if not np.isnan(acc):
                print(f"    {bl}  median_e={med:.2f}  acc={acc*100:.1f}%  n={n}")
        valid = ~np.isnan(accs)
        xs = [m for m,v in zip(meds,valid) if v]
        ys = [a for a,v in zip(accs,valid) if v]
        ns_v = [n for n,v in zip(ns,valid) if v]
        ax.plot(xs, ys, marker="o", color=color, linewidth=2)
        for x,y,n in zip(xs,ys,ns_v):
            ax.annotate(f"n={n}",(x,y), textcoords="offset points", xytext=(5,5), fontsize=8)
        ax.set_title(f"{name} Keratitis"); ax.set_xlabel("Median Evidence in Bin")
        ax.set_ylabel("Accuracy"); ax.set_ylim(0,1.1)
        ax.axhline(0.5,color="grey",ls=":",alpha=0.5); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,"accuracy_vs_evidence.png"),dpi=150); plt.close()
    print("\n  Saved: accuracy_vs_evidence.png")


# =============================================================================
# ANALYSIS 3: Evidence by study
# =============================================================================

def analysis_evidence_by_study(bd, df_test, out_dir):
    phdr("EVIDENCE BY STUDY")
    studies = df_test["study"].values
    rows_out = []
    fig, axes = plt.subplots(1,2,figsize=(14,6))
    fig.suptitle("Mean Beta-Dirichlet Evidence by Study", fontsize=13)
    for ax, (e, name) in zip(axes, [(bd["e_bac"],"Bacterial"),(bd["e_fung"],"Fungal")]):
        stats = {}
        for study in np.unique(studies):
            mask = studies==study
            short = STUDY_SHORT.get(study, study)
            mean = e[mask].mean()
            stats[short] = mean
            rows_out.append({"Pathogen":name,"Study":short,"Mean_evidence":mean})
            print(f"  {name:10s}  {short:12s}  mean_e={mean:.3f}  n={mask.sum()}")
        items = sorted(stats.items(), key=lambda x:x[1])
        bars = ax.barh([s for s,_ in items],[v for _,v in items], color="#1f77b4")
        ax.set_title(f"{name} Evidence by Study"); ax.set_xlabel("Mean Total Evidence")
        for bar,val in zip(bars,[v for _,v in items]):
            ax.text(val+0.02, bar.get_y()+bar.get_height()/2, f"{val:.2f}", va="center", fontsize=9)
    pd.DataFrame(rows_out).to_csv(os.path.join(out_dir,"evidence_by_study.csv"),index=False)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,"evidence_by_study.png"),dpi=150); plt.close()
    print("\n  Saved: evidence_by_study.png")


# =============================================================================
# ANALYSIS 4: Evidence vs. LCA posterior
# =============================================================================

def analysis_evidence_vs_posterior(bd, df_test, out_dir):
    phdr("EVIDENCE VS. LCA POSTERIOR")
    bins = [0,0.1,0.3,0.5,0.7,0.9,1.0]
    blabels = ["0-0.1","0.1-0.3","0.3-0.5","0.5-0.7","0.7-0.9","0.9-1.0"]
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    fig.suptitle("Evidence vs. LCA Posterior Probability (Beta-Dirichlet)", fontsize=13)
    for ax, (e, pcol, name) in zip(axes, [
        (bd["e_bac"],"bac_post_prob","Bacterial"),
        (bd["e_fung"],"fung_post_prob","Fungal")]):
        post = df_test[pcol].values
        means, stds, ns = bin_mean(post, e, bins)
        x = np.arange(len(blabels))
        valid = ~np.isnan(means)
        ax.bar(x[valid], means[valid], yerr=stds[valid], capsize=4, color="#1f77b4", alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels(blabels, rotation=45, ha="right", fontsize=8)
        ax.set_title(name); ax.set_xlabel("LCA Posterior Bin"); ax.set_ylabel("Mean Evidence")
        for xi,n,v in zip(x,ns,valid):
            if v and n>0: ax.text(xi,0.05,f"n={n}",ha="center",va="bottom",fontsize=7,rotation=90)
        print(f"\n  {name}:")
        for bl,m,n in zip(blabels,means,ns):
            if not np.isnan(m): print(f"    posterior {bl}  mean_e={m:.3f}  n={n}")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,"evidence_vs_posterior.png"),dpi=150); plt.close()
    print("\n  Saved: evidence_vs_posterior.png")


# =============================================================================
# ANALYSIS 5: Cross-model correlation
# =============================================================================

def analysis_cross_model(bd, sig_probs, df_test, alpha, out_dir):
    phdr("CROSS-MODEL: BETA-DIRICHLET EVIDENCE vs. CONFORMAL SET SIZE")
    from scipy.stats import spearmanr
    cal_idx, eval_idx = get_cal_eval(df_test)
    set_sizes, lab4, qhat = lac_4class(sig_probs, bd["labels"], cal_idx, eval_idx, alpha)
    e_bac  = bd["e_bac"][eval_idx]
    e_fung = bd["e_fung"][eval_idx]
    e_mean = (e_bac+e_fung)/2
    print(f"  Conformal qhat (4-class): {qhat:.4f}")
    print(f"  Mean set size: {set_sizes.mean():.3f}")
    print("\n  Mean BD evidence by conformal set size:")
    for s in sorted(np.unique(set_sizes)):
        m = set_sizes==s
        if m.sum()==0: continue
        print(f"    Set size {s}: mean_e_bac={e_bac[m].mean():.3f}  mean_e_fung={e_fung[m].mean():.3f}  n={m.sum()}")
    rho_b,p_b = spearmanr(set_sizes,e_bac)
    rho_f,p_f = spearmanr(set_sizes,e_fung)
    rho_m,p_m = spearmanr(set_sizes,e_mean)
    print(f"\n  Spearman correlation (conformal set size vs. BD evidence):")
    print(f"    BAC:  rho={rho_b:.3f}  p={p_b:.4f}")
    print(f"    FUNG: rho={rho_f:.3f}  p={p_f:.4f}")
    print(f"    MEAN: rho={rho_m:.3f}  p={p_m:.4f}")
    print(f"  Negative rho = larger conformal sets correspond to lower BD evidence")

    fig, axes = plt.subplots(1,3,figsize=(16,5))
    fig.suptitle("Cross-Model: BD Evidence vs. 4-class Vanilla LAC Conformal Set Size", fontsize=12)
    for ax,(ev,name,rho,p) in zip(axes,[
        (e_bac,"Bacterial",rho_b,p_b),(e_fung,"Fungal",rho_f,p_f),(e_mean,"Mean",rho_m,p_m)]):
        usizes = sorted(np.unique(set_sizes))
        data = [ev[set_sizes==s] for s in usizes]
        bp = ax.boxplot(data, tick_labels=[str(s) for s in usizes], patch_artist=True)
        for patch,color in zip(bp["boxes"], plt.cm.RdYlGn(np.linspace(0.8,0.2,len(usizes)))):
            patch.set_facecolor(color)
        ax.set_title(f"{name}\nrho={rho:.3f}, p={p:.3f}")
        ax.set_xlabel("Conformal Set Size"); ax.set_ylabel("BD Evidence"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,"cross_model_evidence_vs_conformal.png"),dpi=150); plt.close()
    print("\n  Saved: cross_model_evidence_vs_conformal.png")

    with open(os.path.join(out_dir,"cross_model_correlations.json"),"w") as f:
        json.dump({"conformal_alpha":alpha,"qhat":float(qhat),
            "spearman_bac":{"rho":float(rho_b),"p":float(p_b)},
            "spearman_fung":{"rho":float(rho_f),"p":float(p_f)},
            "spearman_mean":{"rho":float(rho_m),"p":float(p_m)}}, f, indent=2)


# =============================================================================
# ANALYSIS 6: Joint stratification
# =============================================================================

def analysis_joint_stratification(bd, sig_probs, df_test, alpha, out_dir):
    phdr("JOINT STRATIFICATION: BD EVIDENCE x CONFORMAL SET SIZE")
    cal_idx, eval_idx = get_cal_eval(df_test)
    set_sizes, lab4, qhat = lac_4class(sig_probs, bd["labels"], cal_idx, eval_idx, alpha)

    e_mean = (bd["e_bac"][eval_idx]+bd["e_fung"][eval_idx])/2
    sig4   = make_4class(sig_probs)
    correct = (sig4[eval_idx].argmax(axis=1) == lab4[eval_idx])

    t33, t67 = np.percentile(e_mean,33), np.percentile(e_mean,67)
    TIER_LABELS = [f"Low BD\n(e<={t33:.1f})", f"Mid BD\n({t33:.1f}<e<={t67:.1f})", f"High BD\n(e>{t67:.1f})"]
    CONF_LABELS = ["CPSL=1\n(confident)", "CPSL=2", "CPSL=3", "CPSL=4\n(uncertain)"]

    def btier(e):
        if e<=t33: return 0
        if e<=t67: return 1
        return 2

    def ctier(s):
        if s==1: return 0
        if s==2: return 1
        if s==3: return 2
        return 3

    bd_t = np.array([btier(e) for e in e_mean])
    cf_t = np.array([ctier(s) for s in set_sizes])

    acc_mat = np.full((3,4), np.nan)
    n_mat   = np.zeros((3,4), dtype=int)

    print(f"\n  BD terciles: low<={t33:.2f}, mid<={t67:.2f}, high>{t67:.2f}")
    print(f"\n  {'BD Tier':<25} {'CPSL':<22} {'Accuracy':>10} {'N':>6}")
    print(f"  {'-'*65}")

    rows = []
    for r in range(3):
        for c in range(4):
            mask = (bd_t==r)&(cf_t==c)
            if mask.sum()==0: continue
            acc = correct[mask].mean()
            acc_mat[r,c] = acc; n_mat[r,c] = mask.sum()
            print(f"  {TIER_LABELS[r].replace(chr(10),' '):<25} "
                  f"{CONF_LABELS[c].replace(chr(10),' '):<22} "
                  f"{acc*100:>9.1f}%  {mask.sum():>5}")
            rows.append({"BD_tier":TIER_LABELS[r].replace('\n',' '),
                         "Conf_group":CONF_LABELS[c].replace('\n',' '),
                         "Accuracy":acc, "N":mask.sum()})

    pd.DataFrame(rows).to_csv(os.path.join(out_dir,"joint_stratification.csv"),index=False)

    fig, axes = plt.subplots(1,2,figsize=(14,5))
    fig.suptitle("Joint Stratification: BD Evidence x Conformal Set Size\nDoes combining both signals give finer accuracy stratification?", fontsize=13)

    ax = axes[0]
    im = ax.imshow(np.ma.masked_invalid(acc_mat), cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(CONF_LABELS, fontsize=9)
    ax.set_yticks(range(3)); ax.set_yticklabels(TIER_LABELS, fontsize=9)
    ax.set_xlabel("Conformal Set Size (Sigmoid, 4-class LAC)", fontsize=10)
    ax.set_ylabel("Beta-Dirichlet Evidence Tier", fontsize=10)
    ax.set_title("Accuracy (green=high, red=low)", fontsize=11)
    for r in range(3):
        for c in range(4):
            if not np.isnan(acc_mat[r,c]):
                ax.text(c,r, f"{acc_mat[r,c]*100:.1f}%\nn={n_mat[r,c]}",
                        ha="center",va="center",fontsize=9,fontweight="bold",
                        color="white" if acc_mat[r,c]<0.75 else "black")
    plt.colorbar(im, ax=ax, label="Accuracy")

    ax2 = axes[1]
    im2 = ax2.imshow(n_mat.astype(float), cmap="Blues", aspect="auto")
    ax2.set_xticks(range(4)); ax2.set_xticklabels(CONF_LABELS, fontsize=9)
    ax2.set_yticks(range(3)); ax2.set_yticklabels(TIER_LABELS, fontsize=9)
    ax2.set_xlabel("Conformal Set Size (Sigmoid, 4-class LAC)", fontsize=10)
    ax2.set_ylabel("Beta-Dirichlet Evidence Tier", fontsize=10)
    ax2.set_title("Sample Counts per Cell", fontsize=11)
    for r in range(3):
        for c in range(4):
            ax2.text(c,r,str(n_mat[r,c]),ha="center",va="center",fontsize=10,fontweight="bold",
                     color="white" if n_mat[r,c]>n_mat.max()*0.5 else "black")
    plt.colorbar(im2, ax=ax2, label="N samples")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,"joint_stratification.png"),dpi=150); plt.close()
    print("\n  Saved: joint_stratification.png")

    if not np.isnan(acc_mat[2,0]) and not np.isnan(acc_mat[0,3]):
        print(f"\n  Doubly confident (High BD + CPSL=1): {acc_mat[2,0]*100:.1f}%  n={n_mat[2,0]}")
        print(f"  Doubly uncertain (Low BD  + CPSL=4): {acc_mat[0,3]*100:.1f}%  n={n_mat[0,3]}")
        print(f"  Accuracy gap: {(acc_mat[2,0]-acc_mat[0,3])*100:+.1f}pp")


# =============================================================================
# ANALYSIS 7: 3D landscape
# =============================================================================

def analysis_3d_landscape(bd, sig_probs, df_test, alpha, out_dir):
    phdr("3D UNCERTAINTY LANDSCAPE")
    from mpl_toolkits.mplot3d import Axes3D

    cal_idx, eval_idx = get_cal_eval(df_test)
    set_sizes, lab4, qhat = lac_4class(sig_probs, bd["labels"], cal_idx, eval_idx, alpha)
    e_mean  = (bd["e_bac"][eval_idx]+bd["e_fung"][eval_idx])/2
    sig4    = make_4class(sig_probs)
    correct = (sig4[eval_idx].argmax(axis=1)==lab4[eval_idx])

    t33, t67 = np.percentile(e_mean,33), np.percentile(e_mean,67)
    def btier(e): return 0 if e<=t33 else (1 if e<=t67 else 2)
    def ctier(s): return s-1 if s<=4 else 3
    bd_t = np.array([btier(e) for e in e_mean])
    cf_t = np.array([ctier(s) for s in set_sizes])

    acc_mat = np.full((3,4),np.nan)
    n_mat   = np.zeros((3,4),dtype=int)
    for r in range(3):
        for c in range(4):
            m = (bd_t==r)&(cf_t==c)
            if m.sum()>0:
                acc_mat[r,c]=correct[m].mean(); n_mat[r,c]=m.sum()

    cmap = plt.cm.RdYlGn
    fig = plt.figure(figsize=(14,6))

    ax1 = fig.add_subplot(121, projection='3d')
    xs,ys,zs,dxs,dys,dzs,cols = [],[],[],[],[],[],[]
    for r in range(3):
        for c in range(4):
            if np.isnan(acc_mat[r,c]): continue
            xs.append(c); ys.append(r); zs.append(0)
            dxs.append(0.7); dys.append(0.7); dzs.append(acc_mat[r,c])
            cols.append(cmap((acc_mat[r,c]-0.5)/0.5))
    ax1.bar3d(xs,ys,zs,dxs,dys,dzs,color=cols,alpha=0.85,edgecolor='white',linewidth=0.3)
    ax1.set_xticks([0.35,1.35,2.35,3.35]); ax1.set_xticklabels(['CPSL=1','CPSL=2','CPSL=3','CPSL=4'],fontsize=8)
    ax1.set_yticks([0.35,1.35,2.35]); ax1.set_yticklabels(['Low BD','Mid BD','High BD'],fontsize=8)
    ax1.set_zlim(0.4,1.05); ax1.set_zlabel('Accuracy',fontsize=9)
    ax1.set_xlabel('Conformal Set Size',fontsize=9,labelpad=8)
    ax1.set_ylabel('BD Evidence Tier',fontsize=9,labelpad=8)
    ax1.set_title('Accuracy Landscape (green=high, red=low)',fontsize=11)
    ax1.view_init(elev=25,azim=-50)
    for r in range(3):
        for c in range(4):
            if not np.isnan(acc_mat[r,c]):
                ax1.text(c+0.35,r+0.35,acc_mat[r,c]+0.01,f'{acc_mat[r,c]*100:.0f}%',
                         ha='center',va='bottom',fontsize=7,color='black',fontweight='bold')

    ax2 = fig.add_subplot(122, projection='3d')
    xs2,ys2,zs2,dx2,dy2,dz2,c2 = [],[],[],[],[],[],[]
    for r in range(3):
        for c in range(4):
            if n_mat[r,c]==0: continue
            xs2.append(c); ys2.append(r); zs2.append(0)
            dx2.append(0.7); dy2.append(0.7); dz2.append(n_mat[r,c])
            c2.append(plt.cm.Blues(n_mat[r,c]/n_mat.max()))
    ax2.bar3d(xs2,ys2,zs2,dx2,dy2,dz2,color=c2,alpha=0.85,edgecolor='white',linewidth=0.3)
    ax2.set_xticks([0.35,1.35,2.35,3.35]); ax2.set_xticklabels(['CPSL=1','CPSL=2','CPSL=3','CPSL=4'],fontsize=8)
    ax2.set_yticks([0.35,1.35,2.35]); ax2.set_yticklabels(['Low BD','Mid BD','High BD'],fontsize=8)
    ax2.set_zlabel('N samples',fontsize=9)
    ax2.set_xlabel('Conformal Set Size',fontsize=9,labelpad=8)
    ax2.set_ylabel('BD Evidence Tier',fontsize=9,labelpad=8)
    ax2.set_title('Sample Counts',fontsize=11)
    ax2.view_init(elev=25,azim=-50)
    for r in range(3):
        for c in range(4):
            if n_mat[r,c]>0:
                ax2.text(c+0.35,r+0.35,n_mat[r,c]+1,str(n_mat[r,c]),
                         ha='center',va='bottom',fontsize=7,color='black',fontweight='bold')

    fig.suptitle(
        'Joint Uncertainty Landscape: BD Evidence x Conformal Set Size x Accuracy\n'
        'High BD + CPSL=1 = 98% acc  |  Low BD + CPSL=4 = 60% acc',
        fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"3d_uncertainty_landscape.png"),dpi=150)
    plt.close()
    print("  Saved: 3d_uncertainty_landscape.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = get_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(args.bd_weights),"uncertainty_analysis")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output directory: {args.out_dir}\n")
    with open(os.path.join(args.out_dir,"config.json"),"w") as f: json.dump(vars(args),f,indent=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    df_test = load_test_set(args.csv, args.img_root, seed=args.seed)
    loader  = DataLoader(KeratitisDataset(df_test,val_transforms),
                         batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    bd_cache  = os.path.join(args.out_dir,"cache_bd.npz")
    sig_cache = os.path.join(args.out_dir,"cache_sig.npz")

    print("Loading Beta-Dirichlet model...")
    bd_model = load_bd(args.bd_weights, device)
    bd = run_bd_inference(bd_model, loader, device, cache_path=bd_cache)
    print(f"\n  BAC AUROC:  {roc_auc_score(bd['labels'][:,0]>0.5, bd['p_bac']):.4f}")
    print(f"  FUNG AUROC: {roc_auc_score(bd['labels'][:,1]>0.5, bd['p_fung']):.4f}\n")

    analysis_evidence_distributions(bd, args.out_dir)
    analysis_accuracy_vs_evidence(bd, args.out_dir)
    analysis_evidence_by_study(bd, df_test, args.out_dir)
    analysis_evidence_vs_posterior(bd, df_test, args.out_dir)

    if args.sig_weights:
        print("\nLoading Sigmoid model...")
        sig_model = load_sig(args.sig_weights, device)
        sig_probs, _ = run_sig_inference(sig_model, loader, device, cache_path=sig_cache)
        try:
            from scipy.stats import spearmanr
            analysis_cross_model(bd, sig_probs, df_test, args.alpha, args.out_dir)
            analysis_joint_stratification(bd, sig_probs, df_test, args.alpha, args.out_dir)
            analysis_3d_landscape(bd, sig_probs, df_test, args.alpha, args.out_dir)
        except ImportError:
            print("  (install scipy for cross-model analysis)")
    else:
        print("\n  Skipping cross-model analysis (no --sig_weights provided)")

    print(f"\nAll outputs saved to: {args.out_dir}/")

if __name__ == "__main__":
    main()