"""
Interactive 3D Uncertainty Landscape
=====================================
Generates a rotatable HTML plot of:
    X = Conformal Set Size (CPSL 1-4)
    Y = Beta-Dirichlet Evidence Tier (Low / Mid / High)
    Z = Accuracy

Uses cached inference results from dirichlet_analysis.py.
Requires: pip install plotly pandas numpy scikit-learn

Run:
    python interactive_3d.py
        --cache_dir runs/beta_dirichlet_.../uncertainty_analysis
        --out_dir   runs/beta_dirichlet_.../uncertainty_analysis
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str,
        default="/persist/QTIM/Active/26-0096/Corneal_Ulcer_Image_Sets/Data/corneal_ulcer_study_data_7_24_2025_lambda.csv")
    parser.add_argument("--cache_dir", type=str, required=True,
        help="Directory containing cache_bd.npz and cache_sig.npz")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed",  type=int,   default=42)
    parser.add_argument("--out_dir", type=str, default=None)
    return parser.parse_args()


PACS_PREFIX = "/pacs/OSHU/Corneal_Ulcer_Image_Sets"

def load_test_set(csv_path, seed=42):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["file_path","subject"])
    df = df[(df["fung_LCA_bin"]!=-1)|(df["bac_LCA_bin"]!=-1)].reset_index(drop=True)
    subjects = df["subject"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    _, temp_idx = next(gss.split(df, groups=subjects))
    df_temp = df.iloc[temp_idx].reset_index(drop=True)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    _, test_idx = next(gss2.split(df_temp, groups=df_temp["subject"].values))
    return df_temp.iloc[test_idx].reset_index(drop=True)


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


def main():
    args = get_args()
    if args.out_dir is None:
        args.out_dir = args.cache_dir

    # --- Load caches ---
    bd_cache  = os.path.join(args.cache_dir, "cache_bd.npz")
    sig_cache = os.path.join(args.cache_dir, "cache_sig.npz")
    assert os.path.exists(bd_cache),  f"BD cache not found: {bd_cache}"
    assert os.path.exists(sig_cache), f"Sig cache not found: {sig_cache}"

    bd_data  = np.load(bd_cache)
    sig_data = np.load(sig_cache)
    bd_params  = bd_data["params"];   bd_labels  = bd_data["labels"]
    sig_probs  = sig_data["probs"];   sig_labels = sig_data["labels"]

    # Derived BD quantities
    a_bac, b_bac, a_fung, b_fung = bd_params[:,0], bd_params[:,1], bd_params[:,2], bd_params[:,3]
    e_bac  = a_bac + b_bac
    e_fung = a_fung + b_fung

    # --- Load test set metadata ---
    df_test = load_test_set(args.csv, seed=args.seed)

    # --- Cal/eval split ---
    subjects = df_test["subject"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.80, random_state=42)
    cal_idx, eval_idx = next(gss.split(df_test, groups=subjects))

    # --- Conformal set sizes ---
    set_sizes, lab4, qhat = lac_4class(sig_probs, bd_labels, cal_idx, eval_idx, args.alpha)
    print(f"Conformal qhat: {qhat:.4f}  mean set size: {set_sizes.mean():.3f}")

    # --- Eval subset ---
    e_mean  = (e_bac[eval_idx] + e_fung[eval_idx]) / 2
    sig4    = make_4class(sig_probs)
    correct = (sig4[eval_idx].argmax(axis=1) == lab4[eval_idx])
    studies = df_test["study"].values[eval_idx]

    # --- Bin by BD evidence (terciles) ---
    t33, t67 = np.percentile(e_mean, 33), np.percentile(e_mean, 67)
    TIER_LABELS = [f"Low BD (e≤{t33:.1f})", f"Mid BD ({t33:.1f}<e≤{t67:.1f})", f"High BD (e>{t67:.1f})"]
    CONF_LABELS = ["CPSL=1", "CPSL=2", "CPSL=3", "CPSL=4"]

    def btier(e): return 0 if e<=t33 else (1 if e<=t67 else 2)
    def ctier(s): return min(s-1, 3)

    bd_t = np.array([btier(e) for e in e_mean])
    cf_t = np.array([ctier(s) for s in set_sizes])

    # --- Build matrices ---
    acc_mat = np.full((3,4), np.nan)
    n_mat   = np.zeros((3,4), dtype=int)
    hover_mat = [[""]*4 for _ in range(3)]

    for r in range(3):
        for c in range(4):
            mask = (bd_t==r)&(cf_t==c)
            if mask.sum()==0: continue
            acc = correct[mask].mean()
            n   = mask.sum()
            acc_mat[r,c] = acc
            n_mat[r,c]   = n
            # Build hover: study breakdown
            study_counts = {}
            for st in studies[mask]:
                short = st.replace(" photos","").replace("-Pictures","")
                study_counts[short] = study_counts.get(short,0)+1
            study_str = "<br>".join(f"  {k}: {v}" for k,v in sorted(study_counts.items(), key=lambda x:-x[1]))
            e_vals = e_mean[mask]
            hover_mat[r][c] = (
                f"<b>{TIER_LABELS[r]}</b><br>"
                f"<b>{CONF_LABELS[c]}</b><br>"
                f"Accuracy: {acc*100:.1f}%<br>"
                f"N: {n}<br>"
                f"Mean BD evidence: {e_vals.mean():.2f}<br>"
                f"<br>Studies:<br>{study_str}"
            )

    # ── Build interactive Plotly figure ──────────────────────────────────────

    # Color scale: red → yellow → green mapped to accuracy
    def acc_to_color(acc):
        if np.isnan(acc): return "rgba(50,50,50,0.1)"
        # Normalize 0.5-1.0 → 0-1
        t = (acc - 0.5) / 0.5
        t = max(0, min(1, t))
        if t < 0.5:
            r = 1.0;       g = t*2;      b = 0.0
        else:
            r = 1-(t-0.5)*2; g = 1.0;   b = 0.0
        return f"rgba({int(r*220)},{int(g*180)},{int(b*50)},0.85)"

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type":"scene"}, {"type":"scene"}]],
        subplot_titles=["Accuracy Landscape (rotate me!)", "Sample Counts"]
    )

    # --- Left: accuracy bars ---
    for r in range(3):
        for c in range(4):
            if np.isnan(acc_mat[r,c]): continue
            acc = acc_mat[r,c]
            n   = n_mat[r,c]
            col = acc_to_color(acc)

            # Draw a bar as a mesh (6 faces of a box)
            x0, x1 = c+0.1, c+0.85
            y0, y1 = r+0.1, r+0.85
            z0, z1 = 0, acc

            fig.add_trace(go.Mesh3d(
                x=[x0,x1,x1,x0, x0,x1,x1,x0],
                y=[y0,y0,y1,y1, y0,y0,y1,y1],
                z=[z0,z0,z0,z0, z1,z1,z1,z1],
                i=[0,0,0,4,4,4, 2,2,2,6,6,6],
                j=[1,2,3,5,6,7, 3,0,1,7,4,5],
                k=[2,3,1,6,7,5, 0,1,2,4,5,6],
                color=col,
                opacity=0.9,
                flatshading=True,
                hoverinfo="text",
                text=hover_mat[r][c],
                showscale=False,
                name=f"{TIER_LABELS[r]} / {CONF_LABELS[c]}",
            ), row=1, col=1)

            # Add text annotation as scatter3d
            fig.add_trace(go.Scatter3d(
                x=[c+0.475], y=[r+0.475], z=[z1+0.01],
                mode="text",
                text=[f"{acc*100:.0f}%<br>n={n}"],
                textfont=dict(size=9, color="black"),
                hoverinfo="skip",
                showlegend=False,
            ), row=1, col=1)

    # --- Right: sample count bars ---
    max_n = n_mat.max()
    for r in range(3):
        for c in range(4):
            n = n_mat[r,c]
            if n==0: continue
            t = n/max_n
            blue = f"rgba({int(20+t*30)},{int(60+t*100)},{int(150+t*80)},0.85)"

            x0, x1 = c+0.1, c+0.85
            y0, y1 = r+0.1, r+0.85
            z0, z1 = 0, n

            fig.add_trace(go.Mesh3d(
                x=[x0,x1,x1,x0, x0,x1,x1,x0],
                y=[y0,y0,y1,y1, y0,y0,y1,y1],
                z=[z0,z0,z0,z0, z1,z1,z1,z1],
                i=[0,0,0,4,4,4, 2,2,2,6,6,6],
                j=[1,2,3,5,6,7, 3,0,1,7,4,5],
                k=[2,3,1,6,7,5, 0,1,2,4,5,6],
                color=blue, opacity=0.9, flatshading=True,
                hoverinfo="text",
                text=hover_mat[r][c],
                showscale=False,
                showlegend=False,
            ), row=1, col=2)

            fig.add_trace(go.Scatter3d(
                x=[c+0.475], y=[r+0.475], z=[z1+1],
                mode="text", text=[str(n)],
                textfont=dict(size=9, color="black"),
                hoverinfo="skip", showlegend=False,
            ), row=1, col=2)

    # Axis labels and layout
    axis_common = dict(
        backgroundcolor="rgba(240,244,248,0.8)",
        gridcolor="white", showbackground=True,
    )
    tick_x = dict(tickvals=[0.475,1.475,2.475,3.475],
                  ticktext=["CPSL=1<br>(confident)","CPSL=2","CPSL=3","CPSL=4<br>(uncertain)"],
                  title="Conformal Set Size")
    tick_y = dict(tickvals=[0.475,1.475,2.475],
                  ticktext=[f"Low BD<br>(e≤{t33:.1f})",
                            f"Mid BD<br>({t33:.1f}<e≤{t67:.1f})",
                            f"High BD<br>(e>{t67:.1f})"],
                  title="BD Evidence Tier")

    fig.update_layout(
        title=dict(
            text=(
                "<b>Joint Uncertainty Landscape: BD Evidence × Conformal Set Size × Accuracy</b><br>"
                "<sub>Hover over bars for study breakdown  ·  Drag to rotate  ·  Scroll to zoom</sub>"
            ),
            x=0.5, font=dict(size=15)
        ),
        scene=dict(
            xaxis={**axis_common, **tick_x},
            yaxis={**axis_common, **tick_y},
            zaxis={**axis_common, "title": "Accuracy", "range":[0.35,1.05]},
            camera=dict(eye=dict(x=-1.5, y=-1.8, z=1.2)),
            aspectmode="manual",
            aspectratio=dict(x=1.2, y=1.0, z=0.8),
        ),
        scene2=dict(
            xaxis={**axis_common, **tick_x},
            yaxis={**axis_common, **tick_y},
            zaxis={**axis_common, "title": "N samples"},
            camera=dict(eye=dict(x=-1.5, y=-1.8, z=1.2)),
            aspectmode="manual",
            aspectratio=dict(x=1.2, y=1.0, z=0.8),
        ),
        height=650,
        paper_bgcolor="white",
        showlegend=False,
        hoverlabel=dict(bgcolor="white", font_size=12),
    )

    out_path = os.path.join(args.out_dir, "3d_uncertainty_landscape_interactive.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Saved: {out_path}")
    print("Open in any browser — fully rotatable, hover for details.")


if __name__ == "__main__":
    main()