"""
Is there a genuine "both" or "neither" group? Latent class models fitted to the four tests of the patients who
have all four (MADURAI), at the subject level, from the redo's label table (paper_labels.csv).

Models (tests: Gram, KOH, blood agar, PDA; each test independent of the others given the latent state):
  2-class          the paper's LCA: every ulcer is bacterial or fungal, never both or neither.
  2-class + RE     the same with a random effect shared by the four tests (tests correlated within a class), as
                   randomLCA(random = TRUE, constload = TRUE) in the authors' demo.
  4-state          two latent pathogens, bacteria present yes/no and fungi present yes/no, with a free joint
                   distribution (bacterial only, fungal only, both, neither). Gram and blood agar depend only on
                   bacteria, KOH and PDA only on fungi. The 2-class model is this one with P(both) = P(neither) = 0.

Reports fit (log-likelihood, AIC, BIC), the estimated groups, test sensitivity/specificity, observed vs expected
patients per test pattern, and profile-likelihood intervals for P(neither) and P(both) under the 4-state model.
Aggregates only.

    python src/lca_states.py --labels results/labels/paper_labels.csv --out results/lca_states
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eda_paper_data import md  # noqa: E402

TESTS = ["bac_smear", "fung_smear", "bac_cult_positive", "fung_cult_positive"]   # Gram, KOH, BA, PDA
NAMES = ["Gram", "KOH", "BA", "PDA"]
STATES = [(1, 0), (0, 1), (1, 1), (0, 0)]                                         # (bacteria, fungi)
STATE_NAMES = ["bacterial only", "fungal only", "both", "neither"]


def patterns(labels, study):
    t = pd.read_csv(labels)
    t = t[t["study"].astype(str) == study]
    sub = t.groupby("subject")[TESTS].agg(["max", "min"])
    conflicts = int(sum(((sub[(c, "max")] == 1) & (sub[(c, "min")] == 0)).sum() for c in TESTS))
    s = pd.DataFrame({c: sub[(c, "max")] for c in TESTS})
    s = s[s[TESTS].notna().all(axis=1)].astype(int)
    grid = pd.MultiIndex.from_product([[0, 1]] * 4, names=TESTS).to_frame(index=False)
    cnt = s.groupby(TESTS).size().rename("n").reset_index()
    cnt = grid.merge(cnt, on=TESTS, how="left").fillna({"n": 0})
    return cnt[TESTS].to_numpy(), cnt["n"].to_numpy(float), len(t), conflicts


def sens_spec_pairs(th):
    """(sensitivity, specificity) for Gram, BA, KOH, PDA from th[3:11] = (a, u) per test: P(+ | organism absent)
    = expit(a), P(+ | present) = that + (1 - that) * expit(u), so a test is always more often positive when its
    organism is present (this fixes which latent state is which; no label switching)."""
    out = []
    for j in range(4):
        q0 = expit(th[3 + 2 * j])
        q1 = q0 + (1 - q0) * expit(th[4 + 2 * j])
        out.append((q1, 1 - q0))
    return out


class Models:
    def __init__(self, Y, n):
        self.Y, self.n = Y, n
        x, w = np.polynomial.hermite_e.hermegauss(31)
        self.gh_x, self.gh_lw = x, np.log(w / w.sum())

    # --- 4-state (the 2-class model is the special case with both/neither weights at ~0) ---------------
    def comp_4state(self, th):
        """log P(pattern, state) for each state: (patterns, 4)."""
        z = np.r_[0.0, th[:3]]
        lw = z - logsumexp(z)
        (seG, spG), (seB, spB), (seK, spK), (seP, spP) = sens_spec_pairs(th)
        G, K, B, P = self.Y.T
        out = []
        for (b, f), lwk in zip(STATES, lw):
            p = [seG if b else 1 - spG, seK if f else 1 - spK, seB if b else 1 - spB, seP if f else 1 - spP]
            lp = sum(y * np.log(q) + (1 - y) * np.log(1 - q) for y, q in zip((G, K, B, P), p))
            out.append(lwk + lp)
        return np.stack(out, 1)

    def ll_4state(self, th):
        return float(np.sum(self.n * logsumexp(self.comp_4state(th), axis=1)))

    @staticmethod
    def th_2class(t):
        return np.r_[t[0], -40.0, -40.0, t[1:9]]

    def ll_2class(self, t):
        return self.ll_4state(self.th_2class(t))

    # --- 2-class with a shared random effect ---------------------------------------------------------------
    def comp_re(self, t):
        a, muB, muF, lam = t[0], t[1:5], t[5:9], t[9]
        def lpc(mu):
            lp = mu[None, None, :] + lam * self.gh_x[None, :, None]
            Y = self.Y[:, None, :]
            ll = (Y * -np.logaddexp(0, -lp) + (1 - Y) * -np.logaddexp(0, lp)).sum(-1)
            return logsumexp(ll + self.gh_lw[None, :], axis=1)
        return np.stack([-np.logaddexp(0, -a) + lpc(muB), -np.logaddexp(0, a) + lpc(muF)], 1)

    def ll_re(self, t):
        return float(np.sum(self.n * logsumexp(self.comp_re(t), axis=1)))


def fit(f, k, x0s):
    best = None
    for x0 in x0s:
        r = minimize(lambda t: -f(t), x0[:k], method="L-BFGS-B", bounds=[(-12, 12)] * k)
        if best is None or r.fun < best.fun:
            best = r
    return best


def starts(rng, k, m):
    for _ in range(m):
        x = rng.normal(0, 1.5, k)
        off = k - 8                                   # the 8 test parameters come last
        x[off::2] = rng.normal(-3, 1, 4)              # false-positive rate around 5%
        x[off + 1::2] = rng.normal(0, 1.5, 4)
        yield x


def starts_re(rng, m):
    for _ in range(m):
        x = rng.normal(0, 1, 10)
        x[1:5] = np.array([-0.5, -2.5, -0.5, -3.0]) + rng.normal(0, .5, 4)   # bacterial class: Gram, KOH, BA, PDA
        x[5:9] = np.array([-3.0, 2.0, -3.0, 1.0]) + rng.normal(0, .5, 4)     # fungal class
        yield x


def profile(M, which, values, rng, n_starts, th_best):
    """Max log-likelihood of the 4-state model with P(state `which`) fixed; each point starts from the best
    full fit and from the previous point (warm starts), plus a few random starts."""
    w_best = np.exp(np.r_[0.0, th_best[:3]] - logsumexp(np.r_[0.0, th_best[:3]]))
    others = [i for i in range(4) if i != which]
    warm = np.r_[np.log(w_best[others[1:]] / w_best[others[0]]), th_best[3:]]
    out = []
    for v in values:
        def f(t, v=v):
            z = np.r_[0.0, t[:2]]
            w = np.exp(z - logsumexp(z)) * (1 - v)
            full = np.empty(4)
            full[others] = w
            full[which] = max(v, 1e-12)
            lw = np.log(full)
            return M.ll_4state(np.r_[lw[1:] - lw[0], t[2:]])
        r = fit(f, 10, [warm] + list(starts(rng, 10, n_starts)))
        warm = r.x
        out.append(-r.fun)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="results/labels/paper_labels.csv")
    ap.add_argument("--study", default="MADURAI")
    ap.add_argument("--out", default="results/lca_states")
    ap.add_argument("--starts", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    Y, n, n_images, conflicts = patterns(a.labels, a.study)
    N = n.sum()
    M = Models(Y, n)

    r2 = fit(M.ll_2class, 9, list(starts(rng, 9, a.starts)))
    rre = fit(M.ll_re, 10, list(starts_re(rng, a.starts)))
    if rre.x[1] < rre.x[5]:                           # Gram more often positive in the "fungal" class: swap names
        rre.x = np.r_[-rre.x[0], rre.x[5:9], rre.x[1:5], rre.x[9]]
    r4 = fit(M.ll_4state, 11, list(starts(rng, 11, a.starts)))
    th2, th4 = M.th_2class(r2.x), r4.x

    def summary(name, ll, k):
        return {"model": name, "parameters": k, "log-likelihood": ll, "AIC": -2 * ll + 2 * k, "BIC": -2 * ll + k * np.log(N)}
    fits = pd.DataFrame([summary("2-class (the paper's)", -r2.fun, 9), summary("2-class + random effect", -rre.fun, 10),
                         summary("4-state (bacteria, fungi)", -r4.fun, 11)])

    def groups(th):
        z = np.r_[0.0, th[:3]]
        return np.exp(z - logsumexp(z))
    g2, g4 = groups(th2), groups(th4)
    pb_re = expit(rre.x[0])
    grp = pd.DataFrame({"group": STATE_NAMES, "2-class": g2, "2-class + RE": [pb_re, 1 - pb_re, 0, 0], "4-state": g4})
    order = [0, 2, 1, 3]                              # pairs come as Gram, BA, KOH, PDA; report Gram, KOH, BA, PDA
    s2, s4 = sens_spec_pairs(th2), sens_spec_pairs(th4)
    ss = pd.DataFrame({"test": NAMES,
                       "2-class sens": [s2[i][0] for i in order], "2-class spec": [s2[i][1] for i in order],
                       "4-state sens": [s4[i][0] for i in order], "4-state spec": [s4[i][1] for i in order]})

    # observed vs expected patients per pattern, and who the all-negative patients are under each model
    e2 = N * np.exp(logsumexp(M.comp_4state(th2), axis=1))
    ere = N * np.exp(logsumexp(M.comp_re(rre.x), axis=1))
    c4 = M.comp_4state(th4)
    e4 = N * np.exp(logsumexp(c4, axis=1))
    pat = [" ".join(f"{nm}{'+' if v else '-'}" for nm, v in zip(NAMES, row)) for row in Y]
    obs = pd.DataFrame({"pattern": pat, "observed": n, "expected, 2-class": e2, "expected, 2-class + RE": ere,
                        "expected, 4-state": e4}).sort_values("observed", ascending=False)
    neg = int(np.where((Y == 0).all(1))[0][0])
    c2, cre = M.comp_4state(th2)[neg], M.comp_re(rre.x)[neg]
    post4 = np.exp(c4[neg] - logsumexp(c4[neg]))
    allneg = pd.DataFrame({"model": ["2-class", "2-class + RE", "4-state"],
                           "P(bacterial)": [float(np.exp(c2[0] - logsumexp(c2))), float(np.exp(cre[0] - logsumexp(cre))), post4[0]],
                           "P(fungal)": [float(np.exp(c2[1] - logsumexp(c2))), float(np.exp(cre[1] - logsumexp(cre))), post4[1]],
                           "P(both)": [0.0, 0.0, post4[2]], "P(neither)": [0.0, 0.0, post4[3]]})

    # profile-likelihood intervals (drop of 1.92 in log-likelihood) for P(neither), P(both)
    prof_rows, ci = [], {}
    for which, grid in ((3, np.round(np.arange(0, 0.41, 0.02), 2)), (2, np.round(np.arange(0, 0.061, 0.005), 3))):
        pl = profile(M, which, grid, rng, 4, th4)
        keep = grid[pl >= -r4.fun - 1.92]
        ci[STATE_NAMES[which]] = (keep.min(), keep.max()) if len(keep) else (np.nan, np.nan)
        prof_rows += [{"group": STATE_NAMES[which], "fixed at": v, "log-likelihood": x} for v, x in zip(grid, pl)]
    prof = pd.DataFrame(prof_rows)
    prof.to_csv(os.path.join(a.out, "profile.csv"), index=False)

    lr = 2 * (-r4.fun + r2.fun)
    lines = [f"Study {a.study}: {n_images:,} images; {int(N):,} subjects with all four tests (subjects whose images disagree "
             f"on a test: {conflicts}).",
             f"4-state vs the paper's 2-class model: likelihood-ratio statistic {lr:.1f} on 2 parameters (at the boundary, "
             "so a plain chi-square p-value is only approximate).",
             f"4-state groups: neither {g4[3]:.3f} (profile 95% interval {ci['neither'][0]:.2f}-{ci['neither'][1]:.2f}), "
             f"both {g4[2]:.3f} ({ci['both'][0]:.3f}-{ci['both'][1]:.3f}); about {g4[3] * N:.0f} and {g4[2] * N:.0f} subjects.",
             "Caution: with two tests per pathogen, 'neither' cannot be told apart from 'infected, but every test missed it' "
             "without a model; the three models make different assumptions about that, and fit compares them only on "
             "these four tests."]
    with open(os.path.join(a.out, "lca_states_report.md"), "w") as f:
        f.write("# Is there a genuine both / neither group?\n\nAggregates only.\n\n" + "\n".join(f"- {x}" for x in lines)
                + "\n\n## Fit\n\n" + md(fits, ".1f") + "\n\n## Groups\n\n" + md(grp)
                + "\n\n## Test sensitivity and specificity\n\n" + md(ss)
                + "\n\n## The all-negative patients under each model\n\n" + md(allneg)
                + "\n\n## Observed vs expected subjects per pattern\n\n" + md(obs, ".1f")
                + "\n\n## Profile likelihood (4-state)\n\n" + md(prof, ".3f") + "\n")
    print("\n".join(lines))
    print(fits.to_string(index=False))
    print(f"\nReport: {a.out}/lca_states_report.md")


if __name__ == "__main__":
    main()
