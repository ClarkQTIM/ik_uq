# Latent Supervision for Infectious Keratitis Classification
## Paper Summary and Research Extensions

**Reference:** Yazbeck et al., "Latent Supervision: A Method for Improved Performance and Calibration of Machine Learning Classification Models in Ophthalmology," *Ophthalmology Science*, 2026.

---

## The Problem the Paper Solves

Training a machine learning classifier requires ground-truth labels. In medicine, ground truth is rarely clean. For infectious keratitis — bacterial or fungal corneal ulcers — the available labels come from smear and culture results, each with well-documented limitations: Gram stain is only ~23% sensitive for bacteria; KOH prep is ~94% sensitive for fungi; cultures are ~50% sensitive overall. No single test is definitive.

Standard supervised learning treats these imperfect test results as if they were true labels, producing models that are confidently wrong — they learn the noise in the labels as if it were signal. The result is classifiers that are poorly calibrated: a model that predicts "87% bacterial" may only be correct 60% of the time at that confidence level.

---

## The Latent Supervision Solution

Yazbeck et al. propose a three-step pipeline that replaces hard binary labels with principled probabilistic estimates before any model training occurs.

**Step 1 — Latent Class Analysis (LCA).** Rather than assuming any test is correct, LCA treats the true disease class as an unobserved (latent) variable and estimates the sensitivity and specificity of each diagnostic test simultaneously, without a gold standard. Applied to the four available tests (Gram stain, KOH smear, bacterial culture, fungal culture), the winning model is a two-class LCA with random effects and variable loadings — meaning it also accounts for patient-level variation in test performance. The estimated sensitivities confirm what clinicians know: KOH is highly sensitive for fungi (94%), Gram stain poorly sensitive for bacteria (23%).

**Step 2 — Bayesian Posterior Estimation.** Using the LCA-estimated test parameters, a Bayesian update computes for each patient the posterior probability of bacterial infection (`bac_post_prob`) and fungal infection (`fung_post_prob`) given their specific combination of test results. These are soft labels in [0, 1] — a patient with three positive fungal tests might receive `fung_post_prob = 0.9997`; a patient with ambiguous results might receive `fung_post_prob = 0.61`. Crucially, these two values are computed independently and need not sum to 1 — a patient can have elevated posterior probability for both pathogens (polymicrobial infection).

**Step 3 — Training with Soft Labels.** A single EfficientNetB7 model (64M parameters, pretrained on ImageNet) is trained using these posterior probabilities as targets, replacing the hard binary labels that would have come from individual test results. The loss function is rescaled Mean Squared Error (MSE × 10) rather than cross-entropy. This choice is deliberate: cross-entropy penalises predictions that don't reach 0 or 1, which would force the model to ignore the genuine uncertainty in soft labels. MSE naturally accommodates continuous targets — a prediction of 0.73 against a label of 0.73 incurs zero loss.

Training uses two phases: first with the backbone frozen (training only the classification head), then with progressive unfreezing of the deeper backbone layers. The model outputs two independent sigmoid probabilities: `[p_bac, p_fung]`.

**A note on independence.** The two outputs are computed via separate sigmoid activations — not softmax — meaning `p_fung ≠ 1 − p_bac`. A softmax formulation would force the outputs to sum to 1, imposing a zero-sum competition between pathogens and making it impossible to express polymicrobial infection. Sigmoid keeps the two predictions independent: a patient can have high `p_bac` and high `p_fung` simultaneously. This is the architecturally correct choice for a multilabel task.

In practice, however, the label distribution means most cases behave almost like a binary choice. Six of the seven studies are single-etiology: SCUT and CLAIR I are bacterial only (`p_bac ≈ 1, p_fung ≈ 0`); MUTT1, MUTT2, MUTT Pilot, and CLAIR II are fungal only (`p_fung ≈ 1, p_bac ≈ 0`). Only **MADURAI** contains genuine mixing, where ambiguous diagnostic test results produce posteriors like `p_bac = 0.73, p_fung = 0.61` simultaneously. The independence of the two sigmoid outputs matters most for exactly these cases.

This is also why our joint Dirichlet extension failed: it forces `p_bac + p_fung = 1` through the probability simplex, which is approximately true for single-etiology studies but wrong for MADURAI's polymicrobial cases. The Beta-Dirichlet head — two independent Beta distributions, one per pathogen — preserves the correct multilabel geometry.

---

## What the Paper Reports

The paper compares three separately trained versions of the same architecture, differing only in their training labels:

**A note on evaluation labels.** AUROC, F1, and accuracy are all computed against binarized LCA posteriors — not confirmed culture or smear results. Specifically, `true_bac = (bac_post_prob > 0.5)` and `true_fung = (fung_post_prob > 0.5)`. This means evaluation measures how well the model recovers the LCA's own probability estimates, not true underlying disease status. Since there is no gold standard, this is the best available reference. The same caveat applies to all three models in the comparison — the evaluation is imperfect but consistent across models.

| Model | Labels used | BAC AUROC | FUNG AUROC |
|---|---|---|---|
| Culture-supervised | Hard labels from cultures | 0.87 | 0.78 |
| Smear-supervised | Hard labels from smears | 0.79 | 0.85 |
| **Latent supervision** | **Soft LCA posteriors** | **0.86** | **0.86** |

The latent supervision model performs consistently well across both tasks — neither over-optimised for bacterial (like culture-supervised) nor for fungal (like smear-supervised). More importantly, Figure 2 shows that the latent supervision model's calibration curves follow the diagonal more closely than either supervised baseline, meaning its predicted probabilities are more trustworthy as probability estimates.

---

## Our Reproduction

We reproduced the latent supervision model in Python (PyTorch + timm), using the precomputed posterior probabilities already present in the dataset CSV. Our reproduction achieves:

| Task | Our result | Paper |
|---|---|---|
| BAC AUROC | **0.892** | 0.86 |
| FUNG AUROC | **0.897** | 0.86 |

The small improvement over the paper likely reflects minor differences in data filtering, the pretrained checkpoint used (`tf_efficientnet_b7.ra_in1k`), and random seed.

---

## Our Extensions

The paper produces well-calibrated probability outputs but provides no formal uncertainty quantification — there is no mechanism to flag cases where the model genuinely does not know the answer, and no statistical guarantee on the prediction sets. We address this with two complementary extensions.

### Extension 1 — Beta-Dirichlet Output Head

#### Architecture Comparison

The sigmoid and Beta-Dirichlet heads share the same EfficientNetB7 backbone and the same dense trunk (512 → 32 neurons with ReLU and dropout). They differ only in the final layer:

**Sigmoid head (paper reproduction):**
```
Linear(32, 2) → Sigmoid → [p_bac, p_fung]
```
Two output neurons, one per pathogen. Each output is squashed into (0, 1) by the sigmoid function and interpreted directly as a probability.

**Beta-Dirichlet head (our extension):**
```
Shared trunk: Linear(32, 32) → ReLU

Bac head:  Linear(32, 2) → Softplus → [α_bac, β_bac]
Fung head: Linear(32, 2) → Softplus → [α_fung, β_fung]
```
Four output neurons total — two per pathogen. Two separate linear layers (one per pathogen) each produce a pair of Beta distribution parameters. Softplus is used instead of ReLU to ensure α and β are strictly positive — exact zeros would cause log(0) in the KL term of the loss function.

The predicted probability is recovered as `p_bac = α_bac / (α_bac + β_bac)`, equivalent in range and interpretation to the sigmoid output. The additional information is `α+β` — total evidence — which has no equivalent in the sigmoid formulation.

#### Loss Function Comparison

**Sigmoid loss — Rescaled MSE (paper):**
```
L = 10 × MSE(y_true, p_hat)
  = 10 × mean[(p_bac_true - p_bac_pred)² + (p_fung_true - p_fung_pred)²]
```
The ×10 scaling compensates for MSE gradients being smaller in magnitude than cross-entropy. The loss teaches the model one thing: make your predictions close to the LCA posterior probabilities.

**Beta-Dirichlet loss — Dirichlet MSE + annealed KL:**
```
L = MSE_term + λ_KL × KL_term

MSE_term = Σ_k [(y_k - α_k/α_0)² + (α_k/α_0)(1 - α_k/α_0)/(α_0 + 1)]

KL_term  = KL[ Dir(α_tilde) || Dir(1,...,1) ]
           where α_tilde removes evidence for the true class

λ_KL(n)  = (λ_0 / K) × (n / N)   [linearly annealed from 0]
```
The MSE term works like the sigmoid loss — match the predicted probability to the label. The KL term adds a regularizer that penalises overconfident predictions when labels are uncertain, pulling toward a uniform prior `Dir(1,1)`. The annealing schedule starts λ_KL at zero and increases it linearly over training, allowing the model to first learn to predict probabilities correctly before being penalised for overconfidence.

The loss teaches the model two things simultaneously: (1) make your predictions close to the LCA posteriors, and (2) don't accumulate strong evidence when the label itself is uncertain. This is why Beta-Dirichlet evidence correlates with conformal set size — both are measuring, in different ways, how much diagnostic information the image contains.

We replace the sigmoid output layer with two independent Beta distributions, one per pathogen, yielding four parameters per image: `[α_bac, β_bac, α_fung, β_fung]`. The predicted probability for each pathogen is `α/(α+β)`, identical to sigmoid in interpretation but now accompanied by total evidence `α+β`. High evidence means the model has accumulated strong signal from image features; low evidence means the model is genuinely uncertain regardless of which direction it leans.

This distinction — between "uncertain" and "confidently borderline" — is clinically meaningful and impossible to extract from a sigmoid output alone. A sigmoid score of 0.5 could mean "I've looked carefully and it's genuinely 50/50" or "I've seen nothing useful in this image." The Beta parameters separate these cases.

The Beta-Dirichlet model achieves BAC AUROC 0.849 and FUNG AUROC 0.838 — approximately 5pp below sigmoid — trading a small amount of discriminative performance for genuine uncertainty semantics.

### Extension 2 — Conformal Prediction (LAC)

We apply split conformal prediction (Least Ambiguous Classifier, Angelopoulos & Bates 2022) post hoc to the sigmoid model. Rather than a single probability per pathogen, the model now outputs a **prediction set** — the set of classes that cannot be ruled out at a given confidence level. With α = 0.10, the guarantee is:

> P(true label ∈ prediction set) ≥ 90%

This is a distribution-free, finite-sample guarantee — it holds regardless of model quality or data distribution. One important caveat: the "true label" used to assess coverage is the binarized LCA posterior (`bac_post_prob > 0.5`, `fung_post_prob > 0.5`), not confirmed disease status. The coverage guarantee therefore reads: "the prediction set contains the LCA-derived label at least 90% of the time" — not "contains the true disease at least 90% of the time." These are close but not identical. The size of the prediction set (CPSL — Conformal Prediction Set Length) is the key clinical output: CPSL = 1 means confident, CPSL = 4 means maximally uncertain.

We evaluate three LAC variants: Vanilla (marginal coverage), Group-Balanced (per-study coverage), and Class-Conditional (per-pathogen coverage), across both 2-class (independent binary) and 4-class (joint label space) framings.

### Combined Analysis

The most novel finding is the convergence of the two independent frameworks. The Spearman correlation between Beta-Dirichlet total evidence and conformal set size is ρ = −0.675 (p < 0.0001) for bacterial keratitis — meaning cases the Beta-Dirichlet model finds uncertain (low α+β) are the same cases conformal identifies as ambiguous (large prediction set). This agreement was not guaranteed: one framework is learned from image features during training; the other is computed post hoc from calibration statistics. Their convergence validates both approaches simultaneously.

Combining the two signals in a joint stratification (BD evidence tier × conformal set size) yields a clinically actionable uncertainty landscape: cases flagged as confident by both frameworks achieve 98% accuracy, while cases flagged as uncertain by both achieve only 60% accuracy — a 38 percentage point gap.

The full stratification across BD evidence tier and conformal set size is shown below:

| BD Evidence | CPSL=1 | CPSL=2 | CPSL=3 | CPSL=4 |
|---|---|---|---|---|
| High (e > 6.8) | **98%** | 100% | 60% | 50% |
| Mid (4.6 < e ≤ 6.8) | 92% | 67% | 73% | 53% |
| Low (e ≤ 4.6) | 79% | 47% | 78% | **60%** |

The top-left cell (High BD + CPSL=1, n=196) represents cases where both frameworks agree the model is confident — 98% accurate. The bottom-right region (Low BD + CPSL=4, n=98) represents cases where both agree the model is uncertain — 60% accurate. Cases in the middle represent genuine disagreement between the two frameworks, and are clinically the most interesting to investigate.

---

## Summary of Contributions

| | Paper | Our work |
|---|---|---|
| Labelling | LCA + Bayes soft labels | Reproduced (used precomputed) |
| Architecture | EfficientNetB7, sigmoid | Reproduced + Beta-Dirichlet head |
| Loss | Rescaled MSE | Reproduced + Dirichlet loss |
| Uncertainty | None | Conformal LAC (3 variants, 2 framings) |
| Calibration | Calibration curves | + Conformal coverage guarantees |
| Clinical utility | AUROC/F1/Accuracy | + Selective classification framework |
| Novel finding | — | Cross-framework convergence (ρ = −0.675) |