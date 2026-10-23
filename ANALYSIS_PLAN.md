# Analysis plan (written 2026-10-22, before the cross-validation runs)

Implemented in `src/analyze_full.py`. Every result below is computed for every stratum and reported
whatever its direction. Reference label: the LCA bin, binary (1 = bacterial, 0 = fungal).

## Models

| Key | Head | KL | Why it is included |
|---|---|---|---|
| sigmoid / soft, sigmoid / hard | sigmoid | – | The 2×2 |
| BD ICMLA / soft, BD ICMLA / hard | Beta-Dirichlet | ICMLA eq. 7 schedule | BD as published |
| BD Sensoy 1 / soft, BD Sensoy 1 / hard | Beta-Dirichlet | Sensoy form, weight 1, ramp over 10 epochs | Standard evidential setting (Sensoy et al.) |

The BD arms are fixed in advance, not chosen by test-set results. Other KL arms are reported in the
KL ablation (`src/analyze_kl.py`). With cross-validation, a data-driven BD choice may be added only if
it is made on each fold's validation set.

Score: p = (p_bac + 1 − p_fung) / 2. Uncertainty signals: entropy of p (every model); low evidence,
(α_bac + β_bac + α_fung + β_fung) / 2 (BD); a two-label binary conformal set.

## Strata

| Stratum | Definition | Why |
|---|---|---|
| All | Every test image | The pooled view the abstract reported |
| All, enrolment only | Excluding 3-week, 3-month and 12-month photos | Follow-ups inflate performance |
| Single-etiology trials | MUTT Pilot, MUTT1, MUTT2, SCUT, CLAIR I, CLAIR II; enrolment only | Labels set by study design; confounded with trial |
| MADURAI | The mixed-etiology study | Same setting, labels from tests: the clinical question |
| MADURAI, confident label | Posterior ≤ 0.1 or ≥ 0.9 | Concordant evidence |
| MADURAI, ambiguous label | 0.1 < posterior < 0.9 | Where soft and hard targets differ |
| MADURAI, one pathogen + | Positive tests for one pathogen only (participant level) | Comparison group |
| MADURAI, no positive test | Tests recorded, none positive | Least informative tests |
| MADURAI, co-positive | A bacterial-positive and a fungal-positive test | Discordant tests |

The last three need the raw test columns (the CSV), so they are computed on the server.
Non-conformal metrics use every test image in the stratum; conformal metrics use the evaluation part
only (the calibration part sets the threshold).

## Hypotheses and measures

**H1. Soft labels help with uncertainty** (soft − hard, same head and KL arm, paired).
- H1a calibration: Brier, Murphy reliability (10 bins), calibration error (ECE, 10 bins). **Primary.**
- H1b error detection: AUROC of entropy (and evidence) for ranking wrong predictions above right ones.
- H1c selective prediction: accuracy on the 25%, 50% and 75% most confident images; area under the
  risk–coverage curve (AURC).
- H1d label-uncertainty tracking (images with a posterior): AUROC of entropy for ambiguous labels;
  Spearman of model entropy with label entropy. **Primary.**
- H1e conformal efficiency: two-label rate and coverage at α = 0.1.
- H1f fidelity to the posterior: mean |p − posterior|; Brier against the posterior (added 2026-10-22).
- Descriptive: mean p(bacterial) and share called bacterial by reference class, per stratum (added
  2026-10-22, to show whether a soft–hard difference is a shift of the operating point).
- Also reported: discrimination (AUROC), accuracy.

**H2. Uncertainty differs by stratum** (per model; contrasts in mean uncertainty):
MADURAI − single-etiology; MADURAI confident − single-etiology (setting vs label);
MADURAI ambiguous − MADURAI confident; no positive test − one pathogen +; co-positive − one pathogen +.
Per-study means are also tabulated.

**H3. Uncertainty tracks performance:** per stratum, error detection and accuracy by entropy tertile
(tertiles within the stratum); pooled vs within-study error detection (a pooled value above the
within-study values means between-study differences drive it); per-study mean uncertainty vs accuracy
(studies with ≥ 10 images; descriptive).

**H4. How the signals relate** (per model, per stratum): Spearman of entropy with evidence and with
conformal set size; sigmoid entropy vs BD evidence (same labels); overlap of the 25% most uncertain
images by entropy and by evidence (Jaccard); combined filter (sigmoid conformal single label + BD
evidence above the calibration 67th percentile) vs each alone at the same coverage.

## Statistics and decision rules

- 95% intervals: bootstrap over participants (images within a participant are not independent),
  1,000 resamples, the same resamples for every metric in a stratum.
- Two-sided bootstrap p-values for paired differences; Holm correction across the primary family
  (H1a Brier and H1d ambiguous-label AUROC, MADURAI stratum, every soft/hard pair).
- Verdicts: "holds" (claimed direction, 95% interval excludes 0), "inconclusive", "reversed".
- With cross-validation: predictions from all folds are pooled (each image predicted once, by a model
  that never saw it); conformal is calibrated within each fold; per-fold values of AUROC and Brier are
  also reported to show training variation.

## Conformal

Binary LAC, α = 0.1, textbook rank k = ⌈(n + 1)(1 − α)⌉. Calibrated at participant level: one image
per calibration participant, 1,000 random draws, median threshold. The draws are made once and shared
by every model (amended 2026-10-22: with 200 draws taken from one shared random stream, the two-label
rate of a model moved by up to 0.03 depending on which other runs were loaded). Every random stream
(conformal draws, each stratum's bootstrap) is seeded by name, so a result does not depend on which
other runs or strata are in the analysis.

## Amendment, 23 Oct 2026: cross-validation, the head gap, and test-supervised baselines

Written before any cross-validation run.

**Design.** 5 folds grouped by pID, stratified by study × label, seed 42; every image predicted
once; per-fold calibration/evaluation split 20/80 as before. Setups: sigmoid soft/hard, BD ICMLA
soft/hard, BD Sensoy 1 soft/hard (all as above), plus the smear- and culture-supervised baselines.
Hypotheses H1–H4, strata and statistics as above, on the pooled out-of-fold predictions.

**Reading rule** (proposed; to be confirmed before results): a soft − hard difference counts if it
has the same sign in at least 4 of 5 folds and the pooled interval excludes 0. Per-fold AUROC and
Brier are in the report.

**H5, the gap between the two outputs, d = p_bac + p_fung − 1 (exploratory).** Generated on the
v2 single split, so tested twice: on all images, and leaving out the 837 v2 test images
(`--exclude_rows_from`); only the second counts as a test. Expected directions:
- errors have lower d (BD): entropy + gap beats entropy for error detection (logistic regression
  fitted on each fold's calibration images, AUROC on evaluation images);
- MADURAI negative-test patients have lower d than one-pathogen-positive patients;
- MADURAI co-positive patients have higher d than one-pathogen-positive patients;
- 12-month photos have lower d than enrolment photos of the same studies.

**H6, the two outputs as a joint.** For each MADURAI image with both tests of a pair (cultures;
smears): the product rule P(both) = p_bac × p_fung against the observed both-positive share, with a
participant-bootstrap interval for the difference; AUROC of P(both) for both-positive and of
P(neither) for both-negative. For the smear- and culture-supervised baselines, observed below
expected means the two tests stay negatively dependent even given the image, i.e. the outputs
should not be combined as independent. For the LCA-label models the comparison is descriptive.

**Test-supervised baselines.** Targets: Gram and KOH (smear) or blood agar and PDA (culture), each
its own sigmoid output, binary cross-entropy over the tests done, results carried across a
participant's images; supplement settings (smear 11 epochs from layer 753; culture 22 epochs from
layer 559). Scored against the LCA bin like every other model (score (p_bac + 1 − p_fung) / 2), and
each output against its own test.
