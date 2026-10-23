# v2 runs: fixed labels, paper recipe

## What changed from the runs in `runs/`

- **Labels** (`src/labels.py`, used by `src/model.py`): soft target = LCA posterior where one exists,
  otherwise the LCA bin; hard target = bin; every model is scored against the bin, as a binary
  bacterial-vs-fungal task. The old soft-label runs trained on the −1 "no posterior" code for about
  half the images (audit C1).
- **Recipe** (`--recipe paper`, the default), from the supplement §3 and the R demo:
  EfficientNetB7, 600×600; only the last stage (Keras layer 753 = timm `blocks[6]`), the top conv and
  the head train; backbone BatchNorm frozen; Adam, lr 1e-5, clipnorm 1; dropout 0.3; batch 32;
  15 epochs in one phase; no early stopping; the final-epoch model is the model.
  Augmentation as in the demo (rotation ±54°, shift ±10%, flips, contrast ±10%, reflect fill).
  Keras defaults matched where cheap: stochastic depth 0.2, head BatchNorm eps/momentum,
  Adam epsilon 1e-7, Glorot head initialisation.
- **Split** grouped by `pID` (was `subject`, which let 5 participants straddle splits).
  So the test set is not identical to the old one.
- **Fails loudly**: missing images stop the run before training, unreadable images raise, NaN loss
  raises. Outputs refuse to go into any folder called `runs`.
- Follow-up photos stay in training (`--exclude_followups_train` removes them); every test image is
  tagged with its visit so evaluation can filter later.

## 0. Put the code on the server

```bash
cd /scratch90/chris_
git clone --branch main /path/to/ik_uq_2026-10-21.bundle ik_uq
cd ik_uq
git fetch origin v2-paper-recipe:v2-paper-recipe
# check the baseline matches the code that produced the old runs (no output = identical)
diff -r /scratch90/chris_/infectious_ker_ulcer/src src | head
git remote set-url origin https://github.com/ClarkQTIM/ik_uq.git
git push -u origin main v2-paper-recipe
git checkout v2-paper-recipe
```

If the `diff` prints anything, the server code changed after 15 Oct; send me the differences
before running.

## 1. Dry run (labels and image files only, a minute or two)

```bash
python src/model.py --dry_run
```

Expect about 5,385 labelled rows, 2,534 with a posterior and 2,851 using the bin, a per-study table,
the split sizes, and "All N image files found". It stops with a list if any image is missing.

## 2. Smoke run on a GPU (a few minutes)

```bash
python src/model.py --smoke --head beta_dirichlet --label_type soft \
    --out_root /scratch90/chris_/ik_uq_runs --device cuda:0
```

One epoch on 48 images, end to end, including `test_predictions.npz`. It prints the peak GPU memory
after the epoch. Delete the `SMOKE_...` folder afterwards.

## 3. Launch the four cells

```bash
OUT_ROOT=/scratch90/chris_/ik_uq_runs scripts/launch_paper_2x2.sh
```

`cuda:0` runs A (sigmoid, soft) then C (sigmoid, hard); `cuda:1` runs B (BD, soft) then D
(BD, hard). Runs survive logging out. Follow with the `tail -f` commands it prints, and `nvidia-smi`.

If batch 32 does not fit in GPU memory, stop both chains and relaunch with gradient accumulation:

```bash
EXTRA="--batch 16 --accum 2" OUT_ROOT=/scratch90/chris_/ik_uq_runs scripts/launch_paper_2x2.sh
```

## 4. What each run writes

`model.pt` (final epoch: the model), `best_val_model.pt` (reference only), `train_log.csv`,
`test_metrics.json`, `test_predictions.npz`, `split.csv`, `label_summary.csv`, `config.json`
(including the git commit).

`test_predictions.npz` and `test_metrics.json` contain no identifiers (`csv_row` is a row number,
`participant` an integer code), so those, plus `train_log.csv` and `config.json`, are what to send
back for the analysis.

## Keeping the two copies in step

- Code from Claude arrives as a git bundle: `git pull /path/to/update.bundle v2-paper-recipe`,
  then `git push`.
- Changes made on the server: commit them and send
  `git bundle create server.bundle <last shared commit>..HEAD` (or the changed files).

## KL ablation (BD cells)

The published ICMLA schedule gives the KL term a weight of at most about 5e-5, so the B and D runs
were effectively trained without it. The ablation adds four arms for both B (soft) and D (hard),
weight ramping as `kl_max * min(1, epoch / 10)`:

| Arm | KL form | kl_max |
|---|---|---|
| S1 | Sensoy (KL on alpha_tilde; the code's default form) | 1 (Sensoy et al.'s setting) |
| S0.1 | Sensoy | 0.1 |
| E1 | ICMLA eq. 5 (KL on alpha) | 1 |
| S0.01 | Sensoy | 0.01 |

```bash
OUT_ROOT=/scratch90/chris_/ik_uq_runs DEVICE=cuda:1 scripts/launch_kl_ablation.sh
```

Eight runs in sequence (about 45 minutes each). Then:

```bash
R=/scratch90/chris_/ik_uq_runs
python src/analyze_kl.py --A $R/A_sigmoid_soft_paper_* --C $R/C_sigmoid_hard_paper_* \
    --bd $R/B_beta_dirichlet_soft_paper_2026-10-21_16-14 $R/D_beta_dirichlet_hard_paper_2026-10-21_16-14 \
         $R/[BD]_kl*_paper_* --out $R/kl_ablation
```

Criteria fixed before the runs: (a) on enrolment photos and within MADURAI, evidence flags errors
at least as well as the model's own entropy; (b) sigmoid conformal singleton + high BD evidence
beats sigmoid confidence alone at the same coverage. AUROC, Brier and calibration error are
reported for every arm, and every arm is reported.

## Cross-validation (v2.1)

5 folds grouped by pID and stratified by study × label (`labels.split_cv`): each participant is in
the test set of exactly one fold, so every image gets one prediction from a model that never saw
it. Within a fold, 15% of all participants (from the other folds) are validation, the rest train,
and the test fold is split 20/80 into calibration/evaluation as before. The same seed gives the
same folds for every setup, so soft and hard (and every other pair) are compared on identical
participants.

Eight setups × 5 folds = 40 runs:

| Setup | Head / labels | Notes |
|---|---|---|
| A, C | sigmoid / soft, hard | the 2×2 |
| B, D | BD / soft, hard | ICMLA KL as published |
| BS1, DS1 | BD / soft, hard | Sensoy KL, weight 1 |
| SM | sigmoid / smear | Gram and KOH as two separate targets; BCE; 11 epochs (supplement) |
| CU | sigmoid / culture | blood agar and PDA; BCE; 22 epochs; trains from Keras layer 559 (timm `blocks[5]`) |

SM and CU are the paper's test-supervised baselines, which were never trained in the old code.
Their targets are the raw tests (carried across a participant's images), so [1, 1] and [0, 0]
occur; only images with at least one of the two tests train, and every test image is predicted.

```bash
cd /scratch90/chris_/ik_uq
git pull /scratch90/chris_/ik_uq_cv.bundle v2-paper-recipe && git push
DRY_RUN=1 OUT_ROOT=/scratch90/chris_/ik_uq_cv scripts/launch_cv.sh     # print the plan
python src/model.py --dry_run --n_folds 5 --fold 0                     # split sizes for one fold
OUT_ROOT=/scratch90/chris_/ik_uq_cv scripts/launch_cv.sh               # start (two GPUs)
```

Runs go fold by fold, round-robin over `DEVICES` (default `cuda:0 cuda:1`); a failed run doesn't
stop its chain. Rough time: about 45 minutes per LCA-label run, 20 for SM, 40 for CU, so about
5.5 GPU-hours per fold, 27 in all (about 14 on two GPUs). For one GPU: `DEVICES="cuda:1"`.
A subset: `SETUPS="A C" FOLDS="0"`.

Analysis, once runs finish (partial folds work too; each setup is pooled over the folds present):

```bash
C=/scratch90/chris_/ik_uq_cv; R=/scratch90/chris_/ik_uq_runs
python src/analyze_full.py --runs $C/*_f[0-9]_* --out $C/analysis_all
# the same, leaving out the 837 images of the v2 single-split test set (hypotheses generated there)
python src/analyze_full.py --runs $C/*_f[0-9]_* --out $C/analysis_new_images \
    --exclude_rows_from $R/A_sigmoid_soft_paper_2026-10-21_16-14
```

No `--csv` is needed: v2.1 runs carry each image's test results. Send back `report.md` from each.
