# Redo on the paper's own data (from 23 Oct 2026)

Travis sent the LCA posteriors for every image used in the original paper, as two files:
`training images.csv` and `test images.csv`, with columns `subject, Drive_image_path, fung_smear,
bac_cult_positive, fung_cult_positive, bac_smear, fung_post_prob, bac_post_prob`.

His hard-label rule: bacterial = 1 if the bacterial smear **or** the bacterial culture is positive,
otherwise 0; the same for fungal. So hard labels are per pathogen and need not be complementary:
[1, 1] (both positive) and [0, 0] (neither positive, or no test recorded) can occur.

This branch (`redo`) carries all the earlier code; the work on the paper's data starts from the EDA.

## 0. Set up the folder (once)

The CSVs are already in `/scratch90/chris_/ik_redo`; the repository goes in the same folder. CSVs are
git-ignored, and so is `results/`.

```bash
cd /scratch90/chris_/ik_redo
git init -q
git fetch /scratch90/chris_/ik_uq_redo.bundle redo:redo
git checkout -q redo
git remote add origin https://github.com/ClarkQTIM/ik_uq.git
git push -u origin redo
git status --short        # should print nothing: the CSVs are ignored
```

Later updates arrive as bundles: `git pull /scratch90/chris_/<update>.bundle redo && git push`.

## 1. EDA (a minute or two)

```bash
cd /scratch90/chris_/ik_redo
python src/eda_paper_data.py --train "training images.csv" --val "validation images.csv" \
    --test "test images.csv" --out results/eda
```

The validation file arrived on 27 Oct; `--val` is optional, so the earlier two-file run still works.

It also compares against our study CSV (default
`/persist/QTIM/Active/26-0096/Corneal_Ulcer_Image_Sets/Data/corneal_ulcer_study_data_7_24_2025_lambda.csv`;
`--old_csv none` skips that) and checks the matched images exist under `/pacs/OSHU/Corneal_Ulcer_Image_Sets`.

Send back `results/eda/eda_report.md`: aggregates only, no subject IDs, paths or image names.

## What the EDA answers

| Question (from the brief) | Section |
|---|---|
| Which images and participants: 4,471 / 2,100? Follow-ups? | Answers; 0, 1, 8 |
| Is the paper's split grouped by participant (subject, and pID)? | 2, 8 |
| Did trial participants get test results (MUTT smears, SCUT cultures)? | 3, 8 |
| Two-class posteriors (sum to 1, depend on all four tests) or per pathogen? | 4 |
| What does "all four tests negative" get (our CSV 0.688, naive 0.946)? | 4 (pattern table) |
| What do Travis's hard labels give: how many [1,1] and [0,0], and where? | 5 |
| Co-positivity vs independence, per study | 6 |
| Where soft and hard targets differ (where soft vs hard can show anything) | 7 |
| Paper vs our CSV: tests, posteriors, bins, images left out, files on disk | 8 |

## 2. Label table and image files

```bash
python src/paper_labels.py --train "training images.csv" --val_csv "validation images.csv" \
    --test "test images.csv" --out results/labels --img_roots /pacs/OSHU/Corneal_Ulcer_Image_Sets /persist/QTIM/Active/26-0096/Corneal_Ulcer_Image_Sets
```

Writes `results/labels/paper_labels.csv` (identifiers; stays here) and `labels_report.md` (send).
As of 26 Oct: 2,883 of 3,813 images found; 930 trial enrolment photos (MUTT1 547, SCUT 220, MUTT2 163)
exist only on the paper's Google Drive; asked Travis. When they arrive, put them under an image root
and rerun this step; nothing else changes.

## 3. What each analysis gets (seconds)

```bash
python src/analysis_sets.py --labels results/labels/paper_labels.csv --out results/analysis_sets
```

Two analyses (decided 28 Oct), both with two independent outputs and Travis's posteriors as soft labels:

- **LCA analysis:** all images; hard = posterior > 0.5 (Travis's rule); scored against those hard labels.
- **Culture analysis:** only images with at least one culture (MADURAI, SCUT); hard = the cultures (blood agar
  for bacterial, PDA for fungal); scored against the cultures.

For each, the report gives images and patients by split and study, hard-label combinations, where soft and
hard labels differ, the evaluation set, cultures vs LCA labels, and what the missing photos take away. Each
comes twice: **full** (everything in the paper's files) and **provisional** (images on our server now).
Send back `results/analysis_sets/analysis_sets.md`.

## 4. Is there a genuine both / neither group? (about a minute)

```bash
python src/lca_states.py --labels results/labels/paper_labels.csv --out results/lca_states
```

Fits the paper's two-class LCA, the same with a random effect (tests correlated within a class), and a
two-pathogen model (bacterial only / fungal only / both / neither) to the four tests of the MADURAI subjects.
Send back `results/lca_states/lca_states_report.md`.

## 5. Train: provisional now, full when the photos arrive

```bash
python src/model.py --labels_csv results/labels/paper_labels.csv --label_type culture_hard --subset culture --dry_run
DRY_RUN=1 OUT_ROOT=results/provisional scripts/launch_redo.sh      # the 10-run plan
OUT_ROOT=results/provisional scripts/launch_redo.sh                # go (cuda:0 + cuda:1)
```

| Setup | Analysis | Head / soft or hard |
|---|---|---|
| A_cul, C_cul | culture (`--subset culture`) | sigmoid / paper_soft, culture_hard |
| B_cul, D_cul | culture | BD / paper_soft, culture_hard |
| A_lca, C_lca | LCA | sigmoid / paper_soft, paper_hard |
| B_lca, D_lca | LCA | BD / paper_soft, paper_hard |
| SM, CU | the paper's baselines | sigmoid / smear, culture (BCE, the supplement's recipes) |

Soft and hard runs in each analysis share everything but the labels (paper recipe, rescaled MSE or the
evidential loss). Every run's `test_metrics.json` scores each output against both references where it can:
`per_pathogen_vs_hard_labels` and `per_pathogen_vs_culture` (all test images, MADURAI, other studies,
evaluation set).

**Provisional vs full.** The label table only uses images with a file, so a run today is the provisional
version. The culture analysis is nearly final already (every MADURAI photo is here; SCUT is short). When
Travis's photos arrive: put them under an image root, rerun step 2 (`paper_labels.py`), then
`OUT_ROOT=results/full scripts/launch_redo.sh`. Nothing else changes.

Paper recipe throughout; with the validation file, validation loss is logged each epoch but the final epoch
is kept, as the paper did. About 5 GPU-hours for all ten (about 2.5 on two GPUs). If CU runs out of memory
(two stages train): `EXTRA="--batch 16 --accum 2" SETUPS="CU" OUT_ROOT=... scripts/launch_redo.sh`.

## After the EDA (plan, to settle with the EDA in hand)

1. Labels from the paper's files: soft = the paper's posteriors; hard = Travis's rule; the paper's
   train/test split (validation carved from train by participant); images located through our CSV.
2. Replicate the paper's 2×2 on its own split with the paper recipe (direct comparison with the
   published numbers), scoring each pathogen as the paper did and the binary task as we did.
3. Cross-validation on the paper's image set, with the analyses already built (H1–H6), if the
   replication holds up.
