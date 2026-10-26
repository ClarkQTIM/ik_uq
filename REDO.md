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
python src/eda_paper_data.py --train "training images.csv" --test "test images.csv" --out results/eda
```

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
python src/paper_labels.py --train "training images.csv" --test "test images.csv" --out results/labels \
    --img_roots /pacs/OSHU/Corneal_Ulcer_Image_Sets /persist/QTIM/Active/26-0096/Corneal_Ulcer_Image_Sets
```

Writes `results/labels/paper_labels.csv` (identifiers; stays here) and `labels_report.md` (send).
As of 26 Oct: 2,883 of 3,813 images found; 930 trial enrolment photos (MUTT1 547, SCUT 220, MUTT2 163)
exist only on the paper's Google Drive; asked Travis. When they arrive, put them under an image root
and rerun this step; nothing else changes.

## 3. Train on the paper's split

```bash
python src/model.py --labels_csv results/labels/paper_labels.csv --label_type paper_soft --dry_run
DRY_RUN=1 OUT_ROOT=results/paper_split scripts/launch_redo.sh     # the 10-run plan
OUT_ROOT=results/paper_split scripts/launch_redo.sh               # go (cuda:0 + cuda:1)
```

| Setup | Head / labels | Scored |
|---|---|---|
| A_paper, C_paper | sigmoid / paper_soft, paper_hard | per pathogen vs Travis's hard labels (the paper's metric), and binary |
| B_paper, D_paper | BD / paper_soft, paper_hard | same |
| A_2c, C_2c | sigmoid / twoclass_soft, twoclass_hard | binary vs the two-class label |
| B_2c, D_2c | BD / twoclass_soft, twoclass_hard | same |
| SM, CU | sigmoid / smear, culture (BCE) | each output vs its own test |

Paper recipe throughout; trains on the full paper training split (no validation set: the recipe keeps
the final epoch; `--val_share 0.15` would hold out 15% of training subjects to watch validation loss).
Test split divided 20/80 into calibration/evaluation for conformal. Each run's `test_metrics.json`
has `per_pathogen_vs_hard_labels` (all test images, MADURAI, other studies).
About 5 GPU-hours in all (about 2.5 on two GPUs). If CU runs out of memory (two stages train):
`EXTRA="--batch 16 --accum 2" SETUPS="CU" OUT_ROOT=results/paper_split scripts/launch_redo.sh`.

## After the EDA (plan, to settle with the EDA in hand)

1. Labels from the paper's files: soft = the paper's posteriors; hard = Travis's rule; the paper's
   train/test split (validation carved from train by participant); images located through our CSV.
2. Replicate the paper's 2×2 on its own split with the paper recipe (direct comparison with the
   published numbers), scoring each pathogen as the paper did and the binary task as we did.
3. Cross-validation on the paper's image set, with the analyses already built (H1–H6), if the
   replication holds up.
