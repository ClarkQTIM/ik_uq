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
