# Specialist encoder baseline (gz-evo / Zoobot)

The specialist baseline fine-tunes pretrained image encoders (MaxViT / SoViT / ConvNeXt —
referred to as "CNN" in the results tables for brevity, though not all are CNNs). It is
**gz-evo** by Mike Walmsley et al., vendored under `gz-evo/` with its original `LICENSE` (MIT)
preserved.

- **Upstream:** https://github.com/mwalmsley/gz-evo, commit `8a0bdca` (2025-07-29), built on
  Zoobot (https://github.com/mwalmsley/zoobot).
- **What we use:** `gz_evo/downstream/finetune.py` to fine-tune a per-task pretrained encoder
  (MaxViT-Tiny / SoViT-400m / ConvNeXt-Base / MaxViT-Base, all `baseline-encoder-regression-*`)
  on the same labelled subsets as the VLMs. Label-efficiency curves use upstream's `divisor`
  option (train on `len(train) // divisor` images).

## Our modifications to upstream

All in `gz_evo/downstream/finetune.py`, plus two added scripts:

- **New task `streams_shells_neither`** (`astronolan/streams_shells_neither`): renames the
  `image_rgb` column to `image` and encodes labels in a fixed order (stream=0, shell=1,
  neither=2) so class indices match our VLM runs.
- **`GZEVO_TRAIN_CSV` hook:** trains on an explicit list of `id_str` values (e.g. the exact
  100-image set a VLM was fine-tuned on) for a strictly matched CNN-vs-VLM comparison. It takes
  priority over `divisor`, so these runs are stored under `div1/`.
- **Synthesized ids** for datasets without an `id_str` column (e.g. `gz_rings`):
  `<split>_<row>` (e.g. `test_1234` = row 1234 of the test split), created before any
  subsampling so ids always refer to rows of the original split.
- **Safety check in `save_predictions`:** predictions are joined to ids by position, so the run
  fails if the number of predictions differs from the number of rows, instead of silently
  mislabeling them.
- **Run layout and output location:** runs are named `div<N>/seed<k>_<time>`; results go to
  the Narval / geir results directories.
- **Added scripts:** `gz_evo/downstream/predict_from_checkpoint.py` (re-run test predictions
  from a saved checkpoint, no retraining) and `gz_evo/downstream/predict_rings_simple.py`
  (the same for rings).

## Jobs (`jobs/`)
- `cnn_bal100.sbatch`, `cnn_rings_bal100.sbatch`, `cnn_sl_bal100.sbatch`,
  `cnn_streams_shells.sbatch` — 100-label matched runs.
- `array_*.sbatch` — label-efficiency sweeps (divisor subsampling) per task.
- `score_cnn_rings.py`, `verify_rings_align.py`, `prep_classification_cache.py` — scoring/prep.

## Results (`results/`)
Per-seed **test** predictions (`predictions_test.csv`) + metric CSVs are committed here so the
reported CNN numbers are verifiable. Train/validation predictions are omitted (large, and not
needed to verify test numbers). Score with `analysis/`, or for Euclid with
`tasks/euclid/encoder_baseline/compute_euclid_label_efficiency.py` (uses `tasks/euclid/euclid_VLMs/shared/path_scores.py`).
