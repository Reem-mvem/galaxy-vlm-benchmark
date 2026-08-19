# Specialist encoder baseline (gz-evo / Zoobot)

The specialist baseline fine-tunes pretrained image encoders (MaxViT / SoViT / ConvNeXt —
referred to as "CNN" in the results tables for brevity, though not all are CNNs). It is
**gz-evo** by Mike Walmsley et al., vendored under
`gz-evo/` with its original `LICENSE` preserved.

- **Upstream:** https://github.com/mwalmsley/zoobot (and the gz-evo benchmark repo)
- **What we use:** `gz_evo/downstream/finetune.py` to fine-tune a per-task pretrained encoder
  (MaxViT-Tiny / SoViT-400m / ConvNeXt-Base / MaxViT-Base, all `baseline-encoder-regression-*`)
  on the same labelled subsets as the VLMs.

## Our modifications to upstream
- Added task cases to `finetune.py` for `gz-rings-binary` and `streams_shells_neither`
  (label encoding, dataset loading, `image_rgb → image` rename).
- Added a `GZEVO_TRAIN_CSV` hook so the CNN can be forced onto the *exact* 100-label subset a
  VLM used (matched comparison), and a `divisor` path for label-efficiency curves.

## Jobs (`jobs/`)
- `cnn_bal100.sbatch`, `cnn_rings_bal100.sbatch`, `cnn_sl_bal100.sbatch`,
  `cnn_streams_shells.sbatch` — 100-label matched runs.
- `array_*.sbatch` — label-efficiency sweeps (divisor subsampling) per task.
- `score_cnn_rings.py`, `verify_rings_align.py`, `prep_classification_cache.py` — scoring/prep.

## Results (`results/`)
Per-seed **test** predictions (`predictions_test.csv`) + metric CSVs are committed here so the
reported CNN numbers are verifiable. Train/validation predictions are omitted (large, and not
needed to verify test numbers). Score with `analysis/` or `experiments/euclid_tree/path_scores.py`
(for euclid).
