# GZ Euclid — Gemini zero-shot vs fine-tuned CNN (13-question tree)

Gemini answers all 13 GZ Euclid decision-tree questions per galaxy in one
call; the fine-tuned CNN's vote-fraction predictions (bundle 2026-07-02) are
argmax-ed into answers. Both are scored identically against the volunteers'
majority answer, on the same 1000-galaxy sample of the 2378-galaxy test set.

Scoring rules (reverse-engineered to exactly reproduce
`zoobot_project/gz_euclid_cnn_pernode_accuracy.csv`):
- **relevant**: question routed to >= 50% of the galaxy's volunteers
- **correct**: predicted answer == argmax of true vote fractions
- **subsets**: `all-relevant` and `confident(maj>=0.8)`
- Gemini answers unconditionally (its own routing errors must not decide
  which questions it is graded on); irrelevant answers are ignored.

## Run order

```bash
python sample_test_set.py          # 1. download HF test split, pick 1000 ids
python euclid_eval_gemini.py --limit 5   # 2. smoke test (5 galaxies)
python euclid_eval_gemini.py       # 3. full run — resumable, rerun after crashes
python score_euclid.py             # 4. score both, print side-by-side table
```

Needs `GEMINI_API_KEY` (found via `../zoobot_project/.env`) and possibly
`HF_TOKEN` for the dataset download.

## Files

| file | what |
|---|---|
| `euclid_schema.py` | question/answer schema, relevance rule, truth helpers |
| `sample_test_set.py` | downloads test split, writes `euclid_sample_ids.txt` + `euclid_sample_truth.csv` |
| `euclid_eval_gemini.py` | Gemini eval -> `gemini_euclid_predictions.csv` (append/resume) |
| `score_euclid.py` | -> `gz_euclid_gemini_pernode_accuracy.csv`, `gz_euclid_cnn_pernode_accuracy_sample.csv` |
| `sent_images_sample/` | first 15 images actually sent (audit) |

Sample = all galaxies reaching the rare questions (edge-on-bulge, problem,
artifact) + random fill to 1000 (seed 42). The CNN is re-scored on exactly
these ids (all 6 seeds), so the comparison stays apples-to-apples.
