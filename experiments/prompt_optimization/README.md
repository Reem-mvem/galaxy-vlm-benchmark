# Prompt optimization (experimental)

An exploratory experiment, not a headline result. The question: for the **zero-shot API**
models, how much of the accuracy gap is just a *prompt* problem — could a better-worded
prompt close it without any fine-tuning?

To test this without hand-tuning prompts ourselves, we ran an **automated prompt-search
loop**:

- **Generator** model (`gemini-3.1-flash-lite`) proposes a batch of diverse candidate
  prompts. In later rounds it sees the best-scoring past prompts and tries to improve on them.
- **Evaluator** model (`gemini-3.5-flash`) scores each candidate prompt on a held-out
  **validation** set (~300 images per task) by **balanced accuracy**.
- Best prompts seed the next round; identical prompts are cached so they are never re-scored.

## Why this stayed small

The whole loop runs on **paid API models** (Gemini as both generator and evaluator), so every
round costs money. We deliberately **capped** the number of rounds, prompts per round, and
validation-set size to keep the cost bounded. That makes the results **suggestive, not
conclusive** — there simply weren't enough rounds to be sure the search had converged.

## What we saw

On the **validation** set, optimized prompts improved balanced accuracy on several tasks:

| Task | best val balanced accuracy |
|---|---|
| `is_lsb` | ~0.91 |
| `strong_lens` | ~0.79 |
| `which_lsb` | ~0.68 |
| `rings` | ~0.66 |

So there are **signs** that prompt wording matters on some tasks. **But** the gains are
partly optimistic: because the validation set is small and the search optimizes directly
against it, the best prompt can **overfit the validation set** and not carry over to the
held-out **test** set. `gemini_val_vs_test.png` shows this val-vs-test gap — the difference
between the optimized (validation) score and the score on the untouched test set.

## Future direction

The cost cap is the main limitation. Re-running the loop with a **free, local open model**
(e.g. Qwen3.5-VL-4B) as the evaluator would remove the per-call cost, allowing **many more
rounds, more candidate prompts, and a larger validation set** — which should make the
direction much clearer and tell us whether the val gains are real or just search noise.

## Files

```
<task>/
  download_*_val.py        # build the validation catalog for the task
  *_val_catalog.csv        # the validation images used for scoring
  *_prompt_opt.py          # the generator/evaluator search loop
  *_prompt_opt_results.csv # every prompt tried, its round, and its val score
gemini_val_vs_test.png     # optimized (val) score vs. held-out test score
```

Each `*_prompt_opt_results.csv` has columns `round, prompt, score, n_failed, n_pos_pred,
cached` — the full search trace, so every reported score is reproducible from the prompt text.
