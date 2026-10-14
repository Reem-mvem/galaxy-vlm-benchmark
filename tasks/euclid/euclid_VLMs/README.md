# Euclid — API VLMs (shared prompt)

Gemini 3.5 Flash, GPT-5.6-sol and Qwen3.5-397B on the GZ Euclid 13-question decision tree,
all run with the **same shared prompt and response schema**. Copies of the files in the repo's
`tasks/euclid/` (verified byte-identical).

## Layout

| Folder | Files |
|---|---|
| `shared/` | `euclid_eval_common.py` (prompt, 13-question schema, eval loop), `euclid_schema.py` (questions + truth helpers), `path_scores.py` (`build_truth_paths`, `score_model`), `sample_test_set.py` (selects the 1000 galaxies), `euclid_sample_ids.txt` (the 1000 galaxies), `euclid_truth_paths.csv` (volunteer truth paths) |
| `gemini/` | `euclid_eval_gemini_structured.py`, `gemini_euclid_predictions_structured.csv` |
| `gpt/` | `euclid_eval_openai_structured.py`, `openai_euclid_predictions_structured.csv` |
| `qwen/` | `euclid_eval_qwen_structured.py`, `qwen_euclid_predictions_structured.csv` |

## How a number is produced

1. Provider script + `shared/euclid_eval_common.py` → asks the model all 13 questions per galaxy
   with a structured schema (only allowed options possible) → one row per galaxy in the CSV,
   raw reply kept. An unreadable reply → blank row (counted wrong); a failed API call → galaxy
   skipped (not written).
2. `shared/path_scores.py` scores the CSV against `shared/euclid_truth_paths.csv`: per galaxy, the
   fraction of the volunteers' path questions answered correctly, averaged over galaxies.

## Settings

| | Gemini | GPT | Qwen-397B |
|---|---|---|---|
| model | `gemini-3.5-flash` | `gpt-5.6-sol` | `qwen/qwen3.5-397b-a17b` (OpenRouter) |
| reasoning | thinking `minimal` | effort `low` | off |
| other | — | image detail `high`, max 3000 tokens | temperature 0, max 1200 tokens |
| images | resized to 424×424 JPEG | same | same |

## Results (path accuracy, current truth-path rule)

| | Galaxies answered | Missing counted 0 (n=1000) | **Missing left out** |
|---|---|---|---|
| Gemini | 970 | 73.7 | **76.0** |
| GPT | 982 | 64.7 | **65.9** |
| Qwen-397B | 962 (+1 unreadable, counted wrong) | 67.1 | **69.7** |

Missing galaxies = API calls that failed and were never re-run.

Not included: the earlier Gemini run with the old prompt (`gemini_euclid_predictions_structured.oldprompt.csv`,
75.0, the previously reported number) and the old result plots (CNN vs old Gemini only).
No result plots exist yet for these shared-prompt runs.
