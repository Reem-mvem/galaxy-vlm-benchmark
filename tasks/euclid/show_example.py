"""Print one galaxy's path with both models' answers side by side."""
import sys

import pandas as pd

from euclid_schema import QUESTIONS, cnn_answers, load_bundle_seed

gid = sys.argv[1] if len(sys.argv) > 1 else (
    "01101572d34e275b2f31e3aca2a2a5ec5d3c2f6f0a6e3c326d5cca251dfdfa02")

paths = pd.read_csv("euclid_truth_paths.csv", dtype=str).set_index("id_str").fillna("")
gem = pd.read_csv("gemini_euclid_predictions_structured.csv", dtype=str).set_index("id_str")
cnn = load_bundle_seed(model="gz_euclid_maxvit_base", seed=0).set_index("id_str")

row = paths.loc[gid]
print(f"galaxy {gid[:16]}...\n")
print(f"{'question':20s} {'truth (majority)':18s} {'CNN (seed 0)':20s} {'Gemini':20s}")
gc = cc = n = 0
for q in QUESTIONS:
    if not row[q]:
        continue
    c = cnn_answers(cnn.loc[[gid]], q).iloc[0]
    g = gem.loc[gid, q]
    n += 1
    cc += c == row[q]
    gc += g == row[q]
    cm = "OK" if c == row[q] else "X"
    gm = "OK" if g == row[q] else "X"
    print(f"{q:20s} {row[q]:18s} {c:<17s}{cm:3s} {g:<17s}{gm}")
print(f"\ngalaxy score:  CNN {cc}/{n} = {cc/n:.2f}   Gemini {gc}/{n} = {gc/n:.2f}")
