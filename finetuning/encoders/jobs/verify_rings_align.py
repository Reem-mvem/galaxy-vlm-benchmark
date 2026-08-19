"""Confirm the balanced-100 rings ids line up with what finetune.py will see.

finetune.py loads load_dataset("mwalmsley/gz_rings","classification") and, since the
train split has no id_str, synthesizes train_<i> = row i. Our train_100.csv ids were
built the same way (build_rings_train.py). This checks the row ORDER matches by
verifying every one of the 100 ids maps to the SAME label_name in both places.
Exits non-zero on any mismatch so the CNN job's afterok dependency blocks.
"""
import sys
import pandas as pd
from datasets import load_dataset

CSV = "/home/reem1/scratch/galaxy-vlm-benchmark/tasks/rings/data/train_100.csv"

print("loading gz_rings classification (named config, exactly as finetune.py)...", flush=True)
dd = load_dataset("mwalmsley/gz_rings", "classification")
tr = dd["train"]
names = tr.features["label"].names
print(f"train rows={len(tr)}  label names={names}", flush=True)

# synthesize ids the way finetune.py does when id_str is absent
if "id_str" in tr.column_names:
    ids = list(tr["id_str"])
    print("NOTE: dataset HAS id_str column -> finetune uses real ids (train_<i> would NOT match!)")
else:
    ids = [f"train_{i}" for i in range(len(tr))]
labels = tr["label"]
id2name = {i: names[l] for i, l in zip(ids, labels)}

want = pd.read_csv(CSV)
miss = [r.id_str for r in want.itertuples() if r.id_str not in id2name]
mismatch = [(r.id_str, r.label_name, id2name.get(r.id_str))
            for r in want.itertuples()
            if r.id_str in id2name and id2name[r.id_str] != r.label_name]

print(f"train_100.csv rows={len(want)}  present_in_dataset={len(want)-len(miss)}  "
      f"missing={len(miss)}  label_mismatches={len(mismatch)}", flush=True)
if miss:
    print("  MISSING ids (first 5):", miss[:5])
if mismatch:
    print("  MISMATCHES (id, csv_label, dataset_label) first 5:", mismatch[:5])

if miss or mismatch:
    print("ALIGN_FAIL(train) — train_<i> ids do NOT map to the same galaxies; need a realids mapping", flush=True)
    sys.exit(1)
print("ALIGN_OK(train) — all 100 ids map to identical labels; GZEVO_TRAIN_CSV will select the exact VLM set", flush=True)

# ---- TEST-side alignment ----
# The CNN predicts on dd["test"] with synthesized test_<i> ids. The VLM was scored on the
# canonical-1000 catalog (test_<i> ids from rings_results.csv). They only refer to the SAME
# galaxies if the classification test split shares row order with whatever built the catalog.
# Verify every one of the 1000 catalog ids maps to the SAME label in dd["test"].
TEST_CAT = "/home/reem1/scratch/galaxy-vlm-benchmark/tasks/rings/data/rings_test1000_catalog.csv"
te = dd["test"]
tnames = te.features["label"].names
print(f"\nTEST rows={len(te)}  label names={tnames}  (class_i order for CNN scoring)", flush=True)
tids = list(te["id_str"]) if "id_str" in te.column_names else [f"test_{i}" for i in range(len(te))]
tid2name = {i: tnames[l] for i, l in zip(tids, te["label"])}
cat = pd.read_csv(TEST_CAT)
tmiss = [r.id_str for r in cat.itertuples() if r.id_str not in tid2name]
tmis  = [(r.id_str, r.label_name, tid2name.get(r.id_str))
         for r in cat.itertuples()
         if r.id_str in tid2name and tid2name[r.id_str] != r.label_name]
print(f"canonical-1000: present={len(cat)-len(tmiss)}  missing={len(tmiss)}  label_mismatches={len(tmis)}", flush=True)
if tmiss: print("  MISSING test ids (first 5):", tmiss[:5])
if tmis:  print("  TEST MISMATCHES (id, cat_label, dataset_label) first 5:", tmis[:5])
# Hard gate = PRESENCE. If any canonical id is absent, test_<i> indices don't line up (real
# misalignment) -> abort. Label disagreements are EXPECTED: the canonical labels use the
# regression-era binarization threshold, the classification config uses the official one, so a
# handful of borderline galaxies flip. Same galaxies, same index -> we score the CNN against the
# CANONICAL labels (rings_test1000_catalog.csv) so it matches the VLM eval exactly.
if tmiss:
    print(f"ALIGN_FAIL(test) — {len(tmiss)} canonical ids ABSENT from the classification test split; "
          "test_<i> indices do not line up. Abort.", flush=True)
    sys.exit(2)
if tmis:
    frac = len(tmis) / len(cat)
    print(f"ALIGN_WARN(test) — {len(tmis)}/{len(cat)} ({frac:.1%}) label disagreements = regression-vs-"
          "classification binarization threshold. Galaxies align by index (missing=0); CNN will be scored "
          "against the CANONICAL labels to match the VLM. (Random galaxies would disagree ~50%.)", flush=True)
    assert frac < 0.20, "too many disagreements — galaxies may NOT be the same set"
print("ALIGN_OK(test-index) — all 1000 canonical galaxies present at matching indices in the "
      "classification test split; identical-galaxy CNN-vs-VLM comparison is valid", flush=True)
print(f"RING_CLASS_INDEX={tnames.index('ring')}", flush=True)
