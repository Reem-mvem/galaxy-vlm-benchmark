import csv,glob,sys
from collections import defaultdict
BASE="/scratch/reem1/rings_runs/results/downstream/rings/balanced100_2seed_2026-08-04"
CAT="/home/reem1/scratch/galaxy-vlm-benchmark/tasks/rings/data/rings_test1000_catalog.csv"
truth={r["id_str"]:r["label_name"] for r in csv.DictReader(open(CAT))}   # canonical labels (== VLM)
res=[]
for s in (0,1):
    p=glob.glob(f"{BASE}/div1/seed{s}_*/predictions_test.csv")   # finetune.py names run_id div1/seed<S>_<ts>
    if not p: print(f"  seed{s}: predictions MISSING"); continue
    t=defaultdict(int); c=defaultdict(int); n=0
    for r in csv.DictReader(open(p[0])):
        g=r["id_str"]
        if g not in truth: continue      # keep only the canonical 1000
        n+=1
        pred="ring" if float(r["class_1"])>=float(r["class_0"]) else "not_ring"  # RING_CLASS_INDEX=1
        t[truth[g]]+=1
        if pred==truth[g]: c[truth[g]]+=1
    ba=sum((c[k]/t[k] if t[k] else 0) for k in ["not_ring","ring"])/2; res.append(ba)
    print(f"  ★ CNN rings@100 seed{s} = {ba:.3f}  (matched {n}/1000, identical galaxies, canonical labels)  "
          f"recall(not_ring={c['not_ring']/max(t['not_ring'],1):.2f}, ring={c['ring']/max(t['ring'],1):.2f})")
if res: print(f"  CNN rings mean={sum(res)/len(res):.3f}  min={min(res):.3f}  | vs 4B=0.686 7B=0.660 9B=0.719 Gemini=0.613")
