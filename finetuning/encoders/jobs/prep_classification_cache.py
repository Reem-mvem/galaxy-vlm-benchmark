import os
for k in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE","HF_DATASETS_OFFLINE"): os.environ.pop(k,None)
os.environ["HF_HOME"]="/scratch/reem1/rings_runs/hf_cache"
os.environ["HF_DATASETS_CACHE"]="/scratch/reem1/rings_runs/hf_cache/datasets"
from datasets import load_dataset
dd=load_dataset("mwalmsley/gz_rings","classification")
print("PREPARED:", {k:len(v) for k,v in dd.items()}, flush=True)
