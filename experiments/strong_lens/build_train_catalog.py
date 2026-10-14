import collections, pandas as pd
from datasets import load_dataset
ds = load_dataset("mwalmsley/euclid_strong_lens_expert_judges","classification",split="train")
names = ds.features["label"].names
rows = [{"id_str": ds[i]["id_str"], "label": int(ds[i]["label"]),
         "label_name": names[int(ds[i]["label"])]} for i in range(len(ds))]
df = pd.DataFrame(rows)
df.to_csv("/scratch/reem1/finetuning/strong_lens_train_catalog.csv", index=False)
c = collections.Counter(df["label"])
print("train size:", len(df), "counts:", dict(c), "positives:", c[1], "max balanced:", 2*c[1])
print("CATALOG_DONE")
