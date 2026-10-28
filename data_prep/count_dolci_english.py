"""Count English examples in allenai/Dolci-Instruct-SFT.

Runs fastText lid.176 on the first user turn of every row in the sources that
can contain non-English text, and reports per-source language counts.
"""
import collections
import glob
import json
import sys

import fasttext
import pyarrow.parquet as pq

DATA = "/gpfs/scratch/ehpc532/eit234648/datasets/allenai/Dolci-Instruct-SFT/data/*.parquet"
LID = "/gpfs/scratch/ehpc532/eit234648/models/fasttext/lid.176.bin"
OUT = "/gpfs/scratch/ehpc532/eit234648/datasets/allenai/Dolci-Instruct-SFT-langid.json"

model = fasttext.load_model(LID)
per_source = collections.defaultdict(collections.Counter)

for f in sorted(glob.glob(DATA)):
    t = pq.read_table(f, columns=["id", "source_dataset", "messages"]).to_pylist()
    for row in t:
        user = next((m["content"] for m in row["messages"] if m["role"] == "user"), "") or ""
        text = " ".join(user.split())[:2000]
        if not text:
            lang = "empty"
        else:
            (label,), (prob,) = model.predict(text, k=1)
            lang = label.replace("__label__", "")
            if prob < 0.5:
                lang += "?"
        per_source[row["source_dataset"]][lang] += 1
    print(f, file=sys.stderr, flush=True)

total = collections.Counter()
for counts in per_source.values():
    total.update(counts)
summary = {
    "total_rows": sum(total.values()),
    "en": total["en"],
    "en_low_conf": total["en?"],
    "per_source": {s: dict(c.most_common()) for s, c in per_source.items()},
}
json.dump(summary, open(OUT, "w"), indent=1)
print(json.dumps({k: summary[k] for k in ("total_rows", "en", "en_low_conf")}))
for s, c in sorted(per_source.items(), key=lambda x: -sum(x[1].values())):
    n = sum(c.values())
    print(f"{s:42s} {n:>8}  en={c['en']:>8} ({c['en'] / n:.1%})  top={c.most_common(3)}")
