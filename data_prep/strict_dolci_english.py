"""Strict English filter for allenai/Dolci-Instruct-SFT.

A row is kept only if EVERY message in it (all roles, all turns) is English:
- Aya rows are dropped outright (multilingual by design);
- each message is cleaned (code blocks, inline code, LaTeX, URLs, digits and
  symbols removed) and split into 1,000-character windows; a window of >= 20
  characters fails if fastText lid.176 says another language with prob >= 0.3;
- a message fails if more than 5% of its letters are not Latin script.

Writes the kept ids and a JSON report with drop reasons and samples.
"""
import collections
import glob
import json
import random
import re
import unicodedata
from multiprocessing import Pool

import pyarrow.parquet as pq

ROOT = "/gpfs/scratch/ehpc532/eit234648"
DATA = f"{ROOT}/datasets/allenai/Dolci-Instruct-SFT/data/*.parquet"
LID = f"{ROOT}/models/fasttext/lid.176.bin"
OUT_IDS = f"{ROOT}/datasets/allenai/Dolci-Instruct-SFT-english-strict-ids.txt"
OUT_REPORT = f"{ROOT}/datasets/allenai/Dolci-Instruct-SFT-english-strict-report.json"
WINDOW = 1000

STRIP = [
    re.compile(r"```.*?```", re.S),
    re.compile(r"`[^`]*`"),
    re.compile(r"\$\$.*?\$\$", re.S),
    re.compile(r"\$[^$]*\$"),
    re.compile(r"\\\[.*?\\\]", re.S),
    re.compile(r"\\\(.*?\\\)", re.S),
    re.compile(r"https?://\S+"),
    re.compile(r"[0-9_=+\-*/<>{}\[\]()|\\#@%^&~]+"),
]

model = None


def clean(text):
    for p in STRIP:
        text = p.sub(" ", text)
    return " ".join(text.split())


def non_latin_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    bad = sum(1 for c in letters if not unicodedata.name(c, "").startswith("LATIN"))
    return bad / len(letters)


def check_row(row):
    """Return None if the row is all-English, else a drop reason."""
    if row["source_dataset"] == "Aya":
        return "aya"
    for turn, m in enumerate(row["messages"]):
        t = clean(m.get("content") or "")
        if non_latin_ratio(t) > 0.05:
            return f"{m['role']}:non-latin"
        for i in range(0, len(t), WINDOW):
            w = t[i:i + WINDOW]
            if len(w) < 20:
                continue
            (label,), (prob,) = model.predict(w, k=1)
            lang = label.replace("__label__", "")
            if lang != "en" and prob >= 0.3:
                return f"{m['role']}{'' if turn < 2 else '(later turn)'}:{lang}"
    return None


def process(path):
    global model
    if model is None:
        import fasttext
        fasttext.FastText.eprint = lambda *a, **k: None
        model = fasttext.load_model(LID)
    kept, reasons, by_source, samples = [], collections.Counter(), collections.defaultdict(collections.Counter), []
    for row in pq.read_table(path, columns=["id", "source_dataset", "messages"]).to_pylist():
        why = check_row(row)
        by_source[row["source_dataset"]]["kept" if why is None else "dropped"] += 1
        if why is None:
            kept.append(row["id"])
        else:
            reasons[why] += 1
            if why != "aya" and len(samples) < 200:
                msgs = row["messages"]
                samples.append({"id": row["id"], "source": row["source_dataset"], "why": why,
                                "n_msgs": len(msgs), "user": (msgs[0].get("content") or "")[:200]})
    return path, kept, reasons, {k: dict(v) for k, v in by_source.items()}, samples


if __name__ == "__main__":
    files = sorted(glob.glob(DATA))
    all_ids, reasons, by_source, samples = [], collections.Counter(), collections.defaultdict(collections.Counter), []
    with Pool(len(files)) as pool:
        for path, kept, r, bs, s in pool.imap_unordered(process, files):
            all_ids += kept
            reasons.update(r)
            for k, v in bs.items():
                by_source[k].update(v)
            samples += s
            print("done", path.rsplit("/", 1)[-1], len(kept), flush=True)

    random.seed(0)
    total = sum(sum(v.values()) for v in by_source.values())
    later = sum(n for k, n in reasons.items() if "later turn" in k)
    report = {
        "total_rows": total,
        "english_strict": len(all_ids),
        "dropped": total - len(all_ids),
        "dropped_in_later_turns": later,
        "drop_reasons": dict(reasons.most_common()),
        "per_source": {k: dict(v) for k, v in sorted(by_source.items(), key=lambda x: -sum(x[1].values()))},
        "samples": random.sample(samples, min(40, len(samples))),
    }
    with open(OUT_IDS, "w") as fh:
        fh.write("\n".join(sorted(all_ids)) + "\n")
    json.dump(report, open(OUT_REPORT, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("total_rows", "english_strict", "dropped", "dropped_in_later_turns")}))
    print("top reasons:", reasons.most_common(20))
    for s, c in report["per_source"].items():
        n = c.get("kept", 0) + c.get("dropped", 0)
        print(f"{s:40s} {n:>8}  kept={c.get('kept', 0):>8} ({c.get('kept', 0) / n:.1%})")
