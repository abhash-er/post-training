"""Count strict-English rows of openeurollm/OpenEuroLLM-Excellent-SFT, and how
many of them carry a non-empty <think>...</think> reasoning trace.

"English" uses the same rule as strict_dolci_english.py: every message in the
row (all roles, all turns, reasoning included) must pass. A row "has think" if
any assistant message contains <think> with non-whitespace content before
</think>.
"""
import collections
import glob
import json
import os
import re
import sys
from multiprocessing import Pool

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strict_dolci_english as strict  # noqa: E402

ROOT = "/gpfs/scratch/ehpc532/eit234648"
DATA = f"{ROOT}/datasets/openeurollm/OpenEuroLLM-Excellent-SFT/data/*.parquet"
OUT_IDS = f"{ROOT}/datasets/openeurollm/OpenEuroLLM-Excellent-SFT-english-strict-ids.txt"
OUT_REPORT = f"{ROOT}/datasets/openeurollm/OpenEuroLLM-Excellent-SFT-think-report.json"

THINK = re.compile(r"<think>(.*?)</think>", re.S)


def think_state(messages):
    """'think' if any assistant turn has a non-empty trace, 'empty' if only empty
    <think></think> blocks, 'none' if no think block at all."""
    state = "none"
    for m in messages:
        if m["role"] != "assistant":
            continue
        for body in THINK.findall(m.get("content") or ""):
            if body.strip():
                return "think"
            state = "empty"
    return state


def process(path):
    if strict.model is None:
        import fasttext
        fasttext.FastText.eprint = lambda *a, **k: None
        strict.model = fasttext.load_model(strict.LID)
    counts = collections.defaultdict(collections.Counter)
    kept = []
    for row in pq.read_table(path, columns=["id", "source", "messages"]).to_pylist():
        why = strict.check_row({"source_dataset": row["source"], "messages": row["messages"]})
        lang = "en" if why is None else "other"
        counts[row["source"]][f"{lang}:{think_state(row['messages'])}"] += 1
        if why is None:
            kept.append(row["id"])
    return path, kept, {k: dict(v) for k, v in counts.items()}


if __name__ == "__main__":
    files = sorted(glob.glob(DATA))
    per_source = collections.defaultdict(collections.Counter)
    all_ids = []
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as pool:
        for path, kept, counts in pool.imap_unordered(process, files):
            all_ids += kept
            for k, v in counts.items():
                per_source[k].update(v)
            print("done", path.rsplit("/", 1)[-1], len(kept), flush=True)

    total = collections.Counter()
    for c in per_source.values():
        total.update(c)
    report = {
        "total_rows": sum(total.values()),
        "english_strict": sum(v for k, v in total.items() if k.startswith("en:")),
        "english_with_think": total["en:think"],
        "english_empty_think": total["en:empty"],
        "english_no_think_block": total["en:none"],
        "all_with_think": total["en:think"] + total["other:think"],
        "per_source": {k: dict(v) for k, v in sorted(per_source.items(), key=lambda x: -sum(x[1].values()))},
    }
    with open(OUT_IDS, "w") as fh:
        fh.write("\n".join(sorted(all_ids)) + "\n")
    json.dump(report, open(OUT_REPORT, "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "per_source"}))
    print(f"{'source':45s} {'rows':>9} {'en':>9} {'en+think':>9} {'other+think':>11}")
    for s, c in report["per_source"].items():
        n = sum(c.values())
        en = sum(v for k, v in c.items() if k.startswith("en:"))
        print(f"{s:45s} {n:>9} {en:>9} {c.get('en:think', 0):>9} {c.get('other:think', 0):>11}")
