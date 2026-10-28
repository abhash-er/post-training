"""Probe the strict-English, no-<think> pool of OpenEuroLLM-Excellent-SFT.

Per source, on a sample (first row group of every parquet file):
- rendered length with the qwen3 template + tokenizer-256k@qwen3-tokens,
  and the share of rows over 4,096 tokens;
- signs of reasoning written as plain text (no <think> tags): replies that
  open like a trace, or contain "Wait," mid-answer.
Run inside the training container (needs transformers).
"""
import collections
import glob
import re
import statistics
import sys

import pyarrow.parquet as pq
from transformers import AutoTokenizer

sys.path.insert(0, "src")
from post_training.chat_templates.registry import get_chat_template  # noqa: E402

ROOT = "/gpfs/scratch/ehpc532/eit234648"
DATA = f"{ROOT}/datasets/openeurollm/OpenEuroLLM-Excellent-SFT/data/*.parquet"
IDS = f"{ROOT}/datasets/openeurollm/OpenEuroLLM-Excellent-SFT-english-strict-ids.txt"
TOK = f"{ROOT}/models/openeurollm/tokenizer-256k-qwen3-tokens"
PER_SOURCE = 400
LIMIT = 4096

THINK = re.compile(r"<think>(.*?)</think>", re.S)
OPENER = re.compile(r"^\s*(okay|ok,|alright|let me|let's see|hmm|so,? i need|first,? i need)", re.I)

tok = AutoTokenizer.from_pretrained(TOK)
tok.chat_template = get_chat_template("qwen3")
english = set(open(IDS).read().split())

stats = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(glob.glob(DATA)):
    pf = pq.ParquetFile(f)
    for row in pf.read_row_group(0, columns=["id", "source", "messages", "tools"]).to_pylist():
        s = stats[row["source"]]
        if row["id"] not in english or len(s["tokens"]) >= PER_SOURCE:
            continue
        if any(b.strip() for m in row["messages"] if m["role"] == "assistant"
               for b in THINK.findall(m["content"] or "")):
            continue
        reply = next((m["content"] or "" for m in reversed(row["messages"]) if m["role"] == "assistant"), "")
        text = tok.apply_chat_template(row["messages"], tokenize=False)
        s["tokens"].append(len(tok(text, add_special_tokens=False)["input_ids"]))
        s["chars"].append(len(reply))
        s["opener"].append(bool(OPENER.match(reply)))
        s["wait"].append("Wait," in reply or "wait," in reply)
        s["sys"].append(row["messages"][0]["role"] == "system")

print(f"{'source':48s} {'n':>4} {'median':>7} {'p95':>7} {'>4k':>6} {'opener':>7} {'Wait,':>6} {'system':>6} {'reply chars':>11}")
for src, s in sorted(stats.items(), key=lambda x: -len(x[1]["tokens"])):
    t = s["tokens"]
    if not t:
        continue
    t_sorted = sorted(t)
    p95 = t_sorted[int(0.95 * (len(t) - 1))]
    share = lambda k: sum(s[k]) / len(t)  # noqa: E731
    print(f"{src:48s} {len(t):>4} {int(statistics.median(t)):>7} {p95:>7} "
          f"{sum(x > LIMIT for x in t) / len(t):>6.1%} {share('opener'):>7.1%} {share('wait'):>6.1%} {share('sys'):>6.1%} {int(statistics.mean(s['chars'])):>11}")
