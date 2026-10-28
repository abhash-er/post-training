"""Build the 18 SFT mixes of the Prelude English/EU ratio study.

Phase A (pools): normalise every source to one schema, render each row with the
qwen3 template + tokenizer-256k@qwen3-tokens, and keep rows of <= 4,096 tokens.
Phase B (mixes): sample each run from the pools and write one parquet per run
plus a JSON manifest (rows per source/language, repetition, tokens, steps).

Schema of every pool and mix row:
    id, source, lang, n_tokens, tools (JSON string or null),
    messages: list<{role, content, function_calls, functions}>
The trainer keeps only `messages` and `tools`; the rest is for auditing.

Sampling is nested: one fixed shuffled order per pool, and each run takes a
prefix of it, so the 50% English rows are a subset of the 75% ones, and so on.
Run inside the training container (needs transformers).
"""
import argparse
import glob
import gzip
import json
import math
import os
import random
import re
import sys
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from post_training.chat_templates.registry import get_chat_template  # noqa: E402

ROOT = "/gpfs/scratch/ehpc532/eit234648"
DS = f"{ROOT}/datasets"
TOK = f"{ROOT}/models/openeurollm/tokenizer-256k-qwen3-tokens"
OUT = f"{ROOT}/datasets/prelude_ratio_study"
MAX_TOKENS = 4096
ENGLISH_CAP = 1_968_566
SEED = 1234
TOKENS_PER_STEP = 32 * 32768
EPOCHS = 2

EU12 = ["cs", "de", "el", "es", "fi", "fr", "it", "nl", "pl", "ro", "sv", "uk"]
WILDCHAT_LANG = {"Czech": "cs", "German": "de", "Greek": "el", "Spanish": "es", "Finnish": "fi",
                 "French": "fr", "Italian": "it", "Dutch": "nl", "Polish": "pl", "Romanian": "ro",
                 "Swedish": "sv", "Ukrainian": "uk"}
THINK = re.compile(r"<think>(.*?)</think>", re.S)

MESSAGE = pa.struct([("role", pa.string()), ("content", pa.string()),
                     ("function_calls", pa.string()), ("functions", pa.string())])
SCHEMA = pa.schema([("id", pa.string()), ("source", pa.string()), ("lang", pa.string()),
                    ("n_tokens", pa.int32()), ("tools", pa.string()), ("messages", pa.list_(MESSAGE))])


def msg(role, content, function_calls=None, functions=None):
    return {"role": role, "content": content, "function_calls": function_calls, "functions": functions}


# ── sources ────────────────────────────────────────────────────────────


def dolci_english():
    keep = set(open(f"{DS}/allenai/Dolci-Instruct-SFT-english-strict-ids.txt").read().split())
    for f in sorted(glob.glob(f"{DS}/allenai/Dolci-Instruct-SFT/data/*.parquet")):
        for r in pq.read_table(f, columns=["id", "source_dataset", "messages"]).to_pylist():
            if r["id"] in keep:
                yield {"id": r["id"], "source": f"dolci:{r['source_dataset']}", "lang": "en", "tools": None,
                       "messages": [msg(**m) for m in r["messages"]]}


def excellent_instruct():
    """Strict-English rows of Excellent-SFT without a non-empty <think> trace."""
    keep = set(open(f"{DS}/openeurollm/OpenEuroLLM-Excellent-SFT-english-strict-ids.txt").read().split())
    for f in sorted(glob.glob(f"{DS}/openeurollm/OpenEuroLLM-Excellent-SFT/data/*.parquet")):
        for r in pq.read_table(f, columns=["id", "source", "messages", "tools"]).to_pylist():
            if r["id"] not in keep:
                continue
            if any(b.strip() for m in r["messages"] if m["role"] == "assistant"
                   for b in THINK.findall(m["content"] or "")):
                continue
            yield {"id": r["id"], "source": f"excellent:{r['source']}", "lang": "en", "tools": r["tools"],
                   "messages": [msg(m["role"], m["content"]) for m in r["messages"]]}


def synthetic(lang):
    for f in sorted(glob.glob(f"{ROOT}/hf_release/synthgen-35lang-if/data/{lang}/*.parquet")):
        for r in pq.read_table(f, columns=["id", "messages"]).to_pylist():
            yield {"id": r["id"], "source": "synthetic", "lang": lang, "tools": None,
                   "messages": [msg(m["role"], m["content"]) for m in r["messages"]]}


def translated(lang):
    for f in sorted(glob.glob(f"{DS}/openeurollm/Dolci-Instruct-SFT-translated/{lang}/*.parquet")):
        for r in pq.read_table(f, columns=["id", "messages"]).to_pylist():
            yield {"id": f"{lang}:{r['id']}", "source": "translated", "lang": lang, "tools": None,
                   "messages": [msg(m["role"], m["content"]) for m in r["messages"]]}


def _oasst_path(node):
    """Linearise a reply tree: rank-0 (best) assistant reply at every step,
    first surviving user follow-up; trimmed to end on an assistant turn."""
    out = []
    while node is not None:
        if node.get("deleted") or node.get("review_result") is False:
            break
        out.append(msg("user" if node["role"] == "prompter" else "assistant", node["text"]))
        replies = [c for c in node.get("replies", []) if not c.get("deleted") and c.get("review_result") is not False]
        if not replies:
            break
        if replies[0]["role"] == "assistant":
            ranked = [c for c in replies if c.get("rank") is not None]
            node = min(ranked, key=lambda c: c["rank"]) if ranked else replies[0]
        else:
            node = replies[0]
    while out and out[-1]["role"] != "assistant":
        out.pop()
    return out


def real():
    for f in sorted(glob.glob(f"{DS}/allenai/WildChat-4.8M/data/*.parquet")):
        t = pq.read_table(f, columns=["conversation_hash", "conversation", "language", "toxic", "redacted"])
        for r in t.to_pylist():
            lang = WILDCHAT_LANG.get(r["language"])
            if lang is None or r["toxic"] or r["redacted"]:
                continue
            msgs = [msg(m["role"], m["content"]) for m in r["conversation"]]
            if msgs and msgs[-1]["role"] == "assistant":
                yield {"id": f"wildchat:{r['conversation_hash']}", "source": "real:wildchat", "lang": lang,
                       "tools": None, "messages": msgs}
    with gzip.open(f"{DS}/OpenAssistant/oasst2/2023-11-05_oasst2_ready.trees.jsonl.gz", "rt") as fh:
        for line in fh:
            tree = json.loads(line)
            lang = tree["prompt"].get("lang")
            if lang not in EU12:
                continue
            msgs = _oasst_path(tree["prompt"])
            if msgs:
                yield {"id": f"oasst2:{tree['message_tree_id']}", "source": "real:oasst2", "lang": lang,
                       "tools": None, "messages": msgs}


# ── phase A: pools with the 4k filter ──────────────────────────────────

_tok = None


def _n_tokens(rows):
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(TOK)
        _tok.chat_template = get_chat_template("qwen3")
    out = []
    for r in rows:
        tools = json.loads(r["tools"]) if r["tools"] else None
        text = _tok.apply_chat_template(r["messages"], tools=tools, tokenize=False)
        out.append(len(_tok(text, add_special_tokens=False)["input_ids"]))
    return out


def _chunks(it, n=5000):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def build_pool(name, rows, pool):
    path = f"{OUT}/pools/{name}.parquet"
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = pq.ParquetWriter(path + ".tmp", SCHEMA)
    seen = kept = 0
    batches = list(_chunks(rows))
    for batch, lengths in zip(batches, pool.imap(_n_tokens, batches)):
        seen += len(batch)
        good = []
        for r, n in zip(batch, lengths):
            if n <= MAX_TOKENS:
                r["n_tokens"] = n
                good.append(r)
        kept += len(good)
        if good:
            writer.write_table(pa.Table.from_pylist(good, schema=SCHEMA))
    writer.close()
    os.replace(path + ".tmp", path)
    print(f"pool {name}: {kept:,} of {seen:,} rows <= {MAX_TOKENS} tokens", flush=True)


# ── phase B: mixes ─────────────────────────────────────────────────────


LARGE_MESSAGE = pa.struct([("role", pa.large_string()), ("content", pa.large_string()),
                           ("function_calls", pa.large_string()), ("functions", pa.large_string())])
LARGE_SCHEMA = pa.schema([("id", pa.large_string()), ("source", pa.large_string()), ("lang", pa.large_string()),
                          ("n_tokens", pa.int32()), ("tools", pa.large_string()),
                          ("messages", pa.large_list(LARGE_MESSAGE))])


def load(name):
    # 64-bit offsets: a ~2M-row take of the English pools exceeds the 2 GB
    # limit of 32-bit string/list offsets ("offset overflow").
    return pq.read_table(f"{OUT}/pools/{name}.parquet").cast(LARGE_SCHEMA)


def order(n, key):
    idx = list(range(n))
    random.Random(f"{SEED}:{key}").shuffle(idx)
    return idx


def take(n_avail, n_want, key):
    """Indices for n_want rows: a prefix of a fixed shuffle, repeating full
    shuffled copies when n_want > n_avail."""
    o = order(n_avail, key)
    return o * (n_want // n_avail) + o[: n_want % n_avail]


def split_even(total, langs):
    base, extra = divmod(total, len(langs))
    return {lang: base + (i < extra) for i, lang in enumerate(langs)}


def build_mix(spec):
    name, english, share, eu, langs, cap = spec
    parts, manifest = [], {"run": name, "english_source": english, "english_share": share,
                           "eu_source": eu, "languages": langs, "rows": cap, "parts": []}
    n_en = round(cap * share)
    if n_en:
        t = load(english)
        parts.append(t.take(take(len(t), n_en, english)))
        manifest["parts"].append({"pool": english, "rows": n_en, "available": len(t),
                                  "repetition": round(n_en / len(t), 3)})
    n_eu = cap - n_en
    if n_eu:
        prefix = {"synthetic": "synthetic", "translated": "translated", "real": "real"}[eu]
        tables = {lang: load(f"{prefix}/{lang}") for lang in langs}
        # Uniform for every arm: real-data languages above their share are
        # subsampled (fr, es at 75% English), the rest are repeated to fill it.
        alloc = split_even(n_eu, langs)
        for lang in langs:
            t = tables[lang]
            parts.append(t.take(take(len(t), alloc[lang], f"{prefix}/{lang}")))
            manifest["parts"].append({"pool": f"{prefix}/{lang}", "rows": alloc[lang], "available": len(t),
                                      "repetition": round(alloc[lang] / len(t), 3)})
    mix = pa.concat_tables(parts)
    mix = mix.take(order(len(mix), f"mix:{name}"))
    tokens = int(pc.sum(mix["n_tokens"]).as_py())
    manifest.update({"tokens": tokens,
                     "steps_per_epoch_at_full_packing": math.ceil(tokens / TOKENS_PER_STEP),
                     "steps_total_estimate": EPOCHS * math.ceil(tokens / TOKENS_PER_STEP)})
    pq.write_table(mix, f"{OUT}/mixes/{name}.parquet", row_group_size=20_000)
    json.dump(manifest, open(f"{OUT}/mixes/{name}.json", "w"), indent=1)
    return f"{name}: {len(mix):,} rows, {tokens / 1e9:.2f}B tokens, ~{manifest['steps_total_estimate']:,} steps"


def run_name(n, english, share, eu, langs):
    """e.g. r02_prelude20m-sft_en-dolci-75pct_eu-synthetic-12lang"""
    eng = {"dolci_en": "dolci", "excellent_instruct": "excellent-instruct"}[english]
    eu_part = "eu-none" if eu is None else f"eu-{eu}-{len(langs)}lang"
    return f"r{n:02d}_prelude20m-sft_en-{eng}-{int(share * 100)}pct_{eu_part}"


def run_specs(cap, eu35):
    specs = []
    for e_i, english in enumerate(("dolci_en", "excellent_instruct")):
        base = 1 + 9 * e_i
        specs.append((run_name(base, english, 1.0, None, []), english, 1.0, None, [], cap))
        for s_i, share in enumerate((0.75, 0.50)):
            arms = [("synthetic", EU12), ("synthetic", eu35), ("translated", EU12), ("real", EU12)]
            for a_i, (eu, langs) in enumerate(arms):
                n = base + 1 + 4 * s_i + a_i
                specs.append((run_name(n, english, share, eu, langs), english, share, eu, langs, cap))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    args = ap.parse_args()
    eu35 = sorted(os.path.basename(d) for d in glob.glob(f"{ROOT}/hf_release/synthgen-35lang-if/data/*"))
    assert len(eu35) == 35 and set(EU12) <= set(eu35), eu35

    with Pool(args.workers) as pool:
        build_pool("dolci_en", dolci_english(), pool)
        build_pool("excellent_instruct", excellent_instruct(), pool)
        for lang in eu35:
            build_pool(f"synthetic/{lang}", synthetic(lang), pool)
        for lang in EU12:
            build_pool(f"translated/{lang}", translated(lang), pool)
        if not all(os.path.exists(f"{OUT}/pools/real/{lang}.parquet") for lang in EU12):
            by_lang = {lang: [] for lang in EU12}
            for r in real():
                by_lang[r["lang"]].append(r)
            for lang in EU12:
                build_pool(f"real/{lang}", by_lang[lang], pool)

    cap = min(ENGLISH_CAP, pq.read_metadata(f"{OUT}/pools/dolci_en.parquet").num_rows)
    excellent = pq.read_metadata(f"{OUT}/pools/excellent_instruct.parquet").num_rows
    print(f"run size (cap) = {cap:,}; excellent_instruct pool = {excellent:,}", flush=True)
    if excellent < cap:
        raise SystemExit("excellent_instruct pool is smaller than the cap; stopping before building mixes")

    os.makedirs(f"{OUT}/mixes", exist_ok=True)
    with Pool(min(args.workers, 6)) as pool:
        for line in pool.imap_unordered(build_mix, run_specs(cap, eu35)):
            print(line, flush=True)


if __name__ == "__main__":
    main()
