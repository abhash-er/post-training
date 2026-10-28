"""Re-check the low-confidence English rows of Dolci-Instruct-SFT and write the
final English id list.

A row from count_dolci_english.py is "en" if fastText said en with prob >= 0.5.
Rows with en at prob < 0.5 are re-checked here: code, LaTeX, URLs and digits
are stripped, then the user turn and the assistant turn are classified
separately. A row is kept only if neither cleaned side is detected as another
language (prob >= 0.3) and its letters are almost all Latin script. Aya rows
are always dropped: Aya is multilingual by design and fastText confuses its
Latin-script languages (mg, sn, zu, ...) with low-confidence English.
Confident-English rows whose assistant reply is another language are counted
(not dropped) in "en_confident_reply_other".
"""
import glob
import json
import random
import re
import unicodedata

import fasttext
import pyarrow.parquet as pq

ROOT = "/gpfs/scratch/ehpc532/eit234648"
DATA = f"{ROOT}/datasets/allenai/Dolci-Instruct-SFT/data/*.parquet"
LID = f"{ROOT}/models/fasttext/lid.176.bin"
OUT_IDS = f"{ROOT}/datasets/allenai/Dolci-Instruct-SFT-english-ids.txt"
OUT_REPORT = f"{ROOT}/datasets/allenai/Dolci-Instruct-SFT-lowconf-check.json"

model = fasttext.load_model(LID)

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


def clean(text):
    for p in STRIP:
        text = p.sub(" ", text)
    return " ".join(text.split())


def lid(text):
    (label,), (prob,) = model.predict(text[:2000], k=1)
    return label.replace("__label__", ""), float(prob)


def non_latin_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    bad = sum(1 for c in letters if not unicodedata.name(c, "").startswith("LATIN"))
    return bad / len(letters)


def first(msgs, role):
    return next((m["content"] for m in msgs if m["role"] == role), "") or ""


keep_ids, samples = [], {"kept": [], "dropped": [], "reply_other": []}
verdicts = {"en_confident": 0, "en_confident_reply_other": 0, "kept": 0, "dropped": 0}
reply_langs = {}
reasons = {}
for f in sorted(glob.glob(DATA)):
    for row in pq.read_table(f, columns=["id", "source_dataset", "messages"]).to_pylist():
        user = first(row["messages"], "user")
        text = " ".join(user.split())
        if not text:
            continue
        lang, prob = lid(text)
        if lang == "en" and prob >= 0.5:
            keep_ids.append(row["id"])
            verdicts["en_confident"] += 1
            ca = clean(first(row["messages"], "assistant"))
            if len(ca) >= 20:
                l2, p2 = lid(ca)
                if l2 != "en" and p2 >= 0.5:
                    verdicts["en_confident_reply_other"] += 1
                    reply_langs[l2] = reply_langs.get(l2, 0) + 1
                    if len(samples["reply_other"]) < 2000:
                        samples["reply_other"].append({"id": row["id"], "source": row["source_dataset"], "why": l2, "user": user[:300]})
            continue
        if lang != "en":
            continue  # confidently or weakly another language: not part of this check
        # low-confidence English: re-check
        cu, ca = clean(user), clean(first(row["messages"], "assistant"))
        why = "aya" if row["source_dataset"] == "Aya" else None
        for side, t in (("user", cu), ("assistant", ca)):
            if why:
                break
            if len(t) >= 20:
                l2, p2 = lid(t)
                if l2 != "en" and p2 >= 0.3:
                    why = f"{side}:{l2}"
                    break
            if non_latin_ratio(t) > 0.05:
                why = f"{side}:non-latin"
                break
        key = "dropped" if why else "kept"
        verdicts[key] += 1
        if why:
            reasons[why] = reasons.get(why, 0) + 1
        else:
            keep_ids.append(row["id"])
        if len(samples[key]) < 2000:
            samples[key].append({"id": row["id"], "source": row["source_dataset"], "why": why, "user": user[:300]})

random.seed(0)
report = {
    "verdicts": verdicts,
    "drop_reasons": dict(sorted(reasons.items(), key=lambda x: -x[1])),
    "english_total": len(keep_ids),
    "reply_other_langs": dict(sorted(reply_langs.items(), key=lambda x: -x[1])),
    "samples": {k: random.sample(v, min(15, len(v))) for k, v in samples.items()},
}
with open(OUT_IDS, "w") as fh:
    fh.write("\n".join(keep_ids) + "\n")
json.dump(report, open(OUT_REPORT, "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: report[k] for k in ("verdicts", "english_total")}))
print("reply-other langs:", list(report["reply_other_langs"].items())[:12])
print("top drop reasons:", list(report["drop_reasons"].items())[:15])
