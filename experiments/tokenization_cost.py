#!/usr/bin/env python3
"""Tokenization cost of LLaMA 7B, Mistral 7B and Qwen2.5-7B on NTREX-128 (CPU only, a few minutes).

For 15 languages and 1,997 parallel sentences this reports tokens per sentence, the change for
Qwen2.5 relative to the better 32,000-token vocabulary with a bootstrap 95% interval, characters
per token, the byte-fallback share, and the KV-cache memory ratio LLaMA / Qwen2.5 per sentence.

Usage:  python tokenization_cost.py [output.json]
NTREX-128 is downloaded from its GitHub repository and is not redistributed here.
"""
import json
import re
import sys
import urllib.request

import numpy as np
from transformers import AutoTokenizer

BASE = "https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/main/NTREX-128/newstest2019-"
LANGS = [("English", "src.eng"), ("German", "ref.deu"), ("French", "ref.fra"), ("Spanish", "ref.spa"),
         ("Russian", "ref.rus"), ("Vietnamese", "ref.vie"), ("Arabic", "ref.arb"), ("Chinese", "ref.zho-CN"),
         ("Japanese", "ref.jpn"), ("Korean", "ref.kor"), ("Hindi", "ref.hin"), ("Bengali", "ref.ben"),
         ("Telugu", "ref.tel"), ("Tamil", "ref.tam"), ("Swahili", "ref.swa")]
REPOS = {"llama": "huggyllama/llama-7b", "mistral": "mistralai/Mistral-7B-v0.1", "qwen": "Qwen/Qwen2.5-7B"}
KV_BYTES = {"llama": 2 * 32 * 32 * 128 * 2, "mistral": 2 * 32 * 8 * 128 * 2, "qwen": 2 * 28 * 4 * 128 * 2}
BYTE_TOKEN = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def sentences(code):
    with urllib.request.urlopen(BASE + code + ".txt") as r:
        lines = r.read().decode("utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [l.rstrip("\r") for l in lines]


def main(out_path):
    toks = {k: AutoTokenizer.from_pretrained(v) for k, v in REPOS.items()}
    rng = np.random.default_rng(0)
    rows = []
    for lang, code in LANGS:
        sents = sentences(code)
        assert len(sents) == 1997, (lang, len(sents))
        counts, fallback = {}, {}
        for name, tok in toks.items():
            ids = tok(sents, add_special_tokens=False).input_ids
            counts[name] = np.array([len(x) for x in ids])
            if name != "qwen":
                pieces = [p for x in ids for p in tok.convert_ids_to_tokens(x)]
                fallback[name] = 100 * sum(bool(BYTE_TOKEN.match(p)) for p in pieces) / len(pieces)
        best = "llama" if counts["llama"].sum() <= counts["mistral"].sum() else "mistral"
        delta = 100 * (counts["qwen"].sum() / counts[best].sum() - 1)
        idx = rng.integers(0, len(sents), size=(2000, len(sents)))
        boot = 100 * (counts["qwen"][idx].sum(1) / counts[best][idx].sum(1) - 1)
        chars = sum(len(s) for s in sents)
        row = dict(language=lang, sentences=len(sents),
                   llama=counts["llama"].mean(), mistral=counts["mistral"].mean(), qwen=counts["qwen"].mean(),
                   better_32k=best, delta_pct=delta, delta_ci95=[float(np.percentile(boot, 2.5)),
                                                                 float(np.percentile(boot, 97.5))],
                   chars_per_token={k: chars / int(v.sum()) for k, v in counts.items()},
                   byte_fallback_pct=fallback,
                   kv_ratio_llama_over_qwen=(counts["llama"].sum() * KV_BYTES["llama"])
                   / (counts["qwen"].sum() * KV_BYTES["qwen"]),
                   mistral_vs_llama_pct=100 * (counts["mistral"].sum() / counts["llama"].sum() - 1))
        row = json.loads(json.dumps(row, default=float))
        rows.append(row)
        print(f"{lang:11s} L {row['llama']:6.1f}  M {row['mistral']:6.1f}  Q {row['qwen']:6.1f}  "
              f"delta {delta:6.1f}% [{row['delta_ci95'][0]:.1f}, {row['delta_ci95'][1]:.1f}]  "
              f"BF {fallback['llama']:.0f}/{fallback['mistral']:.0f}  KV L/Q {row['kv_ratio_llama_over_qwen']:.1f}  "
              f"M vs L {row['mistral_vs_llama_pct']:+.1f}%", flush=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    print("saved", out_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tokenization_results.json")
