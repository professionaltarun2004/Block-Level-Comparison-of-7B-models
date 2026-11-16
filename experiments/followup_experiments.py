#!/usr/bin/env python3
"""Follow-up GPU experiments for the 7B block-level paper (free Colab T4).

Experiments (each result is one JSON line, the run can be resumed):
  sweep    Passkey retrieval for Mistral with the key at a set token distance from the
           end of the prompt, 3,600 to 4,600 tokens. Locates the retrieval limit.
  grid     Passkey retrieval at five depths per length, three keys per depth, with a
           different filler phase per trial.
           Variants: "default" (model as released, 4-bit), "nowindow" (Mistral with the
           sliding-window mask and rolling cache turned off, same weights, 4-bit),
           "8bit" (8-bit weights).
  latency  Decode latency, five blocks of 32 steps per context length, plus the time of
           the per-step cache concatenation measured on its own.

Run in Colab:   !python /content/followup_experiments.py
Results go to:  MyDrive/arxiv_7b_experiments/results_followup.jsonl
"""
import importlib
import os
import subprocess
import sys

sys.stdout.reconfigure(line_buffering=True)

TEST = bool(os.environ.get("FOLLOWUP_TEST"))  # CPU self-test with tiny random models
PINS = {"transformers": "5.22.2", "accelerate": "1.15.0"}


def ensure_packages():
    """Install the library versions used for the paper, then restart the process once."""
    if TEST or os.environ.get("FOLLOWUP_SKIP_ENSURE"):
        return
    need = []
    for pkg, ver in PINS.items():
        try:
            if importlib.import_module(pkg).__version__ != ver:
                need.append(f"{pkg}=={ver}")
        except ImportError:
            need.append(f"{pkg}=={ver}")
    try:
        importlib.import_module("bitsandbytes")
    except ImportError:
        need.append("bitsandbytes>=0.48")
    if need:
        if os.environ.get("FOLLOWUP_REEXEC"):
            sys.exit(f"Package install did not take effect: {need}")
        print("Installing pinned packages:", need, flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *need])
        os.environ["FOLLOWUP_REEXEC"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)


ensure_packages()

import gc
import json
import platform
import random
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache

CHUNK = 1024
DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]
KEYS_PER_SETTING = 3
BLOCKS, STEPS = 5, 32  # latency: 5 blocks of 32 decode steps

INTRO = ("There is an important info hidden inside a lot of irrelevant text. "
         "Find it and memorize it. I will quiz you about the important information there.\n\n")
SENTENCES = ["The grass is green. ", "The sky is blue. ", "The sun is yellow. ", "Here we go. ",
             "There and back again. "]
NEEDLE = "The pass key is {key}. Remember it. {key} is the pass key. "
QUESTION = "\n\nWhat is the pass key? The pass key is"

OUT = Path(os.environ.get("FOLLOWUP_OUT", "/content/drive/MyDrive/arxiv_7b_experiments/results_followup.jsonl"))
RUN_ID = time.strftime("%Y%m%dT%H%M%S")


# ------------------------------------------------------------------ helpers
def _dev(model):
    return next(model.parameters()).device


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def read_rows():
    return [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()] if OUT.exists() else []


def done_keys():
    return {r["key"] for r in read_rows()}


def append(row):
    row = dict(row, run_id=RUN_ID)
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(row, flush=True)


@torch.no_grad()
def prefill(model, ids, chunk=CHUNK):
    cache, logits = DynamicCache(config=model.config), None
    for s in range(0, ids.shape[1], chunk):
        out = model(input_ids=ids[:, s:s + chunk], past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache, logits = out.past_key_values, out.logits[:, -1].float()
    return cache, logits


@torch.no_grad()
def decode(model, cache, logits, steps, stop_ids=()):
    """Greedy decoding. Returns tokens, per-step seconds, and the cache and logits to continue from."""
    dev, toks, times = _dev(model), [], []
    nxt = logits.argmax(-1, keepdim=True)
    for i in range(steps):
        toks.append(int(nxt.item()))
        if toks[-1] in stop_ids or i == steps - 1:
            break
        _sync(dev)
        t0 = time.perf_counter()
        out = model(input_ids=nxt, past_key_values=cache, use_cache=True, logits_to_keep=1)
        _sync(dev)
        times.append(time.perf_counter() - t0)
        cache, logits = out.past_key_values, out.logits[:, -1].float()
        nxt = logits.argmax(-1, keepdim=True)
    return toks, times, cache, logits


def cache_positions(cache):
    return max(layer.keys.shape[-2] for layer in cache.layers if isinstance(layer.keys, torch.Tensor))


# ------------------------------------------------------------------ prompts
def n_tok(tok, text):
    return len(tok(text).input_ids)


def filler(n, phase):
    return "".join(SENTENCES[(phase + i) % len(SENTENCES)] for i in range(n))


def build_prompt(tok, total, key, phase, depth=None, distance=None):
    """Passkey prompt of about `total` tokens (never more).

    Give `depth` (needle after that fraction of the filler) or `distance` (target number of
    tokens from the first digit of the key to the end of the prompt).
    Returns ids and the measured distances from the first and from the second mention of
    the key to the end of the prompt. A token at distance d is the d-th token from the end,
    so the last prompt token can attend to it under a window W when d <= W. The count comes
    from tokenizing the text before the key, so it can be off by one token.
    """
    per = (n_tok(tok, filler(200, 0)) - n_tok(tok, filler(100, 0))) / 100  # tokens per filler sentence
    needle = NEEDLE.format(key=key)
    fixed = n_tok(tok, INTRO + needle + QUESTION)
    n = max(2, int((total - fixed) / per))

    def make(n_before, n_after):
        return INTRO + filler(n_before, phase) + needle + filler(n_after, phase + n_before) + QUESTION

    def measure(text):
        ids = tok(text, return_tensors="pt").input_ids
        first = text.index(str(key))
        second = text.index(str(key), first + 1)
        size = ids.shape[1]
        return ids, size - n_tok(tok, text[:first]), size - n_tok(tok, text[:second])

    if distance is None:
        n_before = int(round(depth * n))
        n_after = n - n_before
    else:
        n_after = max(0, int(round((distance - n_tok(tok, needle + QUESTION)) / per)))
        for _ in range(8):  # move the needle one sentence at a time toward the target distance
            _, d1, _ = measure(make(1, n_after))
            if abs(d1 - distance) <= per / 2 + 1:
                break
            n_after = max(0, n_after + (1 if d1 < distance else -1) * max(1, int(round(abs(d1 - distance) / per))))
        n_before = max(1, n - n_after)
    while True:
        ids, d1, d2 = measure(make(n_before, n_after))
        if ids.shape[1] <= total or n_before <= 1:
            return ids, d1, d2
        n_before -= max(1, int((ids.shape[1] - total) / per))


def score(text, key):
    digits = "".join(ch if ch.isdigit() else " " for ch in text).split()
    return bool(digits) and digits[0] == str(key)


def passkey_trial(tok, model, row, total, key, phase, depth=None, distance=None):
    dev = _dev(model)
    stop = {i for i in (tok.eos_token_id,) if i is not None}
    ids, d1, d2 = build_prompt(tok, total, key, phase, depth=depth, distance=distance)
    row = dict(row, ctx=total, ctx_actual=int(ids.shape[1]), passkey=key, phase=phase,
               dist_first=int(d1), dist_second=int(d2))
    try:
        t0 = time.perf_counter()
        cache, logits = prefill(model, ids.to(dev))
        held = cache_positions(cache)
        toks, _, cache, logits = decode(model, cache, logits, 12, stop)
        text = tok.decode(toks)
        row.update(status="ok", output=text, correct=score(text, key), cache_positions=held,
                   seconds=round(time.perf_counter() - t0, 2))
        del cache, logits
    except torch.OutOfMemoryError:
        row.update(status="oom", correct=None)
    _free()
    append(row)


def key_and_phase(tag):
    r = random.Random(tag)
    return r.randint(10000, 99999), r.randrange(len(SENTENCES))


# ------------------------------------------------------------------ experiments
def sweep_distances(coarse_step):
    """Target distances: a coarse grid from 3,600 to 4,600 and a fine grid around 4,096."""
    return sorted(set(range(3600, 4601, coarse_step)) | set(range(4040, 4151, 10)))


def run_sweep(name, variant, tok, model, total, coarse_step, keys):
    done = done_keys()
    for dist in sweep_distances(coarse_step):
        for trial in range(keys):
            k = f"sweep|{name}|{variant}|{total}|{dist}|{trial}"
            if k in done:
                continue
            pk, phase = key_and_phase(f"sweep-{total}-{dist}-{trial}")
            passkey_trial(tok, model, dict(key=k, exp="sweep", model=name, variant=variant, target_dist=dist,
                                           trial=trial), total, pk, phase, distance=dist)


def run_grid(name, variant, tok, model, lengths, keys=KEYS_PER_SETTING):
    done = done_keys()
    for total in lengths:
        for depth in DEPTHS:
            for trial in range(keys):
                k = f"grid|{name}|{variant}|{total}|{depth}|{trial}"
                if k in done:
                    continue
                pk, phase = key_and_phase(f"grid-{total}-{depth}-{trial}")  # same keys for every model
                passkey_trial(tok, model, dict(key=k, exp="grid", model=name, variant=variant, depth=depth,
                                               trial=trial), total, pk, phase, depth=depth)


@torch.no_grad()
def concat_cost_ms(model, cache, reps=10):
    """Time of one decode step's cache update on its own: every layer's K and V get one position appended."""
    dev, times = _dev(model), []
    window = getattr(model.config, "sliding_window", None)
    for _ in range(reps):
        _sync(dev)
        t0 = time.perf_counter()
        for layer in cache.layers:
            for t in (layer.keys, layer.values):
                new = torch.cat([t, t[:, :, -1:, :]], dim=-2)
                if window is not None:
                    new = new[:, :, -window + 1:, :]
                del new
        _sync(dev)
        times.append(time.perf_counter() - t0)
    return 1000 * statistics.median(times)


def run_latency(name, variant, tok, model, contexts, seed=0):
    dev, done = _dev(model), done_keys()
    g = torch.Generator().manual_seed(seed)
    lo, hi = min(300, tok.vocab_size // 2), tok.vocab_size
    todo = [T for T in contexts if f"latency|{name}|{variant}|{T}" not in done]
    if not todo:
        return
    w_cache, w_logits = prefill(model, torch.randint(lo, hi, (1, 256), generator=g).to(dev))  # warm-up
    decode(model, w_cache, w_logits, 4)
    del w_cache, w_logits
    for T in todo:
        _free()
        row = dict(key=f"latency|{name}|{variant}|{T}", exp="latency", model=name, variant=variant, ctx=T)
        try:
            cache, logits = prefill(model, torch.randint(lo, hi, (1, T), generator=g).to(dev))
            meds = []
            for _ in range(BLOCKS):
                _, times, cache, logits = decode(model, cache, logits, STEPS)
                meds.append(1000 * statistics.median(times))
            row.update(status="ok", decode_ms_blocks=[round(m, 2) for m in meds],
                       decode_ms_median=round(statistics.median(meds), 2), decode_ms_min=round(min(meds), 2),
                       decode_ms_max=round(max(meds), 2), concat_ms=round(concat_cost_ms(model, cache), 2),
                       cache_positions=cache_positions(cache))
            del cache, logits
        except torch.OutOfMemoryError:
            row.update(status="oom")
        _free()
        append(row)
        if row["status"] == "oom":
            break


# ------------------------------------------------------------------ models
REPOS = {"mistral-7b": "mistralai/Mistral-7B-v0.1", "qwen2.5-7b": "Qwen/Qwen2.5-7B", "llama-7b": "huggyllama/llama-7b"}


def repo_info(repo):
    """Commit hash of the repository and SHA-256 of its weight files, as listed by the Hub."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, files_metadata=True)
        files = {s.rfilename: s.lfs.sha256 for s in info.siblings
                 if s.lfs is not None and s.rfilename.endswith((".safetensors", ".bin"))}
        return info.sha, files
    except Exception as e:  # the run continues without a pinned revision
        print(f"   could not read repository info for {repo}: {e}", flush=True)
        return None, {}


def load(name, variant):
    from transformers import BitsAndBytesConfig
    repo = REPOS[name]
    sha, files = repo_info(repo)
    tok = AutoTokenizer.from_pretrained(repo, revision=sha)
    cfg = AutoConfig.from_pretrained(repo, revision=sha)
    if variant == "nowindow":
        cfg.sliding_window = None
    if variant == "8bit":
        q = BitsAndBytesConfig(load_in_8bit=True)
    else:
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(repo, revision=sha, config=cfg, dtype=torch.float16,
                                                 device_map={"": 0}, attn_implementation="sdpa",
                                                 quantization_config=q).eval()
    want = "Linear8bitLt" if variant == "8bit" else "Linear4bit"
    n = sum(type(x).__name__ == want for x in model.modules())
    if n == 0:
        raise SystemExit(f"{name}/{variant}: no {want} layers found. Stopping.")
    k = f"load|{name}|{variant}|{RUN_ID}"
    append(dict(key=k, exp="load", model=name, variant=variant, repo=repo, revision=sha, weight_sha256=files,
                quantized_layers=n, weights_gib=round(torch.cuda.memory_allocated() / 2**30, 2),
                sliding_window=getattr(model.config, "sliding_window", None), load_s=round(time.time() - t0)))
    return tok, model


# One entry per model load. Order: the most important results come first.
PLAN = [
    ("mistral-7b", "default", [("sweep", dict(total=8192, coarse_step=50, keys=3)),
                               ("grid", dict(lengths=[4096, 8192, 16384, 32768])),
                               ("sweep", dict(total=16384, coarse_step=100, keys=2)),
                               ("latency", dict(contexts=[1024, 2048, 4096, 8192, 16384, 32768, 65536]))]),
    ("mistral-7b", "nowindow", [("grid", dict(lengths=[4096, 8192, 16384, 32768])),
                                ("sweep", dict(total=8192, coarse_step=100, keys=2))]),
    ("mistral-7b", "8bit", [("grid", dict(lengths=[8192, 16384], keys=2))]),
    ("qwen2.5-7b", "default", [("grid", dict(lengths=[4096, 8192, 16384, 32768])),
                               ("latency", dict(contexts=[1024, 2048, 4096, 8192, 16384, 32768, 65536]))]),
    ("qwen2.5-7b", "8bit", [("grid", dict(lengths=[16384, 32768], keys=2))]),
    ("llama-7b", "default", [("grid", dict(lengths=[1024, 2048, 4096])),
                             ("latency", dict(contexts=[1024, 2048, 4096, 8192, 16384]))]),
]


def expected_keys(name, variant, exp, kw):
    if exp == "sweep":
        return [f"sweep|{name}|{variant}|{kw['total']}|{d}|{t}" for d in sweep_distances(kw["coarse_step"])
                for t in range(kw["keys"])]
    if exp == "grid":
        return [f"grid|{name}|{variant}|{T}|{d}|{t}" for T in kw["lengths"] for d in DEPTHS
                for t in range(kw.get("keys", KEYS_PER_SETTING))]
    return [f"latency|{name}|{variant}|{T}" for T in kw["contexts"]]


def pending(name, variant, steps):
    rows = read_rows()
    done = {r["key"] for r in rows}
    oom = {(r["model"], r["variant"]) for r in rows if r.get("exp") == "latency" and r.get("status") == "oom"}
    out = []
    for exp, kw in steps:
        if exp == "latency" and (name, variant) in oom:
            continue  # the latency run already stopped at an out-of-memory point
        if any(k not in done for k in expected_keys(name, variant, exp, kw)):
            out.append((exp, kw))
    return out


def run_steps(name, variant, tok, model, steps):
    for exp, kw in steps:
        print(f"-- {name}/{variant}: {exp} {kw}", flush=True)
        if exp == "sweep":
            run_sweep(name, variant, tok, model, **kw)
        elif exp == "grid":
            run_grid(name, variant, tok, model, **kw)
        else:
            run_latency(name, variant, tok, model, **kw)


# ------------------------------------------------------------------ preflight
def tiny_mistral(window, dtype=torch.float32):
    from transformers import MistralConfig, MistralForCausalLM
    torch.manual_seed(0)
    cfg = MistralConfig(vocab_size=1000, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=1, sliding_window=window)
    return MistralForCausalLM(cfg).to(dtype).eval()


@torch.no_grad()
def preflight(dev):
    """Check on a tiny random model that the two Mistral variants behave as the paper states."""
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    ids = torch.randint(0, 1000, (1, 300), generator=torch.Generator().manual_seed(1)).to(dev)
    res = {}
    for label, window in (("window", 64), ("nowindow", None)):
        m = tiny_mistral(window, dtype).to(dev)
        cache, chunked = prefill(m, ids, chunk=128)
        held = cache_positions(cache)
        full = m(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        res[label] = dict(cache_positions=held, chunked_vs_full=float((chunked - full).abs().max()),
                          logits=chunked.cpu())
        del m, cache
    assert res["window"]["cache_positions"] == 63, "sliding-window cache does not hold W-1 positions"
    assert res["nowindow"]["cache_positions"] == 300, "full cache does not hold every position"
    differ = float((res["window"]["logits"] - res["nowindow"]["logits"]).abs().max())
    assert differ > 1e-3, "turning the window off did not change the output"
    row = dict(key=f"preflight|{RUN_ID}", exp="preflight",
               window_chunked_vs_full=res["window"]["chunked_vs_full"],
               nowindow_chunked_vs_full=res["nowindow"]["chunked_vs_full"], window_vs_nowindow=differ)
    append(row)
    tol = 0.05 if dtype == torch.float16 else 1e-3
    if max(row["window_chunked_vs_full"], row["nowindow_chunked_vs_full"]) > tol:
        print("WARNING: chunked prefill and one full pass give different logits. Send this line back.", flush=True)
    else:
        print("Preflight OK: the window limits the cache and the mask, chunked prefill matches one full pass, "
              "and turning the window off changes the output.", flush=True)
    _free()


def main():
    if not torch.cuda.is_available():
        sys.exit("No GPU found. Runtime -> Change runtime type -> T4 GPU, then run again.")
    if not OUT.parent.parent.exists():
        sys.exit("Google Drive is not mounted. Run the first cell, then run this cell again.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    import accelerate
    import bitsandbytes
    import transformers
    props = torch.cuda.get_device_properties(0)
    append(dict(key=f"meta|{RUN_ID}", exp="meta", gpu=torch.cuda.get_device_name(0),
                gpu_mem_gib=round(props.total_memory / 2**30, 2), torch=torch.__version__, cuda=torch.version.cuda,
                transformers=transformers.__version__, bitsandbytes=bitsandbytes.__version__,
                accelerate=accelerate.__version__, python=platform.python_version()))
    preflight(torch.device("cuda"))
    last = None
    for i, (name, variant, steps) in enumerate(PLAN):
        todo = pending(name, variant, steps)
        if not todo:
            print(f"== {name}/{variant}: already complete, skipping", flush=True)
        else:
            t0 = time.time()
            print(f"== {name}/{variant}: loading {REPOS[name]}", flush=True)
            tok, model = load(name, variant)
            run_steps(name, variant, tok, model, todo)
            del model
            _free()
            last = name
            print(f"== {name}/{variant}: done in {(time.time() - t0) / 60:.1f} min", flush=True)
        later = {n for n, _, s in PLAN[i + 1:]}
        if name not in later and not os.environ.get("FOLLOWUP_KEEP_MODELS"):
            from huggingface_hub import constants  # free Colab disk space once a model is no longer needed
            shutil.rmtree(Path(constants.HF_HUB_CACHE) / ("models--" + REPOS[name].replace("/", "--")),
                          ignore_errors=True)
    n = sum(r.get("exp") in ("sweep", "grid", "latency") for r in read_rows())
    print(f"\nALL DONE. {n} result rows in {OUT}", flush=True)


# ------------------------------------------------------------------ CPU self-test (no GPU, no downloads)
def self_test():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    global OUT, CHUNK
    OUT = Path(tempfile.mkdtemp()) / "t.jsonl"
    CHUNK = 128
    t = Tokenizer(models.BPE(unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    corpus = [INTRO, QUESTION, NEEDLE.format(key=1234567890)] + SENTENCES
    t.train_from_iterator(corpus * 3, trainers.BpeTrainer(vocab_size=400, special_tokens=["<unk>", "</s>"],
                                                          initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="<unk>", eos_token="</s>")
    for total, dist in ((900, 200), (900, 431), (1500, 777)):
        ids, d1, d2 = build_prompt(tok, total, 48213, 2, distance=dist)
        text = tok.decode(ids[0])
        tail = tok.decode(ids[0, ids.shape[1] - d1:])
        assert ids.shape[1] <= total and total - ids.shape[1] < 12, (total, ids.shape)
        assert abs(d1 - dist) <= 6 and d2 < d1, (dist, d1, d2)
        assert tail.lstrip().startswith("48213") and text.count("48213") == 2, tail[:40]
        assert tok.decode(ids[0, ids.shape[1] - d1 - 4:ids.shape[1] - d1]).count("48213") == 0
    ids, d1, d2 = build_prompt(tok, 1000, 77777, 0, depth=0.5)
    assert 400 < d1 < 600, d1
    assert score(" 48213. The", 48213) and not score(" the password.", 48213) and not score("1 48213", 48213)
    preflight(torch.device("cpu"))
    for window, variant in ((64, "default"), (None, "nowindow")):
        m = tiny_mistral(window)
        run_grid("tiny", variant, tok, m, [300, 500], keys=2)
        run_sweep("tiny", variant, tok, m, 600, 500, 1)
        run_latency("tiny", variant, tok, m, [128, 256])
    rows = read_rows()
    assert all(r["status"] == "ok" for r in rows if r["exp"] in ("grid", "sweep", "latency"))
    w = [r for r in rows if r["exp"] == "grid" and r["variant"] == "default"]
    nw = [r for r in rows if r["exp"] == "grid" and r["variant"] == "nowindow"]
    assert {r["cache_positions"] for r in w} == {63} and min(r["cache_positions"] for r in nw) > 250
    lat = [r for r in rows if r["exp"] == "latency"]
    assert len(lat) == 4 and all(len(r["decode_ms_blocks"]) == BLOCKS and r["concat_ms"] > 0 for r in lat)
    before = len(rows)
    run_grid("tiny", "default", tok, tiny_mistral(64), [300, 500], keys=2)  # resume: nothing new
    assert len(read_rows()) == before
    steps = [("grid", dict(lengths=[300, 500], keys=2)), ("latency", dict(contexts=[128, 256]))]
    assert pending("tiny", "default", steps) == [] and len(pending("tiny", "8bit", steps)) == 2
    print(f"SELF-TEST OK: {len(rows)} rows")


if __name__ == "__main__":
    self_test() if TEST else main()
