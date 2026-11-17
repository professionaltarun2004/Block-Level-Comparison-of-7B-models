# Shared KV Heads, Sliding Windows, and Vocabulary Size

Code, raw results, and paper source for
"Shared KV Heads, Sliding Windows, and Vocabulary Size: Measured Memory and Retrieval
Trade-offs in LLaMA 7B, Mistral 7B, and Qwen2.5-7B"
(B Tarun Sandilya, Durvankur Chavan, Kapardhi Kannekanti).

## Main results (one 16 GB T4 GPU, 4-bit weights)

| Question | Result |
|---|---|
| Where does Mistral 7B v0.1 stop retrieving a passkey? | 72/72 retrieved when the key is within 4,089 tokens of the prompt end, 0/62 when it is 4,098 or more tokens away (window = 4,096). |
| Is the mask the cause? | Yes inside 8K tokens: same weights with `sliding_window=None` retrieve 59/59 at 8K. At 16K and 32K the window-free model fails 30/30, while the windowed model still retrieves keys inside the window. |
| Qwen2.5-7B retrieval | 60/60 from 4K to 32K tokens. |
| KV cache per token | 512 KiB (LLaMA), 128 KiB (Mistral), 56 KiB (Qwen2.5). LLaMA is out of memory at 32K tokens. |
| Tokenization, 15 languages | Qwen2.5 removes 38-58% of tokens for five languages and 3-11% for ten. |
| Decode latency | Grows with cache size; the per-step cache copy alone costs about 18 ms per GiB. |

## Layout

| Path | Content |
|---|---|
| `paper/` | LaTeX source (`main.tex`, `IEEEtran.cls`, figures) and compiled PDF |
| `experiments/t4_experiments.py` | First GPU run: KV-cache memory, peak memory, prefill throughput |
| `experiments/followup_experiments.py` | Passkey grid, distance sweep, window-off and 8-bit variants, latency with cache-copy timing |
| `experiments/tokenization_cost.py` | Token counts on NTREX-128 (CPU only) |
| `notebooks/followup_experiments_T4.ipynb` | Colab notebook that runs the follow-up experiments |
| `results/` | Raw outputs. Every passkey trial includes the generated text |
| `figures/make_figures.py` | Builds `paper/fig_measurements.pdf` from `results/` |

## Reproduce

```bash
pip install -r requirements.txt
python experiments/tokenization_cost.py          # CPU, needs NTREX-128 (see script header)
python experiments/followup_experiments.py       # one T4 GPU, several hours, resumable
python figures/make_figures.py
cd paper && pdflatex main.tex && pdflatex main.tex
```

Versions used: transformers 5.22.2, torch 2.14.1+cu130, bitsandbytes 0.50.2, accelerate 1.15.0, Python 3.13.16.
Model revisions and weight-file SHA-256 hashes are in the `load` rows of `results/results_followup.jsonl`.
You need access to `mistralai/Mistral-7B-v0.1`, `Qwen/Qwen2.5-7B`, and `huggyllama/llama-7b` on the Hugging Face Hub.

## Notes

- `Block_Level_Comparison_v3_final.pdf` and `E2_E4_gpu_experiments_T4_v2 (1).ipynb` are the earlier draft and its notebook.
- The code is released under the MIT License (see `LICENSE`).
