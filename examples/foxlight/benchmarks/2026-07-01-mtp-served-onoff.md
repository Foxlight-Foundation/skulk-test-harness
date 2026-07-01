# Native MTP on-vs-off, served engine (Radeon / kite4), through the production API

Date: 2026-07-01. Node: kite4 (AMD Strix Halo, Radeon 8060S, gfx1151, Vulkan).
Engine: `llama_server` served-backend (llama.cpp `--spec-type draft-mtp`).

This supersedes the methodology of `2026-06-27-mtp-served-radeon.md`, whose
off-arm was a direct `llama-server` run at 500 tokens. Here BOTH arms go through
Skulk's production API with the MLX speculative-decoding protocol (greedy,
200-token completions, median of 3), and the off arm is the IDENTICAL GGUF served
in plain decode via the node env `SKULK_LLAMA_SERVER_FORCE_NO_SPEC=1` (Skulk
PR #434). So the only variable between arms is speculation, and the numbers are
directly comparable to the MLX engine's table.

## Results (median wall_tps of 3 reps)

| Model | Class | Plain (tok/s) | With MTP (tok/s) | Gain |
| --- | --- | ---: | ---: | ---: |
| unsloth/Qwen3.5-9B-MTP | dense, small | 55.64 | 76.16 | 1.37x |
| froggeric/Qwen3.6-27B-MTP | dense, mid | 20.02 | 35.57 | 1.78x |
| unsloth/Qwen3.6-35B-A3B-MTP | MoE (A3B) | 90.69 | 95.81 | 1.06x |
| google/gemma-4-31B (+ draft) | dense, draft-model | 17.38 | 25.16 | 1.45x |

## Reading

- Same shape as the MLX engine and the prior direct-engine run: the dense
  mid-size model gains the most (1.78x, slow base decode gives speculation the
  most to amortize); the MoE gains the least (1.06x, small active-parameter count
  already makes decode memory-bound-fast so per-round overhead nets little).
- Gemma 4 uses the OTHER MTP shape (a separate `--model-draft` GGUF, not baked-in
  heads) and still pays (1.45x), confirming both served MTP shapes work on Radeon.
- Gains here are a touch below the 2026-06-27 note (e.g. 27B 1.78x vs 1.88x)
  because that off-arm was a direct-engine 500-token run; the production-API
  200-token measurement is the more conservative, apples-to-apples number.

## Reproduce

    ./run_mtp_onoff_benchmark.sh

Runs the `mtp-served` model set against the `mtp-benchmark` test set twice on the
same kite4 build, flipping only `SKULK_LLAMA_SERVER_FORCE_NO_SPEC` between arms
(edits the node's systemd EnvironmentFile and restarts), then diffs the two runs
with `combine_mtp_onoff.py` into `runs/mtp_onoff_table.md`. Configure the node
with `BENCH_NODE` (default kite4).
