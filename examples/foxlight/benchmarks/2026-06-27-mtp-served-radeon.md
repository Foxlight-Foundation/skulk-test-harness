# Native MTP (llama_server / draft-mtp) throughput on Radeon (kite4)

Date: 2026-06-27. Node: kite4 (AMD Strix Halo, Radeon 8060S, Vulkan).
Engine: `llama_server` served-backend (llama.cpp `--spec-type draft-mtp`),
llama.cpp build b9820. Quant: Q4_K_M. Greedy (temp 0), 500-token generation,
8192 ctx, `--spec-draft-n-max 3`.

## MTP-on vs MTP-off (direct llama-server, per-model)

This is the speedup measurement (the harness throughput cell can only measure the
production MTP-on rate, so on/off is captured at the engine).

| Model | Class | MTP tok/s | accept | baseline tok/s | speedup |
| --- | --- | --- | --- | --- | --- |
| unsloth/Qwen3.5-9B-MTP | dense, small | 60.31 | 62.9% | 37.37 | 1.61x |
| froggeric/Qwen3.6-27B-MTP | dense, mid | 24.03 | 57.7% | 12.76 | 1.88x |
| unsloth/Qwen3.6-35B-A3B-MTP | MoE (A3B) | 72.28 | 58.3% | 59.81 | 1.21x |

## Reading

- Native MTP works across Qwen3.5/3.6 dense + MoE on the Radeon/Vulkan backend.
- The MoE (35B-A3B) gets the **least** speedup (1.21x) despite comparable
  acceptance: A3B's small active-parameter count already makes decode fast
  (memory-bound), so the per-round draft+verify overhead nets less. Dense models
  with more active params (slower base decode) benefit more (1.6-1.9x). This
  matches the public RTX-3090 "MoE no/low speedup" observation, now confirmed on
  Radeon. Acceptance alone does not predict speedup; the base decode cost does.

## Not included

- jamesdumay/GLM-4.7-Flash-MTP-GGUF: the tested upload fails to load on b9820
  ("wrong number of tensors; expected 868, got 862") -- the GGUF is missing its
  MTP tensors. A verified non-Qwen MTP GGUF is a follow-up.

## Reproduce (through the model store + harness)

    ./run_mtp_battery.sh

Runs the `mtp-served` model-set against the `throughput` test set with
`--ensure-store-downloads` (stage from the store) `--delete-created-instances`
`--delete-staged-models` (evict staged weights after). That measures wall_tps for
the production MTP-on served path through Skulk and cleans up after.
