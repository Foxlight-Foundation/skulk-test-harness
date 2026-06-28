#!/usr/bin/env bash
# Native-MTP throughput battery: the `mtp-served` model set (llama.cpp draft-mtp
# via the llama_server engine) against the `throughput` test set, so wall_tps is a
# steady-state served-MTP decode rate on the GPU node (kite4). Each model:
#   --ensure-store-downloads  stage the GGUF from the model store (not hand-placed)
#   --delete-created-instances tear the served instance down after its run
#   --delete-staged-models     evict the staged weights so test models do not
#                              accumulate on disk (benchmark hygiene)
set -u
cd "$(dirname "$0")"
LOG=runs/mtp_battery.log
: > "$LOG"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

say "MTP BATTERY START"
uv run skulk-harness run \
  --model-set mtp-served \
  --test-set throughput \
  --execute \
  --ensure-store-downloads \
  --delete-created-instances \
  --delete-staged-models >>"$LOG" 2>&1
say "MTP BATTERY END (rc=$?)"
