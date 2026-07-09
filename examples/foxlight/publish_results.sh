#!/usr/bin/env bash
# Publish the harness's accumulated runs into the durable results store and
# trigger the ledger site rebuild, then prune the published local run dirs so
# they do not pile up forever. Called at the end of each battery.
#
# OPT-IN: does nothing unless SKULK_PUBLISH_RESULTS is truthy. This keeps a
# battery on a machine without the ledger checkout behaving exactly as before.
#
#   SKULK_PUBLISH_RESULTS=1   enable (required)
#   SKULK_RESULTS_DATA_DIR    path to the skulk-results-data repo
#                             (default: ../skulk-results-data next to the harness)
#   SKULK_RESULTS_WEB_DIR     path to the skulk-results-ledger-web repo (has
#                             scripts/publish.ts; default: ../skulk-results-ledger-web)
#   SKULK_RESULTS_DEPLOY_REPO GitHub repo to dispatch the Pages deploy on
#                             (default: Foxlight-Foundation/skulk-results-ledger-web;
#                             empty string disables the immediate deploy, leaving
#                             the site's own 6-hourly schedule to pick it up)
#
# Never fails its caller: publishing is post-hoc bookkeeping, so any error here
# is logged and swallowed rather than flipping the battery's exit code.
set -u

_truthy() { case "${1:-}" in 1 | true | yes | on) return 0 ;; *) return 1 ;; esac; }

if ! _truthy "${SKULK_PUBLISH_RESULTS:-}"; then
  echo "[publish] SKULK_PUBLISH_RESULTS not set; skipping results publish."
  exit 0
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_ROOT="$(cd "$HERE/../.." && pwd)"
DATA_DIR="${SKULK_RESULTS_DATA_DIR:-$(cd "$HARNESS_ROOT/.." && pwd)/skulk-results-data}"
WEB_DIR="${SKULK_RESULTS_WEB_DIR:-$(cd "$HARNESS_ROOT/.." && pwd)/skulk-results-ledger-web}"
DEPLOY_REPO="${SKULK_RESULTS_DEPLOY_REPO-Foxlight-Foundation/skulk-results-ledger-web}"

if [ ! -d "$DATA_DIR/reports" ]; then
  echo "[publish] results-data repo not found at $DATA_DIR; skipping." >&2
  exit 0
fi
if [ ! -f "$WEB_DIR/scripts/publish.ts" ]; then
  echo "[publish] publish.ts not found under $WEB_DIR; skipping." >&2
  exit 0
fi

echo "[publish] publishing runs from $HARNESS_ROOT/runs -> $DATA_DIR (push + prune)"
out="$(cd "$WEB_DIR" && npx --yes tsx scripts/publish.ts \
  --data "$DATA_DIR" --runs "$HARNESS_ROOT/runs" --push --prune 2>&1)"
rc=$?
echo "$out"
if [ "$rc" -ne 0 ]; then
  echo "[publish] publish step failed (non-fatal); site keeps its last data." >&2
  exit 0
fi

# Immediate site rebuild so new findings surface in ~a minute instead of waiting
# for the 6-hourly schedule. Only when something actually got pushed, so a
# re-run (or a stability-only battery) does not trigger an empty rebuild.
# Best-effort: needs an authenticated gh.
if echo "$out" | grep -q "Committed + pushed"; then
  if [ -n "$DEPLOY_REPO" ] && command -v gh >/dev/null 2>&1; then
    if gh workflow run deploy.yml --repo "$DEPLOY_REPO" >/dev/null 2>&1; then
      echo "[publish] triggered ledger deploy on $DEPLOY_REPO"
    else
      echo "[publish] deploy dispatch failed (non-fatal); 6-hourly schedule will catch it." >&2
    fi
  fi
else
  echo "[publish] nothing new to publish; no deploy triggered."
fi
exit 0
