#!/usr/bin/env bash
# Publish the harness's accumulated runs into the durable results store and
# trigger the ledger site rebuild, then prune the published local run dirs so
# they do not pile up forever. Called at the end of each battery.
#
# ON BY DEFAULT: publishing is the normal path, so the ledger stays fresh with
# no per-machine setup. Turn it off INTENTIONALLY by EITHER:
#   - SKULK_PUBLISH_RESULTS falsy (0/false/no/off) in the environment, OR
#   - a marker file `.autopublish-results-off` at the harness repo root. This is
#     the "hold publishing on this machine" switch (gitignored, machine-local),
#     e.g. while dev churn on the fleet should stay out of the public ledger.
# SKULK_PUBLISH_RESULTS truthy (1/true/yes/on) forces publishing on even when
# the off-marker exists. Machines without the sibling skulk-results-data /
# skulk-results-ledger-web checkouts (kites, CI) skip automatically below, so
# default-on cannot publish from a machine that was never set up to publish.
#
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
_falsy() { case "${1:-}" in 0 | false | no | off) return 0 ;; *) return 1 ;; esac; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_ROOT="$(cd "$HERE/../.." && pwd)"

if _falsy "${SKULK_PUBLISH_RESULTS:-}"; then
  echo "[publish] disabled via SKULK_PUBLISH_RESULTS=${SKULK_PUBLISH_RESULTS:-}; skipping."
  exit 0
fi
if [ -f "$HARNESS_ROOT/.autopublish-results-off" ] && ! _truthy "${SKULK_PUBLISH_RESULTS:-}"; then
  echo "[publish] disabled via .autopublish-results-off marker; skipping."
  exit 0
fi

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
