# Changelog

All notable changes to skulk-test-harness are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Align Foxlight speech qualification with Skulk's managed-reference voice
  catalog and Qwen Base 6-bit card, including typed voice metadata, TTS language
  propagation, and dashboard assertions that pin voice and language across
  separately synthesized assistant sentences.

- Documented the handoff from an all-green automated candidate matrix to human
  first-install acceptance. Human testing supplements the physical-fleet and
  clean RunPod verdict; it cannot replace either leg, and a product/default/
  installer/dashboard/model-card fix creates a new candidate that must repeat
  fresh-install qualification.

- Physical fresh-install release qualification now installs every explicitly
  declared fleet member together with normal networking, requires the exact
  topology and all-member commit/backend/transport contracts throughout model
  journeys, and restores the complete original fleet before releasing its
  lease. Sandboxed single-node legs remain diagnostic-only.

### Fixed

- Authoritative fleet-lease reads now retry bounded, read-only Git fetch
  transport failures before failing closed, so one transient 30-second fetch
  timeout cannot abort and restore an otherwise healthy multi-hour
  qualification run.
- Fresh-install restoration now suppresses the supervised service's automatic
  update only for the recovery start and reapplies archived configuration after
  API readiness, preserving the exact pre-run commit and config bytes. Eligible
  launchd/systemd targets must archive their `skulk.env` so this cannot silently
  become a no-op.
- Randomized vision qualification still rejects unbounded or immediately
  repeated output, but accepts a correct exact final answer after a bounded
  explanatory response.
- Fresh-install temporary homes now live on each target user's home filesystem
  instead of `/tmp`, preventing Linux tmpfs capacity from producing false
  low-storage warnings or model-download failures that a real new user would
  not experience.

## [0.2.0] - 2026-07-19

The first versioned cut of the harness since the initial 0.1.0 (which was
never tagged). It gathers everything the harness grew while feeding the
public Skulk benchmarks ledger.

### Added

- Report fingerprint schema 2.2: every `report.json` carries a
  self-describing block with source context (git provenance), harness
  runtime, cluster hardware per node, and cache/warmth conditions, so a
  measurement is never separated from what produced it.
- `skulk-harness fleet acquire` / `extend` / `release` / `status`: an
  optional git-backed fleet lease that gives one operator or agent exclusive
  access to a shared test fleet. Execute-mode commands refuse to run while
  another holder has the lease.
- `skulk-harness submit`: one-command submission of a finished run to the
  community benchmarks ledger, with client-side slimming and redaction
  (generated text, operator notes, run-name labels, repo paths, API URLs,
  and node names never leave the machine) and a `--dry-run` payload preview.
- Stability suites (`stability soak` / `failover` / `churn` / `refusal`)
  for operational behavior: sustained concurrent load, master crash
  mid-stream, repeated node crash/relaunch rounds, and impossible-placement
  refusal, with destructive suites gated behind `--execute-destructive`.
- `skulk-harness compare`: like-for-like deltas between two run sets with
  trust guards that call out unfair comparisons (node-set or cache
  mismatch, low sample counts, short-output noise).
- Concurrent load test kind (`kind: concurrent`) with aggregate-throughput
  and per-request latency distributions under load.
- `skulk-harness --version` prints the installed harness version.

### Fixed

- Concurrent-measurement honesty (issues #69, #70, #71; PR #72): the async
  HTTP client is force-closed so a stalled stream cannot leak sockets into
  the next request; the concurrent driver runs all workers on a single
  event loop with symmetric token accounting, so aggregate throughput
  counts tokens from every request that returned them regardless of scoring
  outcome; and per-model readiness waiting has a bounded total ceiling
  across replacement instances, so instance churn fails loudly instead of
  waiting forever.

## [0.1.0] - 2026-06-15

Initial release: an agent-controlled end-to-end test and benchmark harness
for Skulk clusters. Named model sets and test sets, dry-run-by-default
`run`/`plan`/`goal` commands, placement via the Skulk API, real
chat/tool/vision/speech requests with correctness scoring, TTFT and decode
throughput measurement, and per-run report directories (`report.json`,
`summary.md`, `events.jsonl`, artifacts).
