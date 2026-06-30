# AGENTS.md

This repository contains Skulk test harness tooling and example profiles.

## Purpose

Use this repo for Skulk test harness tooling, scenario runners, integration
fixtures, and validation utilities that should live outside the main Skulk
source tree.

## Working Notes

- Treat Skulk itself as the source of truth for runtime behavior, commands, and
  architecture.
- Prefer small, reproducible harnesses over ad hoc scripts.
- Keep generated logs, caches, and local run artifacts out of git.
- Always branch and use a pull request. Never push directly to `main`.
- Do not merge a pull request unless the user explicitly asks for a merge.

## Validation Commands

Run these before committing code or docs changes:

```bash
uv run pytest
uv run ruff check .
uv run basedpyright
git diff --check
```

For shell wrapper changes, also run:

```bash
bash -n run_e2e_battery.sh run_mtp_battery.sh run_throughput_battery.sh
bash -n examples/foxlight/run_e2e_battery.sh examples/foxlight/run_mtp_battery.sh examples/foxlight/run_throughput_battery.sh
```

For documentation site changes, also run:

```bash
cd website
npm ci
npm run build
```

## Review Comment Rubric

Use the same Skulk-family review loop as the sibling Skulk repo. Evaluate each
actionable review comment on a 1-5 severity scale before deciding whether to
change code:

- **Likelihood (1-5):** how likely the reported issue is to be real and to
  occur in normal harness or cluster workflows.
- **Impact (1-5):** how serious the result would be for correctness,
  reproducibility, operator time, cluster health, or data.
- **Triggerability (0-2):** `0` common-path, `1` narrower but plausible setup,
  `2` unusual/operator-only chain.
- **Complexity (0-2):** `0` simple failure mode, `1` moderate system knowledge,
  `2` rare timing or deep system knowledge.

Scoring:

1. If likelihood is `1`, its component is `0`; otherwise use `0.5 * likelihood`.
2. If impact is `1`, severity is `1` immediately.
3. Compute `base_score = likelihood_component + (0.5 * impact) - (0.2 * complexity)`.
4. If `base_score < 1.0`, use it directly; otherwise use
   `base_score ** (1 / (1 + (0.25 * triggerability)))`.
5. Map `> 4.5` to severity 5, `> 3.5` to severity 4, `> 2.5` to severity 3,
   `> 1.7` to severity 2, and the rest to severity 1.

Only fix review comments rated severity 4 or 5 in the active PR. Ignore severity
1-2 comments, note severity 3 comments for follow-up, and do not iterate on
minor wording, style, or speculative automated-review suggestions unless a
maintainer explicitly asks for them in the current PR.

When watching a PR:

1. Inspect new comments, unresolved threads, and failing checks.
2. Score each actionable comment with the rubric above.
3. Fix severity 4-5 comments with the smallest correct change.
4. Add or update focused tests for critical-path correctness fixes.
5. Run focused validation before replying.
6. Reply with the concrete fix or deferral rationale.
7. Resolve only threads actually addressed by code and validation.
8. Repeat until no unresolved severity 4-5 comments remain.

Use thread-aware review reads for GitHub PRs. Flat review/comment lists can miss
unresolved inline threads, so prefer GraphQL `reviewThreads` data when deciding
whether feedback remains open.

Draft PRs do not receive the normal review flow. When a PR is ready for review
and the user wants review activity to begin, mark it ready before monitoring.
