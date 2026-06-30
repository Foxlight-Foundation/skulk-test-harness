# CLAUDE.md

Guidance for AI agents working in the Skulk test harness repo.

## Purpose

This repository contains Skulk test harness tooling, scenario runners,
integration fixtures, example profiles, and validation utilities that should
live outside the main Skulk source tree.

Treat `/Users/thomastupper/projects/foxlight/Skulk` as the source of truth for
runtime behavior, commands, and architecture.

## Toolchain

Use `uv`, not bare `pip`:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run basedpyright
git diff --check
```

For docs:

```bash
cd website
npm ci
npm run build
```

For shell wrappers:

```bash
bash -n run_e2e_battery.sh run_mtp_battery.sh run_throughput_battery.sh
bash -n examples/foxlight/run_e2e_battery.sh examples/foxlight/run_mtp_battery.sh examples/foxlight/run_throughput_battery.sh
```

## Conventions

- Always branch and use a pull request. Never push directly to `main`.
- Do not merge a pull request unless the user explicitly asks for a merge.
- Keep public defaults cluster-neutral.
- Keep private cluster configs, generated logs, caches, and local run artifacts
  out of git.
- Put hardware-specific or organization-specific examples under `examples/`.
- Preserve the root battery script names because Foxlight automation depends on
  them.

## Review Loop

Use the Skulk-family review process from `AGENTS.md`.

- Inspect new comments, unresolved review threads, and failing checks.
- Use thread-aware GitHub reads, such as GraphQL `reviewThreads`, because flat
  comment lists can miss inline thread state.
- Score actionable comments on the severity rubric before changing code.
- Fix only severity 4 or 5 comments in the active PR.
- Note severity 3 comments for follow-up.
- Ignore severity 1 or 2 comments unless a maintainer explicitly asks for them.
- Reply with the concrete fix or deferral rationale after validation.
- Resolve only threads actually addressed by code and validation.
- Repeat until no unresolved severity 4 or 5 comments remain.

Draft PRs do not receive the normal review flow. When a PR is ready for review
and the user wants review activity to begin, mark it ready before monitoring.
