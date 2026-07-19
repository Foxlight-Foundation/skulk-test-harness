# Contributing

Thanks for helping improve the Skulk test harness.

## Development Setup

```bash
uv sync
uv run pytest
uv run ruff check .
uv run basedpyright
```

## Documentation Setup

The Docusaurus site lives in `website/`.

```bash
cd website
npm ci
npm run start
npm run build
```

Keep user-facing documentation beginner-friendly. Prefer concrete examples,
tables, and diagrams when they make the harness easier to understand.

## Local Harness Config

Copy `skulk-harness.example.yaml` to `skulk-harness.yaml` for local use. The
local file is ignored by git so contributors can point at their own Skulk node.

## Versioning

The harness follows [semantic versioning](https://semver.org/):

- **Major**: breaking changes to the CLI surface, config schema, or the
  `report.json` contract that existing consumers cannot read.
- **Minor**: new commands, flags, test kinds, or report fields that stay
  backward compatible.
- **Patch**: bug fixes and documentation-only changes.

Each release gets a git tag (`vX.Y.Z`) on the release commit, created at
merge time. Every user-visible change lands with an entry under
`[Unreleased]` in `CHANGELOG.md` ([Keep a Changelog](https://keepachangelog.com/)
format); cutting a release rolls `[Unreleased]` into a dated version section
and bumps `version` in `pyproject.toml` (the package version is the single
source of truth; `skulk_test_harness.__version__` reads it from installed
metadata).

## Pull Requests

- Keep default configs cluster-neutral.
- Put hardware-, lab-, or organization-specific runs under `examples/`.
- Add focused tests for behavior changes.
- Build the docs site when changing `website/`.
- Do not commit generated `runs/`, caches, logs, or local configs.
