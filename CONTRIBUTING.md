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

## Pull Requests

- Keep default configs cluster-neutral.
- Put hardware-, lab-, or organization-specific runs under `examples/`.
- Add focused tests for behavior changes.
- Build the docs site when changing `website/`.
- Do not commit generated `runs/`, caches, logs, or local configs.
