# Contributing

Thanks for helping improve the Skulk test harness.

## Development Setup

```bash
uv sync
uv run pytest
uv run ruff check .
uv run basedpyright
```

## Local Harness Config

Copy `skulk-harness.example.yaml` to `skulk-harness.yaml` for local use. The
local file is ignored by git so contributors can point at their own Skulk node.

## Pull Requests

- Keep default configs cluster-neutral.
- Put hardware-, lab-, or organization-specific runs under `examples/`.
- Add focused tests for behavior changes.
- Do not commit generated `runs/`, caches, logs, or local configs.
