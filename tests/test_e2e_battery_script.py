"""Regression tests for the full E2E battery wrapper."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest

from skulk_test_harness.specs import load_model_sets, load_test_sets


def test_e2e_battery_stops_when_a_cell_is_interrupted(tmp_path: Path) -> None:
    """An interrupted child must stop the battery instead of starting later cells."""

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls_path = tmp_path / "uv-calls.txt"
    log_path = tmp_path / "e2e-battery.log"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "#!/bin/sh\n"
        "if [ \"$2\" = \"skulk-harness\" ] && [ \"$3\" = \"doctor\" ]; then\n"
        "  echo 'API available'\n"
        "  exit 0\n"
        "fi\n"
        "echo \"$*\" >> \"$FAKE_UV_CALLS\"\n"
        "exit 130\n"
    )
    fake_uv.chmod(0o755)
    repo_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.update(
        {
            "FAKE_UV_CALLS": str(calls_path),
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "SKULK_E2E_BATTERY_LOG": str(log_path),
            "SKULK_PUBLISH_RESULTS": "0",
        }
    )

    completed = subprocess.run(
        ["bash", "examples/foxlight/run_e2e_battery.sh"],
        cwd=repo_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 130
    calls = calls_path.read_text().splitlines()
    assert len(calls) == 1
    assert "--model-set dense-singles" in calls[0]
    assert "BATTERY INTERRUPTED (rc=130)" in completed.stdout


@pytest.mark.parametrize(
    "script_name",
    ["run_e2e_battery.sh", "run_concurrency_battery.sh"],
)
def test_mlx_concurrency_cells_stop_at_runtime_cap(script_name: str) -> None:
    """Route each model to its engine cap and required output budget."""
    root = Path(__file__).resolve().parents[1]
    script = root / "examples" / "foxlight" / script_name
    cells = [
        shlex.split(line.strip())
        for line in script.read_text().splitlines()
        if line.strip().startswith("cell concurrency-")
    ]

    assert ["cell", "concurrency-mlx", "concurrency-16"] in cells
    assert [
        "cell",
        "concurrency-mlx-reasoning",
        "concurrency-reasoning-16",
    ] in cells
    assert any(
        cell[:3] == ["cell", "concurrency-mlx-multinode", "concurrency-16"]
        for cell in cells
    )
    assert ["cell", "concurrency-gguf", "concurrency"] in cells
    assert ["cell", "concurrency-120b", "concurrency-reasoning"] in cells
    assert any(
        cell[:3]
        == ["cell", "concurrency-gguf-pooled", "concurrency-reasoning"]
        for cell in cells
    )

    test_sets = load_test_sets(root / "examples" / "foxlight" / "test_sets.yaml")
    for suite_name in ("concurrency-16", "concurrency-reasoning-16"):
        levels = [
            test.concurrency for test in test_sets.test_sets[suite_name].tests
        ]
        assert levels == [1, 4, 8, 16]
    for suite_name in ("concurrency", "concurrency-reasoning"):
        levels = [
            test.concurrency for test in test_sets.test_sets[suite_name].tests
        ]
        assert levels == [1, 4, 8, 16, 32, 64]
    assert all(
        test.success.min_chars == 1 and test.success.min_generated_chars == 500
        for suite_name in ("concurrency-16", "concurrency")
        for test in test_sets.test_sets[suite_name].tests
    )
    assert all(
        test.max_tokens == 1536 and test.success.min_chars == 500
        for suite_name in ("concurrency-reasoning-16", "concurrency-reasoning")
        for test in test_sets.test_sets[suite_name].tests
    )

    model_sets = load_model_sets(
        root / "examples" / "foxlight" / "model_sets.yaml"
    ).model_sets
    mlx_reasoning_model = "mlx-community/gpt-oss-20b-MXFP4-Q8"
    gguf_reasoning_model = "bartowski/openai_gpt-oss-120b-GGUF"
    assert mlx_reasoning_model not in model_sets["concurrency-mlx"].models
    assert model_sets["concurrency-mlx-reasoning"].models == [mlx_reasoning_model]
    assert gguf_reasoning_model not in model_sets["concurrency-gguf"].models
    assert model_sets["concurrency-120b"].models == [gguf_reasoning_model]
