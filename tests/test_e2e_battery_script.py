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
            "SKULK_E2E_DELETE_STAGED_MODELS": "1",
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
    assert "--delete-staged-models" in calls[0]
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

    assert [
        "cell",
        "concurrency-mlx",
        "concurrency-8",
        "--max-nodes 1",
    ] in cells
    assert [
        "cell",
        "concurrency-mlx-reasoning",
        "concurrency-reasoning-8",
        "--max-nodes 1",
    ] in cells
    assert [
        "cell",
        "concurrency-mlx-multinode",
        "concurrency-8",
        "--sharding Tensor --min-nodes 2 --max-nodes 2",
    ] in cells
    assert [
        "cell",
        "concurrency-gguf",
        "concurrency",
        "--max-nodes 1",
    ] in cells
    assert [
        "cell",
        "concurrency-120b",
        "concurrency-reasoning",
        "--max-nodes 1",
    ] in cells
    assert [
        "cell",
        "concurrency-gguf-pooled",
        "concurrency-reasoning",
        "--min-nodes 2 --max-nodes 2 --instance-meta LlamaRpc",
    ] in cells

    test_sets = load_test_sets(root / "examples" / "foxlight" / "test_sets.yaml")
    for suite_name in ("concurrency-16", "concurrency-reasoning-16"):
        levels = [
            test.concurrency for test in test_sets.test_sets[suite_name].tests
        ]
        assert levels == [1, 4, 8, 16]
    for suite_name in ("concurrency-8", "concurrency-reasoning-8"):
        levels = [
            test.concurrency for test in test_sets.test_sets[suite_name].tests
        ]
        assert levels == [1, 4, 8]
    for suite_name in ("concurrency", "concurrency-reasoning"):
        levels = [
            test.concurrency for test in test_sets.test_sets[suite_name].tests
        ]
        assert levels == [1, 4, 8, 16, 32, 64]
    assert all(
        test.success.min_chars == 1 and test.success.min_generated_chars == 500
        for suite_name in ("concurrency-8", "concurrency-16", "concurrency")
        for test in test_sets.test_sets[suite_name].tests
    )
    assert all(
        test.max_tokens == 1536 and test.success.min_chars == 500
        for suite_name in (
            "concurrency-reasoning-8",
            "concurrency-reasoning-16",
            "concurrency-reasoning",
        )
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
    assert (
        "mlx-community/Qwen3-30B-A3B-4bit" not in model_sets["concurrency-mlx"].models
    )
    assert (
        "mlx-community/Moonlight-16B-A3B-Instruct-4-bit"
        in model_sets["concurrency-mlx"].models
    )
    assert model_sets["concurrency-mlx-multinode"].models == [
        "mlx-community/Qwen3.5-9B-4bit"
    ]


def test_vision_data_plane_cells_respect_family_placement_contracts() -> None:
    """Split distributed-capable VLMs from Gemma 4 default placement."""
    root = Path(__file__).resolve().parents[1]
    script = root / "examples" / "foxlight" / "run_e2e_battery.sh"
    cells = [
        shlex.split(line.strip())
        for line in script.read_text().splitlines()
        if line.strip().startswith("cell vision-")
    ]

    assert [
        "cell",
        "vision-multinode",
        "vision-data-plane",
        "--min-nodes 2",
    ] in cells
    assert [
        "cell",
        "vision-default-placement",
        "vision-data-plane",
    ] in cells

    model_sets = load_model_sets(
        root / "examples" / "foxlight" / "model_sets.yaml"
    ).model_sets
    assert model_sets["vision-multinode"].models == [
        "mlx-community/Qwen3-VL-4B-Instruct-4bit",
        "mlx-community/Qwen3.5-2B-4bit",
        "mlx-community/gemma-3n-E2B-it-4bit",
    ]
    assert model_sets["vision-default-placement"].models == [
        "mlx-community/gemma-4-e2b-it-8bit"
    ]


def test_translation_cell_uses_stt_fixture_instead_of_multilingual_tts() -> None:
    """Keep release translation coverage within the shipped speech contract."""

    root = Path(__file__).resolve().parents[1]
    script = root / "examples" / "foxlight" / "run_e2e_battery.sh"

    assert "cell speech-translation-stt speech-translation" in script.read_text()
    model_sets = load_model_sets(
        root / "examples" / "foxlight" / "model_sets.yaml"
    ).model_sets
    assert "speech-translation-tts" not in model_sets
    assert model_sets["speech-translation-stt"].models == [
        "CogniSoftOrg/canary-1b-v2-mlx-bf16"
    ]


FRESH_FLEET_DROPPED_CELLS = (
    'cell pooled-rpc       llama-cpp        "--min-nodes 2 --instance-meta LlamaRpc"',
    'cell concurrency-120b           concurrency-reasoning  "--max-nodes 1"',
    "cell concurrency-gguf-pooled    concurrency-reasoning  "
    '"--min-nodes 2 --max-nodes 2 --instance-meta LlamaRpc"',
)


def _cell_commands(script: Path) -> list[list[str]]:
    """Return every battery cell invocation in script order."""
    return [
        shlex.split(line.strip())
        for line in script.read_text().splitlines()
        if line.strip().startswith("cell ")
    ]


def _shell_code(script: Path) -> list[str]:
    """Return the script's shell code without comments, cells, or blank lines."""
    return [
        line
        for line in script.read_text().splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
        and not line.strip().startswith("cell ")
    ]


def test_fresh_fleet_battery_tracks_the_full_battery() -> None:
    """Keep the fresh-fleet battery the full battery minus only its huge models.

    A freshly installed fleet elects its store host at random, so the
    fresh-fleet variant leaves out the cells that download a 40 GB-plus GGUF.
    Every other cell, flag, and line of shell machinery must stay identical, so
    an edit to the full battery fails here until the variant follows it.
    """
    root = Path(__file__).resolve().parents[1] / "examples" / "foxlight"
    full_script = root / "run_e2e_battery.sh"
    fresh_script = root / "run_e2e_battery_fresh_fleet.sh"
    dropped = [shlex.split(line) for line in FRESH_FLEET_DROPPED_CELLS]

    assert all(command in _cell_commands(full_script) for command in dropped)
    expected = [
        ["cell", "gguf-big-fresh", *command[2:]]
        if command[:2] == ["cell", "gguf-big"]
        else command
        for command in _cell_commands(full_script)
        if command not in dropped
    ]
    assert _cell_commands(fresh_script) == expected
    assert _shell_code(fresh_script) == _shell_code(full_script)


def test_fresh_fleet_gguf_set_drops_only_the_huge_models() -> None:
    """The fresh-fleet GGUF set is gguf-big without its two 40 GB-plus models."""
    root = Path(__file__).resolve().parents[1]
    model_sets = load_model_sets(
        root / "examples" / "foxlight" / "model_sets.yaml"
    ).model_sets
    huge = {
        "bartowski/Llama-3.3-70B-Instruct-GGUF",
        "bartowski/openai_gpt-oss-120b-GGUF",
    }

    assert huge <= set(model_sets["gguf-big"].models)
    assert model_sets["gguf-big-fresh"].models == [
        model for model in model_sets["gguf-big"].models if model not in huge
    ]
