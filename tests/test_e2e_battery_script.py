"""Regression tests for the full E2E battery wrapper."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


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
