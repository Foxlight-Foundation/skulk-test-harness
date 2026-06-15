from pathlib import Path

from skulk_test_harness.chaos import (
    KILL_COMMAND,
    _friendly_for_node_from_state,
    _node_for_friendly_from_state,
    _present_node_ids,
    _relaunch_command,
    _ssh_argv,
)
from skulk_test_harness.models import ClusterNode, HarnessConfig
from skulk_test_harness.specs import load_config


def test_ssh_argv_uses_batchmode_and_connect_timeout() -> None:
    argv = _ssh_argv("kite1", KILL_COMMAND)

    assert argv[0] == "ssh"
    assert "BatchMode=yes" in argv
    assert "ConnectTimeout=6" in argv
    # Host comes before the command, command is the final token.
    assert argv[-2] == "kite1"
    assert argv[-1] == KILL_COMMAND


def test_relaunch_command_cds_and_detaches() -> None:
    command = _relaunch_command("/Users/kite3/projects/foxlight/Skulk")

    assert command.startswith("cd /Users/kite3/projects/foxlight/Skulk &&")
    assert "nohup uv run skulk -v" in command
    assert command.rstrip().endswith("disown")


def test_kill_command_is_force_kill() -> None:
    assert KILL_COMMAND == 'pkill -9 -f "uv run skulk"'


def _state_with_identities() -> dict[str, object]:
    return {
        "lastSeen": {"node-a": "2026-06-15T00:00:00Z", "node-b": "2026-06-15T00:00:00Z"},
        "nodeIdentities": {
            "node-a": {"friendlyName": "kite1"},
            "node-b": {"friendlyName": "kite2"},
        },
    }


def test_friendly_for_node_maps_known_node() -> None:
    assert _friendly_for_node_from_state(_state_with_identities(), "node-a") == "kite1"


def test_friendly_for_node_falls_back_to_node_id() -> None:
    assert _friendly_for_node_from_state(_state_with_identities(), "node-z") == "node-z"


def test_node_for_friendly_reverse_maps() -> None:
    assert _node_for_friendly_from_state(_state_with_identities(), "kite2") == "node-b"


def test_node_for_friendly_returns_none_when_absent() -> None:
    assert _node_for_friendly_from_state(_state_with_identities(), "ghost") is None


def test_present_node_ids_from_last_seen() -> None:
    assert _present_node_ids(_state_with_identities()) == {"node-a", "node-b"}


def test_present_node_ids_handles_missing_last_seen() -> None:
    assert _present_node_ids({}) == set()


def test_config_loads_cluster_nodes(tmp_path: Path) -> None:
    config_path = tmp_path / "harness.yaml"
    config_path.write_text(
        "api_base_url: http://kite1:52415\n"
        "cluster_nodes:\n"
        "  kite1: { ssh_host: kite1, repo_path: /repo/Skulk }\n"
        "  kite3: { ssh_host: kite3-alias, repo_path: /other/Skulk }\n"
    )

    config = load_config(config_path)

    assert config.cluster_nodes == {
        "kite1": ClusterNode(ssh_host="kite1", repo_path="/repo/Skulk"),
        "kite3": ClusterNode(ssh_host="kite3-alias", repo_path="/other/Skulk"),
    }


def test_config_defaults_to_empty_cluster_nodes() -> None:
    assert HarnessConfig().cluster_nodes == {}
