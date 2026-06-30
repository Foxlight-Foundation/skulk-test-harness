from pathlib import Path

from skulk_test_harness.chaos import (
    KILL_COMMAND,
    _friendly_for_node_from_state,
    _kill_command_for_node,
    _node_for_friendly_from_state,
    _present_node_ids,
    _relaunch_command,
    _relaunch_command_for_node,
    _ssh_argv,
)
from skulk_test_harness.models import ClusterNode, HarnessConfig
from skulk_test_harness.specs import load_config


def test_ssh_argv_uses_batchmode_and_connect_timeout() -> None:
    argv = _ssh_argv("node-a", KILL_COMMAND)

    assert argv[0] == "ssh"
    assert "BatchMode=yes" in argv
    assert "ConnectTimeout=6" in argv
    # Host comes before the command, command is the final token.
    assert argv[-2] == "node-a"
    assert argv[-1] == KILL_COMMAND


def test_relaunch_command_cds_and_detaches() -> None:
    command = _relaunch_command("/repo/Skulk")

    assert command.startswith("cd /repo/Skulk &&")
    assert "nohup uv run skulk -v" in command
    assert command.rstrip().endswith("disown")


def test_kill_command_is_force_kill() -> None:
    assert KILL_COMMAND == 'pkill -9 -f "uv run skulk"'


def _state_with_identities() -> dict[str, object]:
    return {
        "lastSeen": {"node-a": "2026-06-15T00:00:00Z", "node-b": "2026-06-15T00:00:00Z"},
        "nodeIdentities": {
            "node-a": {"friendlyName": "worker-a"},
            "node-b": {"friendlyName": "worker-b"},
        },
    }


def test_friendly_for_node_maps_known_node() -> None:
    assert _friendly_for_node_from_state(_state_with_identities(), "node-a") == "worker-a"


def test_friendly_for_node_falls_back_to_node_id() -> None:
    assert _friendly_for_node_from_state(_state_with_identities(), "node-z") == "node-z"


def test_node_for_friendly_reverse_maps() -> None:
    assert _node_for_friendly_from_state(_state_with_identities(), "worker-b") == "node-b"


def test_node_for_friendly_returns_none_when_absent() -> None:
    assert _node_for_friendly_from_state(_state_with_identities(), "ghost") is None


def test_present_node_ids_from_last_seen() -> None:
    assert _present_node_ids(_state_with_identities()) == {"node-a", "node-b"}


def test_present_node_ids_handles_missing_last_seen() -> None:
    assert _present_node_ids({}) == set()


def test_config_loads_cluster_nodes(tmp_path: Path) -> None:
    config_path = tmp_path / "harness.yaml"
    config_path.write_text(
        "api_base_url: http://node-a:52415\n"
        "cluster_nodes:\n"
        "  node-a: { ssh_host: node-a, repo_path: /repo/Skulk }\n"
        "  node-b:\n"
        "    ssh_host: node-b-alias\n"
        "    kill_command: stop-skulk\n"
        "    relaunch_command: start-skulk\n"
    )

    config = load_config(config_path)

    assert config.cluster_nodes == {
        "node-a": ClusterNode(ssh_host="node-a", repo_path="/repo/Skulk"),
        "node-b": ClusterNode(
            ssh_host="node-b-alias",
            kill_command="stop-skulk",
            relaunch_command="start-skulk",
        ),
    }


def test_config_defaults_to_empty_cluster_nodes() -> None:
    assert HarnessConfig().cluster_nodes == {}


def test_configured_cluster_node_commands_override_defaults() -> None:
    node = ClusterNode(
        ssh_host="node-a",
        kill_command="systemctl --user stop skulk",
        relaunch_command="systemctl --user start skulk",
    )

    assert _kill_command_for_node(node) == "systemctl --user stop skulk"
    assert _relaunch_command_for_node(node) == "systemctl --user start skulk"


def test_relaunch_command_falls_back_to_repo_path() -> None:
    node = ClusterNode(ssh_host="node-a", repo_path="/repo/Skulk")

    command = _relaunch_command_for_node(node)

    assert command is not None
    assert command.startswith("cd /repo/Skulk &&")


def test_relaunch_command_missing_when_no_command_or_repo_path() -> None:
    assert _relaunch_command_for_node(ClusterNode(ssh_host="node-a")) is None
