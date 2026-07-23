"""Clean, cost-bounded RunPod provisioning with confirmed teardown."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

from skulk_test_harness.models import RunPodFreshInstallConfig

_RUNPOD_API = "https://rest.runpod.io/v1"


@dataclass(frozen=True)
class RunPodLease:
    """Minimum private pod metadata needed by the controller."""

    pod_id: str
    hourly_cost_usd: float


@dataclass(frozen=True)
class RunPodSshEndpoint:
    """Ephemeral SSH control surface for a running pod."""

    host: str
    port: int


class RunPodClient:
    """Small RunPod REST client restricted to ephemeral qualification pods."""

    def __init__(
        self,
        config: RunPodFreshInstallConfig,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        api_key = os.environ.get(config.api_key_environment)
        if not api_key:
            raise ValueError(
                f"missing RunPod credential environment "
                f"{config.api_key_environment!r}"
            )
        self._owned_client = client is None
        self._client = client or httpx.Client(
            base_url=_RUNPOD_API,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )

    def close(self) -> None:
        """Close the owned HTTP client."""

        if self._owned_client:
            self._client.close()

    def __enter__(self) -> "RunPodClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def provision(self, *, qualification_id: str) -> RunPodLease:
        """Create a clean pod with no network volume or Skulk image."""

        public_key = self.config.ssh_public_key_file.expanduser().read_text().strip()
        if not public_key:
            raise ValueError("RunPod SSH public key file is empty")
        response = self._client.post(
            "/pods",
            json={
                "name": f"skulk-fresh-{qualification_id[:24]}",
                "imageName": self.config.image_name,
                "computeType": "GPU",
                "cloudType": self.config.cloud_type,
                "gpuCount": 1,
                "gpuTypeIds": self.config.gpu_type_ids,
                "gpuTypePriority": "availability",
                "containerDiskInGb": self.config.container_disk_gb,
                "volumeInGb": 0,
                "ports": ["22/tcp"],
                "interruptible": False,
                "env": {"QUALIFICATION_SSH_PUBLIC_KEY": public_key},
                "dockerEntrypoint": ["/bin/bash", "-lc"],
                "dockerStartCmd": [_sshd_bootstrap_command()],
            },
        )
        response.raise_for_status()
        payload = _object(response.json())
        pod_id = payload.get("id")
        cost = _number(payload.get("adjustedCostPerHr", payload.get("costPerHr")))
        if not isinstance(pod_id, str) or cost is None:
            raise RuntimeError("RunPod create response omitted pod id or hourly cost")
        lease = RunPodLease(pod_id=pod_id, hourly_cost_usd=cost)
        if cost > self.config.maximum_hourly_cost_usd:
            self.terminate_and_confirm(pod_id)
            raise RuntimeError(
                "RunPod hourly cost exceeds configured qualification ceiling"
            )
        if self.config.require_no_network_volume and payload.get("networkVolume"):
            self.terminate_and_confirm(pod_id)
            raise RuntimeError("RunPod unexpectedly attached a network volume")
        return lease

    def wait_for_ssh(self, pod_id: str) -> RunPodSshEndpoint:
        """Wait until the clean pod exposes its mapped SSH port."""

        deadline = time.monotonic() + self.config.readiness_timeout_s
        while time.monotonic() < deadline:
            payload = self.get(pod_id)
            host = payload.get("publicIp")
            mappings = payload.get("portMappings")
            raw_port = mappings.get("22") if isinstance(mappings, dict) else None
            port = _integer(raw_port)
            if (
                payload.get("desiredStatus") == "RUNNING"
                and isinstance(host, str)
                and host
                and port is not None
            ):
                return RunPodSshEndpoint(host=host, port=port)
            time.sleep(self.config.poll_interval_s)
        raise TimeoutError("RunPod did not expose SSH before the readiness deadline")

    def get(self, pod_id: str) -> dict[str, object]:
        """Return the current provider record for one qualification pod."""

        response = self._client.get(f"/pods/{pod_id}")
        response.raise_for_status()
        return _object(response.json())

    def terminate_and_confirm(self, pod_id: str) -> None:
        """Delete a pod and poll until the provider confirms it no longer exists."""

        response = self._client.delete(f"/pods/{pod_id}")
        if response.status_code not in {204, 404}:
            response.raise_for_status()
        deadline = time.monotonic() + max(60.0, self.config.readiness_timeout_s)
        while time.monotonic() < deadline:
            probe = self._client.get(f"/pods/{pod_id}")
            if probe.status_code == 404:
                return
            if probe.status_code >= 400:
                probe.raise_for_status()
            time.sleep(self.config.poll_interval_s)
        raise TimeoutError("RunPod still existed after the teardown deadline")


def _sshd_bootstrap_command() -> str:
    """Return a neutral-container bootstrap that only enables SSH control."""

    return (
        "set -e; "
        "apt-get update -qq; "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
        "openssh-server ca-certificates curl git; "
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -; "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nodejs; "
        "mkdir -p /run/sshd /root/.ssh; "
        'printf "%s\\n" "$QUALIFICATION_SSH_PUBLIC_KEY" '
        "> /root/.ssh/authorized_keys; "
        "chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; "
        "exec /usr/sbin/sshd -D -e"
    )


def _object(value: object) -> dict[str, object]:
    """Validate a JSON object without allowing Any to spread."""

    if not isinstance(value, dict):
        raise TypeError("expected RunPod API to return an object")
    return {str(key): item for key, item in value.items()}


def _number(value: object) -> float | None:
    """Parse a provider numeric field."""

    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _integer(value: object) -> int | None:
    """Parse a provider port mapping."""

    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None
