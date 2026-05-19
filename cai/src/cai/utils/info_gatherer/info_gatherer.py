# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import ipaddress
import os
import shutil
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from subprocess import CalledProcessError
from typing import Any, Self, cast

import anyio
from anyio import fail_after, open_process, to_thread
from anyio.streams.buffered import BufferedByteReceiveStream
from loguru import logger
from pydantic import ValidationError

from cai.shared.constants import CAI_CONFIG_FILE, CAI_DEFAULT_MODELS_DIR
from cai.shared.types.memory import Memory
from cai.shared.types.profiling import (
    AdvertisedTransportEndpoint,
    DiskUsage,
    MemoryUsage,
    NetworkInterfaceInfo,
    SystemPerformanceProfile,
    ThunderboltBridgeStatus,
)
from cai.shared.types.thunderbolt import (
    ThunderboltConnection,
    ThunderboltConnectivity,
    ThunderboltIdentifier,
)
from cai.utils.channels import Sender
from cai.utils.pydantic_ext import TaggedModel
from cai.utils.task_group import TaskGroup

from .macmon import MacmonMetrics
from .system_info import (
    get_cpu_core_counts,
    get_friendly_name,
    get_model_and_chip,
    get_network_interfaces,
    get_os_build_version,
    get_os_version,
    get_total_vram_bytes,
)

IS_DARWIN = sys.platform == "darwin"

_AUTO_ADVERTISE_DISABLED_VALUES = {"0", "false", "no", "off", "disabled"}
_VPN_INTERFACE_KEYWORDS = (
    "clash",
    "radmin",
    "mihomo",
    "sing-box",
    "singbox",
    "tailscale",
    "tap",
    "zerotier",
    "v2ray",
    "v2rayn",
    "wireguard",
    "wintun",
    "openvpn",
    "hamachi",
    "nebula",
    "vpn",
    "tun",
)


async def _get_thunderbolt_devices() -> set[str] | None:
    """Get Thunderbolt interface device names (e.g., en2, en3) from hardware ports.

    Returns None if the networksetup command fails.
    """
    result = await anyio.run_process(
        ["networksetup", "-listallhardwareports"],
        check=False,
    )
    if result.returncode != 0:
        logger.warning(
            f"networksetup -listallhardwareports failed with code "
            f"{result.returncode}: {result.stderr.decode()}"
        )
        return None

    output = result.stdout.decode()
    thunderbolt_devices: set[str] = set()
    current_port: str | None = None

    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Hardware Port:"):
            current_port = line.split(":", 1)[1].strip()
        elif line.startswith("Device:") and current_port:
            device = line.split(":", 1)[1].strip()
            if "thunderbolt" in current_port.lower():
                thunderbolt_devices.add(device)
            current_port = None

    return thunderbolt_devices


async def _get_bridge_services() -> dict[str, str] | None:
    """Get mapping of bridge device -> service name from network service order.

    Returns None if the networksetup command fails.
    """
    result = await anyio.run_process(
        ["networksetup", "-listnetworkserviceorder"],
        check=False,
    )
    if result.returncode != 0:
        logger.warning(
            f"networksetup -listnetworkserviceorder failed with code "
            f"{result.returncode}: {result.stderr.decode()}"
        )
        return None

    # Parse service order to find bridge devices and their service names
    # Format: "(1) Service Name\n(Hardware Port: ..., Device: bridge0)\n"
    service_order_output = result.stdout.decode()
    bridge_services: dict[str, str] = {}  # device -> service name
    current_service: str | None = None

    for line in service_order_output.splitlines():
        line = line.strip()
        # Match "(N) Service Name" or "(*) Service Name" (disabled)
        # but NOT "(Hardware Port: ...)" lines
        if (
            line
            and line.startswith("(")
            and ")" in line
            and not line.startswith("(Hardware Port:")
        ):
            paren_end = line.index(")")
            if paren_end + 2 <= len(line):
                current_service = line[paren_end + 2 :]
        # Match "(Hardware Port: ..., Device: bridgeX)"
        elif current_service and "Device: bridge" in line:
            # Extract device name from "..., Device: bridge0)"
            device_start = line.find("Device: ") + len("Device: ")
            device_end = line.find(")", device_start)
            if device_end > device_start:
                device = line[device_start:device_end]
                bridge_services[device] = current_service

    return bridge_services


async def _get_bridge_members(bridge_device: str) -> set[str]:
    """Get member interfaces of a bridge device via ifconfig."""
    result = await anyio.run_process(
        ["ifconfig", bridge_device],
        check=False,
    )
    if result.returncode != 0:
        logger.debug(f"ifconfig {bridge_device} failed with code {result.returncode}")
        return set()

    members: set[str] = set()
    ifconfig_output = result.stdout.decode()
    for line in ifconfig_output.splitlines():
        line = line.strip()
        if line.startswith("member:"):
            parts = line.split()
            if len(parts) > 1:
                members.add(parts[1])

    return members


async def _find_thunderbolt_bridge(
    bridge_services: dict[str, str], thunderbolt_devices: set[str]
) -> str | None:
    """Find the service name of a bridge containing Thunderbolt interfaces.

    Returns the service name if found, None otherwise.
    """
    for bridge_device, service_name in bridge_services.items():
        members = await _get_bridge_members(bridge_device)
        if members & thunderbolt_devices:  # intersection is non-empty
            return service_name
    return None


async def _is_service_enabled(service_name: str) -> bool | None:
    """Check if a network service is enabled.

    Returns True if enabled, False if disabled, None on error.
    """
    result = await anyio.run_process(
        ["networksetup", "-getnetworkserviceenabled", service_name],
        check=False,
    )
    if result.returncode != 0:
        logger.warning(
            f"networksetup -getnetworkserviceenabled '{service_name}' "
            f"failed with code {result.returncode}: {result.stderr.decode()}"
        )
        return None

    stdout = result.stdout.decode().strip().lower()
    return stdout == "enabled"


class StaticNodeInformation(TaggedModel):
    """Node information that should NEVER change, to be gathered once at startup"""

    model: str
    chip: str
    os_version: str
    os_build_version: str
    cpu_physical_cores: int | None = None
    cpu_logical_cores: int | None = None
    total_vram_bytes: int | None = None

    @classmethod
    async def gather(cls) -> Self:
        model, chip = await get_model_and_chip()
        cpu_physical_cores, cpu_logical_cores = get_cpu_core_counts()
        return cls(
            model=model,
            chip=chip,
            os_version=get_os_version(),
            os_build_version=await get_os_build_version(),
            cpu_physical_cores=cpu_physical_cores,
            cpu_logical_cores=cpu_logical_cores,
            total_vram_bytes=await get_total_vram_bytes(),
        )


class NodeNetworkInterfaces(TaggedModel):
    ifaces: Sequence[NetworkInterfaceInfo]


class MacThunderboltIdentifiers(TaggedModel):
    idents: Sequence[ThunderboltIdentifier]


class MacThunderboltConnections(TaggedModel):
    conns: Sequence[ThunderboltConnection]


class RdmaCtlStatus(TaggedModel):
    enabled: bool

    @classmethod
    async def gather(cls) -> Self | None:
        if not IS_DARWIN or shutil.which("rdma_ctl") is None:
            return None
        try:
            with anyio.fail_after(5):
                proc = await anyio.run_process(["rdma_ctl", "status"], check=False)
        except (TimeoutError, OSError):
            return None
        if proc.returncode != 0:
            return None
        output = proc.stdout.decode("utf-8").lower().strip()
        if "enabled" in output:
            return cls(enabled=True)
        if "disabled" in output:
            return cls(enabled=False)
        return None


class ThunderboltBridgeInfo(TaggedModel):
    status: ThunderboltBridgeStatus

    @classmethod
    async def gather(cls) -> Self | None:
        """Check if a Thunderbolt Bridge network service is enabled on this node.

        Detection approach:
        1. Find all Thunderbolt interface devices (en2, en3, etc.) from hardware ports
        2. Find bridge devices from network service order (not hardware ports, as
           bridges may not appear there)
        3. Check each bridge's members via ifconfig
        4. If a bridge contains Thunderbolt interfaces, it's a Thunderbolt Bridge
        5. Check if that network service is enabled
        """
        if not IS_DARWIN:
            return None

        def _no_bridge_status() -> Self:
            return cls(
                status=ThunderboltBridgeStatus(
                    enabled=False, exists=False, service_name=None
                )
            )

        try:
            tb_devices = await _get_thunderbolt_devices()
            if tb_devices is None:
                return _no_bridge_status()

            bridge_services = await _get_bridge_services()
            if not bridge_services:
                return _no_bridge_status()

            tb_service_name = await _find_thunderbolt_bridge(
                bridge_services, tb_devices
            )
            if not tb_service_name:
                return _no_bridge_status()

            enabled = await _is_service_enabled(tb_service_name)
            if enabled is None:
                return cls(
                    status=ThunderboltBridgeStatus(
                        enabled=False, exists=True, service_name=tb_service_name
                    )
                )

            return cls(
                status=ThunderboltBridgeStatus(
                    enabled=enabled,
                    exists=True,
                    service_name=tb_service_name,
                )
            )
        except Exception as e:
            logger.opt(exception=e).warning("Failed to gather Thunderbolt Bridge info")
            return None


class NodeConfig(TaggedModel):
    """Node configuration from CAI_CONFIG_FILE, reloaded from the file only at startup. Other changes should come in through the API and propagate from there"""

    @classmethod
    async def gather(cls) -> Self | None:
        cfg_file = anyio.Path(CAI_CONFIG_FILE)
        await cfg_file.parent.mkdir(parents=True, exist_ok=True)
        await cfg_file.touch(exist_ok=True)
        async with await cfg_file.open("rb") as f:
            try:
                contents = (await f.read()).decode("utf-8")
                data = tomllib.loads(contents)
                return cls.model_validate(data)
            except (tomllib.TOMLDecodeError, UnicodeDecodeError, ValidationError):
                logger.warning("Invalid config file, skipping...")
                return None


class WorkerStateInfo(TaggedModel):
    """CAI worker and relay availability metadata replicated through node identity state."""

    worker_enabled: bool
    relay_enabled: bool = False
    worker_reward_address: str | None = None
    node_public_key_b64: str | None = None
    node_public_key_address: str | None = None
    readiness: dict[str, Any] = {}

    @classmethod
    async def gather(cls, *, node_id: str) -> Self:
        def _load() -> Self:
            try:
                from cai_compute_chain.model import WalletPolicy
                from cai_compute_chain.decentralized_compute import (
                    cai_owned_transport_runtime_readiness,
                )
                from cai_compute_chain.cai_owned_runtime import (
                    load_cai_owned_llm_shard_self_test_result,
                    load_cai_owned_transport_live_proof_result,
                )
                from cai_compute_chain.node_config import (
                    bind_worker_reward_address,
                    load_or_create_node_config,
                    resolve_worker_reward_address,
                )
                from cai_compute_chain.wallet import (
                    get_active_wallet,
                    load_unlocked_wallet_signing_material,
                    normalize_address,
                )
            except Exception:
                return cls(worker_enabled=False)

            policy = WalletPolicy()
            config = load_or_create_node_config(policy)
            worker_enabled = bool(getattr(config, "worker_enabled", False))
            relay_enabled = bool(getattr(config, "relay_enabled", False))
            reward_address = resolve_worker_reward_address(node_id, policy)
            runtime_enabled = _env_flag(
                "CAI_OWNED_TRANSPORT_RUNTIME_ENABLED",
                True,
            )
            runtime_operational = bool(worker_enabled and runtime_enabled)
            runtime_ready_raw = os.getenv("CAI_OWNED_TRANSPORT_RUNTIME_READY")
            runtime_ready_claim = (
                bool(runtime_operational)
                if runtime_ready_raw is None or not runtime_ready_raw.strip()
                else runtime_ready_raw.strip().lower()
                not in {"0", "false", "no", "off", "disabled"}
            )
            runtime_ready = bool(
                runtime_operational
                and runtime_ready_claim
            )
            llm_self_test = load_cai_owned_llm_shard_self_test_result(policy=policy)
            live_proof = load_cai_owned_transport_live_proof_result(policy=policy)

            if not reward_address:
                active_wallet = get_active_wallet(policy)
                if active_wallet is not None:
                    bind_worker_reward_address(node_id, active_wallet.address, policy=policy)
                    reward_address = active_wallet.address

            normalized_reward_address = None
            node_public_key_b64 = None
            node_public_key_address = None
            if reward_address:
                normalized_reward_address = normalize_address(reward_address)
            active_wallet = get_active_wallet(policy)
            if active_wallet is not None:
                signing_material = load_unlocked_wallet_signing_material(
                    active_wallet,
                    policy,
                )
                if isinstance(signing_material, dict):
                    node_public_key_b64 = str(
                        signing_material.get("public_key_b64") or ""
                    ).strip() or None
                    node_public_key_address = str(
                        signing_material.get("address") or ""
                    ).strip().lower() or None
            effective_worker_enabled = bool(worker_enabled and normalized_reward_address)

            return cls(
                worker_enabled=effective_worker_enabled,
                relay_enabled=relay_enabled,
                worker_reward_address=normalized_reward_address or None,
                node_public_key_b64=node_public_key_b64,
                node_public_key_address=node_public_key_address,
                readiness={
                    "caiOwnedTransport": cai_owned_transport_runtime_readiness(
                        runtime_ready=bool(runtime_ready and effective_worker_enabled),
                        implemented=True,
                        proof_kind="deterministic_bytes_shard_runtime_loop",
                        status=(
                            "ready"
                            if runtime_ready and effective_worker_enabled
                            else "test_adapter_ready"
                            if runtime_operational and effective_worker_enabled
                            else "disabled"
                        ),
                        llm_shard_self_test=llm_self_test,
                        runtime_ready_proof=live_proof,
                        require_runtime_ready_proof=_env_flag(
                            "CAI_OWNED_TRANSPORT_REQUIRE_LIVE_PROOF",
                            True,
                        ),
                    )
                },
            )

        return await to_thread.run_sync(_load)


class MiscData(TaggedModel):
    """Node information that may slowly change that doesn't fall into the other categories"""

    friendly_name: str

    @classmethod
    async def gather(cls) -> Self:
        return cls(friendly_name=await get_friendly_name())


class ApiEndpointInfo(TaggedModel):
    """Publicly advertised API endpoint for this node."""

    host: str | None
    port: int
    data_host: str | None = None
    data_port: int | None = None
    transport_endpoints: Sequence[AdvertisedTransportEndpoint] = []

    @classmethod
    async def gather(cls, *, api_port: int) -> Self:
        host = os.getenv("CAI_PUBLIC_API_HOST")
        port = int(os.getenv("CAI_PUBLIC_API_PORT", str(api_port)))
        data_host = os.getenv("CAI_PUBLIC_DATA_HOST")
        data_port_raw = os.getenv("CAI_PUBLIC_DATA_PORT")
        # The CAI-owned transport data-plane currently rides the same local
        # HTTP service unless a separate public data port is explicitly set.
        data_port = int(data_port_raw) if data_port_raw else port
        try:
            ifaces = await get_network_interfaces()
        except Exception:
            ifaces = []
        resolved_host = await resolve_advertised_host(host, ifaces=ifaces)
        resolved_data_host = (
            data_host.strip() if data_host and data_host.strip() else resolved_host
        )
        transport_endpoints = _collect_advertised_transport_endpoints(
            ifaces=ifaces,
            api_host=resolved_host,
            api_port=port,
            data_host=resolved_data_host,
            data_port=data_port,
            explicit_api_host=host,
            explicit_data_host=data_host,
        )
        return cls(
            host=resolved_host,
            port=port,
            data_host=resolved_data_host,
            data_port=data_port,
            transport_endpoints=transport_endpoints,
        )


async def resolve_advertised_host(
    explicit_host: str | None = None,
    *,
    ifaces: Sequence[NetworkInterfaceInfo] | None = None,
) -> str | None:
    resolved_host = explicit_host.strip() if explicit_host and explicit_host.strip() else None
    if resolved_host is None and _auto_advertise_api_host_enabled():
        try:
            if ifaces is None:
                ifaces = await get_network_interfaces()
            resolved_host = _select_auto_api_host(ifaces)
        except Exception as exc:
            logger.debug(f"Failed to auto-select API host: {exc!r}")
    return resolved_host


def _auto_advertise_api_host_enabled() -> bool:
    raw = os.getenv("CAI_AUTO_ADVERTISE_API_HOST")
    if raw is None:
        return True
    return raw.strip().lower() not in _AUTO_ADVERTISE_DISABLED_VALUES


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _AUTO_ADVERTISE_DISABLED_VALUES


def _select_auto_api_host(ifaces: Sequence[NetworkInterfaceInfo]) -> str | None:
    candidates: list[tuple[int, str]] = []
    allow_overlay_host = _auto_advertise_overlay_api_host_enabled()
    for iface in ifaces:
        address = _normalize_auto_api_address(iface.ip_address)
        if address is None:
            continue
        interface_name = str(iface.name or "").lower()
        if any(keyword in interface_name for keyword in _VPN_INTERFACE_KEYWORDS):
            if allow_overlay_host:
                candidates.append((20, address))
            continue
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            continue
        if ip.version == 4 and ip.is_global:
            candidates.append((10, address))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _auto_advertise_overlay_api_host_enabled() -> bool:
    raw = os.getenv("CAI_AUTO_ADVERTISE_OVERLAY_API_HOST")
    if raw is None:
        return False
    return raw.strip().lower() not in _AUTO_ADVERTISE_DISABLED_VALUES


def _normalize_auto_api_address(address: str) -> str | None:
    value = str(address or "").strip()
    if not value:
        return None
    if "%" in value:
        value = value.split("%", 1)[0]
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return None
    if (
        ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
        or ip.is_reserved
    ):
        return None
    if ip.version != 4:
        return None
    return value


def _is_overlay_interface_name(name: str) -> bool:
    interface_name = str(name or "").strip().lower()
    return any(keyword in interface_name for keyword in _VPN_INTERFACE_KEYWORDS)


def _normalize_transport_interface_address(
    address: str,
    *,
    require_global: bool,
) -> str | None:
    value = str(address or "").strip()
    if not value:
        return None
    if "%" in value:
        value = value.split("%", 1)[0]
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return None
    if (
        ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
        or ip.is_reserved
    ):
        return None
    if ip.version != 4:
        return None
    if require_global and not ip.is_global:
        return None
    return value


def _route_type_for_advertised_host(
    host: str,
    *,
    ifaces: Sequence[NetworkInterfaceInfo],
) -> str:
    normalized_host = _normalize_transport_interface_address(
        host,
        require_global=False,
    )
    if normalized_host is not None:
        for iface in ifaces:
            iface_ip = _normalize_transport_interface_address(
                iface.ip_address,
                require_global=False,
            )
            if iface_ip != normalized_host:
                continue
            if _is_overlay_interface_name(iface.name):
                return "overlay"
            return "direct"
        try:
            if ipaddress.ip_address(normalized_host).is_global:
                return "direct"
            return "overlay"
        except ValueError:
            pass
    return "direct"


def _collect_advertised_transport_endpoints(
    *,
    ifaces: Sequence[NetworkInterfaceInfo],
    api_host: str | None,
    api_port: int,
    data_host: str | None,
    data_port: int | None,
    explicit_api_host: str | None,
    explicit_data_host: str | None,
) -> list[AdvertisedTransportEndpoint]:
    endpoints_by_key: dict[tuple[str, str, str, int | None], AdvertisedTransportEndpoint] = {}

    def _register(
        *,
        purpose: str,
        route_type: str,
        host: str | None,
        port: int | None,
        source: str,
        interface_name: str | None = None,
    ) -> None:
        normalized_host = str(host or "").strip()
        if not normalized_host:
            return
        key = (purpose, route_type, normalized_host, port)
        candidate = AdvertisedTransportEndpoint(
            purpose=purpose,
            route_type=route_type,
            host=normalized_host,
            port=port,
            source=source,
            interface_name=interface_name,
        )
        existing = endpoints_by_key.get(key)
        if existing is None:
            endpoints_by_key[key] = candidate
            return
        existing_priority = 2 if existing.source is None else {"explicit": 0, "auto": 1, "interface_scan": 2}.get(existing.source, 2)
        candidate_priority = {"explicit": 0, "auto": 1, "interface_scan": 2}.get(source, 2)
        if candidate_priority < existing_priority:
            endpoints_by_key[key] = candidate

    for iface in ifaces:
        route_type: str | None = None
        ip = _normalize_transport_interface_address(
            iface.ip_address,
            require_global=False,
        )
        if ip is None:
            continue
        if _is_overlay_interface_name(iface.name):
            route_type = "overlay"
        else:
            try:
                route_type = "direct" if ipaddress.ip_address(ip).is_global else None
            except ValueError:
                route_type = None
        if route_type is None:
            continue
        _register(
            purpose="api",
            route_type=route_type,
            host=ip,
            port=api_port,
            source="interface_scan",
            interface_name=iface.name,
        )
        _register(
            purpose="data",
            route_type=route_type,
            host=ip,
            port=data_port,
            source="interface_scan",
            interface_name=iface.name,
        )

    api_source = "explicit" if explicit_api_host and explicit_api_host.strip() else "auto"
    _register(
        purpose="api",
        route_type=_route_type_for_advertised_host(api_host or "", ifaces=ifaces),
        host=api_host,
        port=api_port,
        source=api_source,
    )

    if data_host:
        data_source = "explicit" if explicit_data_host and explicit_data_host.strip() else api_source
        _register(
            purpose="data",
            route_type=_route_type_for_advertised_host(data_host, ifaces=ifaces),
            host=data_host,
            port=data_port,
            source=data_source,
        )

    return sorted(
        endpoints_by_key.values(),
        key=lambda endpoint: (
            {"direct": 0, "overlay": 1, "relay": 2}.get(endpoint.route_type, 99),
            {"explicit": 0, "auto": 1, "interface_scan": 2}.get(
                endpoint.source or "interface_scan",
                99,
            ),
            endpoint.purpose,
            endpoint.host,
            -1 if endpoint.port is None else endpoint.port,
        ),
    )


class NodeDiskUsage(TaggedModel):
    """Disk space information for the models directory."""

    disk_usage: DiskUsage

    @classmethod
    async def gather(cls) -> Self:
        return cls(
            disk_usage=await to_thread.run_sync(
                DiskUsage.from_path, CAI_DEFAULT_MODELS_DIR
            )
        )


class PsutilSystemMetrics(TaggedModel):
    """Best-effort system load metrics for Linux/WSL and other non-mac nodes."""

    system_profile: SystemPerformanceProfile

    @classmethod
    async def gather(cls) -> Self:
        return cls(
            system_profile=await to_thread.run_sync(
                SystemPerformanceProfile.from_psutil
            )
        )


async def _gather_iface_map() -> dict[str, str] | None:
    proc = await anyio.run_process(
        ["networksetup", "-listallhardwareports"], check=False
    )
    if proc.returncode != 0:
        return None

    ports: dict[str, str] = {}
    port = ""
    for line in proc.stdout.decode("utf-8").split("\n"):
        if line.startswith("Hardware Port:"):
            port = line.split(": ")[1]
        elif line.startswith("Device:"):
            ports[port] = line.split(": ")[1]
            port = ""
    if "" in ports:
        del ports[""]
    return ports


GatheredInfo = (
    MacmonMetrics
    | PsutilSystemMetrics
    | MemoryUsage
    | NodeNetworkInterfaces
    | MacThunderboltIdentifiers
    | MacThunderboltConnections
    | RdmaCtlStatus
    | ThunderboltBridgeInfo
    | NodeConfig
    | WorkerStateInfo
    | MiscData
    | ApiEndpointInfo
    | StaticNodeInformation
    | NodeDiskUsage
)


@dataclass
class InfoGatherer:
    info_sender: Sender[GatheredInfo]
    api_port: int
    node_id: str
    _tg: TaskGroup = field(init=False, default_factory=TaskGroup)
    _psutil_enabled: bool = field(init=False, default=False)

    async def _can_read_macmon_metrics(self, macmon_path: str) -> bool:
        try:
            with fail_after(5):
                proc = await anyio.run_process(
                    [macmon_path, "pipe", "--samples", "1", "--interval", "100"],
                    check=False,
                )
        except Exception as e:
            logger.opt(exception=e).warning(
                f"Failed to validate macmon at {macmon_path}"
            )
            return False

        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            logger.warning(
                f"macmon preflight failed with return code {proc.returncode}: "
                f"{stderr or 'no stderr'}"
            )
            return False

        stdout = proc.stdout.decode("utf-8", errors="replace").strip()
        if not stdout:
            logger.warning("macmon preflight returned no metrics")
            return False

        try:
            MacmonMetrics.from_raw_json(stdout.splitlines()[0])
        except ValidationError as e:
            logger.opt(exception=e).warning(
                "macmon preflight returned unexpected metrics JSON"
            )
            return False

        return True

    async def run(self):
        async with self._tg as tg:
            if IS_DARWIN:
                tg.start_soon(self._monitor_macmon, 1)
                tg.start_soon(self._monitor_system_profiler_thunderbolt_data, 5)
                tg.start_soon(self._monitor_thunderbolt_bridge_status, 10)
                tg.start_soon(self._monitor_rdma_ctl_status, 10)
            if not IS_DARWIN:
                tg.start_soon(self._monitor_memory_usage, 1)
                tg.start_soon(self._monitor_system_profile, 2)
            tg.start_soon(self._watch_system_info, 10)
            tg.start_soon(self._monitor_misc, 60)
            tg.start_soon(self._monitor_api_endpoint, 60)
            tg.start_soon(self._monitor_static_info, 60)
            tg.start_soon(self._monitor_worker_state, 15)
            tg.start_soon(self._monitor_disk_usage, 30)

            nc = await NodeConfig.gather()
            if nc is not None:
                await self.info_sender.send(nc)

    def shutdown(self):
        self._tg.cancel_tasks()

    async def _monitor_static_info(self, static_info_poll_interval: float):
        while True:
            try:
                with fail_after(30):
                    await self.info_sender.send(await StaticNodeInformation.gather())
            except Exception as e:
                logger.opt(exception=e).warning("Error gathering static node info")
            await anyio.sleep(static_info_poll_interval)

    async def _monitor_misc(self, misc_poll_interval: float):
        while True:
            try:
                with fail_after(10):
                    await self.info_sender.send(await MiscData.gather())
            except Exception as e:
                logger.opt(exception=e).warning("Error gathering misc data")
            await anyio.sleep(misc_poll_interval)

    async def _monitor_worker_state(self, worker_state_poll_interval: float):
        while True:
            try:
                with fail_after(10):
                    await self.info_sender.send(
                        await WorkerStateInfo.gather(node_id=self.node_id)
                    )
            except Exception as e:
                logger.opt(exception=e).warning("Error gathering worker state")
            await anyio.sleep(worker_state_poll_interval)

    async def _monitor_api_endpoint(self, api_endpoint_poll_interval: float):
        while True:
            try:
                with fail_after(10):
                    await self.info_sender.send(
                        await ApiEndpointInfo.gather(api_port=self.api_port)
                    )
            except Exception as e:
                logger.opt(exception=e).warning("Error gathering API endpoint info")
            await anyio.sleep(api_endpoint_poll_interval)

    async def _monitor_system_profiler_thunderbolt_data(
        self, system_profiler_interval: float
    ):
        while True:
            try:
                with fail_after(30):
                    iface_map = await _gather_iface_map()
                    if iface_map is None:
                        raise ValueError("Failed to gather interface map")

                    data = await ThunderboltConnectivity.gather()
                    assert data is not None

                    idents = [
                        it for i in data if (it := i.ident(iface_map)) is not None
                    ]
                    await self.info_sender.send(
                        MacThunderboltIdentifiers(idents=idents)
                    )

                    conns = [it for i in data if (it := i.conn()) is not None]
                    await self.info_sender.send(MacThunderboltConnections(conns=conns))
            except Exception as e:
                logger.opt(exception=e).warning("Error gathering Thunderbolt data")
            await anyio.sleep(system_profiler_interval)

    async def _monitor_memory_usage(self, memory_poll_rate: float):
        if self._psutil_enabled:
            return
        self._psutil_enabled = True
        override_memory_env = os.getenv("OVERRIDE_MEMORY_MB")
        override_memory: int | None = (
            Memory.from_mb(int(override_memory_env)).in_bytes
            if override_memory_env
            else None
        )
        while True:
            try:
                await self.info_sender.send(
                    MemoryUsage.from_psutil(override_memory=override_memory)
                )
            except Exception as e:
                logger.opt(exception=e).warning("Error gathering memory usage")
            await anyio.sleep(memory_poll_rate)

    async def _watch_system_info(self, interface_watcher_interval: float):
        while True:
            try:
                with fail_after(10):
                    nics = await get_network_interfaces()
                    await self.info_sender.send(NodeNetworkInterfaces(ifaces=nics))
            except Exception as e:
                logger.opt(exception=e).warning("Error gathering network interfaces")
            await anyio.sleep(interface_watcher_interval)

    async def _monitor_system_profile(self, system_poll_interval: float):
        while True:
            try:
                with fail_after(5):
                    await self.info_sender.send(await PsutilSystemMetrics.gather())
            except Exception as e:
                logger.opt(exception=e).warning("Error gathering system load")
            await anyio.sleep(system_poll_interval)

    async def _monitor_thunderbolt_bridge_status(
        self, thunderbolt_bridge_poll_interval: float
    ):
        while True:
            try:
                with fail_after(30):
                    curr = await ThunderboltBridgeInfo.gather()
                    if curr is not None:
                        await self.info_sender.send(curr)
            except Exception as e:
                logger.opt(exception=e).warning(
                    "Error gathering Thunderbolt Bridge status"
                )
            await anyio.sleep(thunderbolt_bridge_poll_interval)

    async def _monitor_rdma_ctl_status(self, rdma_ctl_poll_interval: float):
        while True:
            try:
                curr = await RdmaCtlStatus.gather()
                if curr is not None:
                    await self.info_sender.send(curr)
            except Exception as e:
                logger.opt(exception=e).warning("Error gathering RDMA ctl status")
            await anyio.sleep(rdma_ctl_poll_interval)

    async def _monitor_disk_usage(self, disk_poll_interval: float):
        while True:
            try:
                with fail_after(5):
                    await self.info_sender.send(await NodeDiskUsage.gather())
            except Exception as e:
                logger.opt(exception=e).warning("Error gathering disk usage")
            await anyio.sleep(disk_poll_interval)

    async def _monitor_macmon(self, macmon_interval: float):
        if (
            macmon_path := os.getenv("CAI_MACMON_PATH") or shutil.which("macmon")
        ) is None:
            logger.warning(
                "macmon not found, falling back to psutil for memory monitoring"
            )
            self._tg.start_soon(self._monitor_memory_usage, 1)
            return
        if not await self._can_read_macmon_metrics(macmon_path):
            logger.warning(
                f"macmon at {macmon_path} is unusable, falling back to psutil memory monitoring"
            )
            self._tg.start_soon(self._monitor_memory_usage, 1)
            return
        # macmon pipe --interval [interval in ms]
        # Timeout: if macmon produces no output for this many seconds, restart it.
        # macmon writes every macmon_interval seconds, so 10x that is generous.
        read_timeout = max(macmon_interval * 10, 30)
        while True:
            try:
                async with await open_process(
                    [
                        macmon_path,
                        "pipe",
                        "--interval",
                        str(macmon_interval * 1000),
                    ]
                ) as p:
                    if not p.stdout:
                        logger.critical("MacMon closed stdout")
                        return
                    stream = BufferedByteReceiveStream(p.stdout)
                    while True:
                        with fail_after(read_timeout):
                            data = await stream.receive_until(
                                delimiter=b"\n", max_bytes=8 * 1024
                            )
                            text = data.decode("utf-8", errors="replace").strip()
                            metrics = MacmonMetrics.from_raw_json(text)
                        await self.info_sender.send(metrics)
            except TimeoutError:
                logger.warning(
                    f"MacMon produced no output for {read_timeout}s, restarting"
                )
                self._tg.start_soon(self._monitor_memory_usage, 1)
            except CalledProcessError as e:
                stderr_msg = "no stderr"
                stderr_output = cast(bytes | str | None, e.stderr)
                if stderr_output is not None:
                    stderr_msg = (
                        stderr_output.decode()
                        if isinstance(stderr_output, bytes)
                        else str(stderr_output)
                    )
                logger.warning(
                    f"MacMon failed with return code {e.returncode}: {stderr_msg}"
                )
                self._tg.start_soon(self._monitor_memory_usage, 1)
            except Exception as e:
                logger.opt(exception=e).warning("Error in macmon monitor")
                self._tg.start_soon(self._monitor_memory_usage, 1)
            await anyio.sleep(macmon_interval)

