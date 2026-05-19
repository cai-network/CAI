# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Self

import psutil

from cai.shared.types.memory import Memory
from cai.shared.types.thunderbolt import ThunderboltIdentifier
from cai.utils.pydantic_ext import CamelCaseModel


class MemoryUsage(CamelCaseModel):
    ram_total: Memory
    ram_available: Memory
    swap_total: Memory
    swap_available: Memory

    @classmethod
    def from_bytes(
        cls, *, ram_total: int, ram_available: int, swap_total: int, swap_available: int
    ) -> Self:
        return cls(
            ram_total=Memory.from_bytes(ram_total),
            ram_available=Memory.from_bytes(ram_available),
            swap_total=Memory.from_bytes(swap_total),
            swap_available=Memory.from_bytes(swap_available),
        )

    @classmethod
    def from_psutil(cls, *, override_memory: int | None) -> Self:
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()

        return cls.from_bytes(
            ram_total=vm.total,
            ram_available=vm.available if override_memory is None else override_memory,
            swap_total=sm.total,
            swap_available=sm.free,
        )


class DiskUsage(CamelCaseModel):
    """Disk space usage for the models directory."""

    total: Memory
    available: Memory

    @classmethod
    def from_path(cls, path: Path) -> Self:
        """Get disk usage stats for the nearest existing parent of *path*."""
        resolved_path = path.expanduser()
        target_path = next(
            (
                candidate
                for candidate in (resolved_path, *resolved_path.parents)
                if candidate.exists()
            ),
            None,
        )
        if target_path is None:
            raise FileNotFoundError(path)

        total, _used, free = shutil.disk_usage(target_path)
        return cls(
            total=Memory.from_bytes(total),
            available=Memory.from_bytes(free),
        )


class SystemPerformanceProfile(CamelCaseModel):
    # TODO: flops_fp16: float

    gpu_usage: float = 0.0
    temp: float = 0.0
    sys_power: float = 0.0
    pcpu_usage: float = 0.0
    ecpu_usage: float = 0.0

    @classmethod
    def from_psutil(cls) -> Self:
        """Best-effort cross-platform system load snapshot.

        We keep units aligned with the existing dashboard path:
        CPU/GPU usages are normalized to the ``0..1`` range and then the UI
        multiplies them back to percentages for rendering.
        """

        cpu_usage = psutil.cpu_percent(interval=None) / 100.0
        avg_temp = 0.0
        try:
            sensors = psutil.sensors_temperatures()
            readings = [
                float(entry.current)
                for entries in sensors.values()
                for entry in entries
                if getattr(entry, "current", None) is not None
            ]
            if readings:
                avg_temp = sum(readings) / len(readings)
        except (AttributeError, NotImplementedError, OSError):
            avg_temp = 0.0

        return cls(
            gpu_usage=0.0,
            temp=avg_temp,
            sys_power=0.0,
            pcpu_usage=cpu_usage,
            ecpu_usage=0.0,
        )


InterfaceType = Literal["wifi", "ethernet", "maybe_ethernet", "thunderbolt", "unknown"]
TransportEndpointPurpose = Literal["api", "data"]
TransportEndpointRouteType = Literal["direct", "overlay", "relay"]
TransportEndpointSource = Literal["explicit", "auto", "interface_scan"]

_TRANSPORT_ROUTE_PRIORITY: dict[TransportEndpointRouteType, int] = {
    "direct": 0,
    "overlay": 1,
    "relay": 2,
}
_TRANSPORT_SOURCE_PRIORITY: dict[TransportEndpointSource, int] = {
    "explicit": 0,
    "auto": 1,
    "interface_scan": 2,
}


class NetworkInterfaceInfo(CamelCaseModel):
    name: str
    ip_address: str
    interface_type: InterfaceType = "unknown"


class AdvertisedTransportEndpoint(CamelCaseModel):
    """Advertised way to reach a node for a specific control/data purpose."""

    purpose: TransportEndpointPurpose
    route_type: TransportEndpointRouteType
    host: str
    port: int | None = None
    source: TransportEndpointSource | None = None
    interface_name: str | None = None


class NodeIdentity(CamelCaseModel):
    """Static and slow-changing node identification data."""

    model_id: str = "Unknown"
    chip_id: str = "Unknown"
    friendly_name: str = "Unknown"
    os_version: str = "Unknown"
    os_build_version: str = "Unknown"
    cpu_physical_cores: int | None = None
    cpu_logical_cores: int | None = None
    total_vram_bytes: int | None = None
    api_host: str | None = None
    api_port: int | None = None
    data_host: str | None = None
    data_port: int | None = None
    transport_endpoints: Sequence[AdvertisedTransportEndpoint] = []
    worker_enabled: bool | None = None
    relay_enabled: bool | None = None
    worker_reward_address: str | None = None
    node_public_key_b64: str | None = None
    node_public_key_address: str | None = None
    readiness: dict[str, Any] = {}

    def transport_endpoints_for(
        self,
        *,
        purpose: TransportEndpointPurpose | None = None,
        route_types: Sequence[TransportEndpointRouteType] | None = None,
        require_port: bool = False,
    ) -> list[AdvertisedTransportEndpoint]:
        filtered: list[AdvertisedTransportEndpoint] = []
        allowed_routes = set(route_types or [])
        for endpoint in self.transport_endpoints:
            if not endpoint.host:
                continue
            if purpose is not None and endpoint.purpose != purpose:
                continue
            if route_types is not None and endpoint.route_type not in allowed_routes:
                continue
            if require_port and endpoint.port is None:
                continue
            filtered.append(endpoint)
        filtered.sort(
            key=lambda endpoint: (
                _TRANSPORT_ROUTE_PRIORITY.get(endpoint.route_type, 99),
                _TRANSPORT_SOURCE_PRIORITY.get(endpoint.source or "interface_scan", 99),
                endpoint.host,
                -1 if endpoint.port is None else endpoint.port,
            )
        )
        return filtered

    def preferred_transport_endpoint(
        self,
        *,
        purpose: TransportEndpointPurpose,
        route_types: Sequence[TransportEndpointRouteType] | None = None,
        require_port: bool = False,
    ) -> AdvertisedTransportEndpoint | None:
        endpoints = self.transport_endpoints_for(
            purpose=purpose,
            route_types=route_types,
            require_port=require_port,
        )
        if not endpoints:
            return None
        return endpoints[0]


class NodeNetworkInfo(CamelCaseModel):
    """Network interface information for a node."""

    interfaces: Sequence[NetworkInterfaceInfo] = []


class NodeThunderboltInfo(CamelCaseModel):
    """Thunderbolt interface identifiers for a node."""

    interfaces: Sequence[ThunderboltIdentifier] = []


class NodeRdmaCtlStatus(CamelCaseModel):
    """Whether RDMA is enabled on this node (via rdma_ctl)."""

    enabled: bool


class ThunderboltBridgeStatus(CamelCaseModel):
    """Whether the Thunderbolt Bridge network service is enabled on this node."""

    enabled: bool
    exists: bool
    service_name: str | None = None

