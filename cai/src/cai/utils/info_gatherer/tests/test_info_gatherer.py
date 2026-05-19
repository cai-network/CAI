# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import anyio
import pytest

from cai.utils.info_gatherer.info_gatherer import (
    ApiEndpointInfo,
    InfoGatherer,
    StaticNodeInformation,
    WorkerStateInfo,
)
from cai.shared.types.profiling import NetworkInterfaceInfo


class _RecordingSender:
    def __init__(self) -> None:
        self.items: list[object] = []

    async def send(self, item: object) -> None:
        self.items.append(item)


@pytest.mark.anyio
async def test_monitor_static_info_repeats(monkeypatch: pytest.MonkeyPatch) -> None:
    sender = _RecordingSender()
    gatherer = InfoGatherer(info_sender=sender, api_port=52425, node_id="node-local")

    async def fake_gather() -> StaticNodeInformation:
        return StaticNodeInformation(
            model="Portable Node",
            chip="Ryzen",
            os_version="Windows",
            os_build_version="11",
            cpu_physical_cores=8,
            cpu_logical_cores=16,
            total_vram_bytes=24 * 1024**3,
        )

    monkeypatch.setattr(StaticNodeInformation, "gather", fake_gather)

    async with anyio.create_task_group() as tg:
        tg.start_soon(gatherer._monitor_static_info, 0.01)
        with anyio.move_on_after(0.1):
            while len(sender.items) < 2:
                await anyio.sleep(0.005)
        tg.cancel_scope.cancel()

    assert len(sender.items) >= 2
    assert sender.items[0].cpu_physical_cores == 8


@pytest.mark.anyio
async def test_monitor_worker_state_repeats(monkeypatch: pytest.MonkeyPatch) -> None:
    sender = _RecordingSender()
    gatherer = InfoGatherer(info_sender=sender, api_port=52425, node_id="node-local")

    async def fake_gather(*, node_id: str) -> WorkerStateInfo:
        assert node_id == "node-local"
        return WorkerStateInfo(
            worker_enabled=True,
            relay_enabled=True,
            worker_reward_address="abcd1234abcd1234abcd1234abcd1234",
        )

    monkeypatch.setattr(WorkerStateInfo, "gather", fake_gather)

    async with anyio.create_task_group() as tg:
        tg.start_soon(gatherer._monitor_worker_state, 0.01)
        with anyio.move_on_after(0.1):
            while len(sender.items) < 2:
                await anyio.sleep(0.005)
        tg.cancel_scope.cancel()

    assert len(sender.items) >= 2
    assert sender.items[0].worker_enabled is True
    assert sender.items[0].relay_enabled is True


@pytest.mark.anyio
async def test_api_endpoint_info_does_not_auto_advertise_overlay_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CAI_PUBLIC_API_HOST", raising=False)
    monkeypatch.delenv("CAI_PUBLIC_DATA_HOST", raising=False)
    monkeypatch.delenv("CAI_AUTO_ADVERTISE_API_HOST", raising=False)
    monkeypatch.delenv("CAI_AUTO_ADVERTISE_OVERLAY_API_HOST", raising=False)

    async def fake_get_network_interfaces() -> list[NetworkInterfaceInfo]:
        return [
            NetworkInterfaceInfo(
                name="Wi-Fi",
                ip_address="192.168.0.120",
                interface_type="wifi",
            ),
            NetworkInterfaceInfo(
                name="Radmin VPN",
                ip_address="26.97.29.153",
                interface_type="unknown",
            ),
        ]

    monkeypatch.setattr(
        "cai.utils.info_gatherer.info_gatherer.get_network_interfaces",
        fake_get_network_interfaces,
    )

    info = await ApiEndpointInfo.gather(api_port=52425)

    assert info.host is None
    assert info.data_host is None
    assert info.port == 52425
    assert info.data_port == 52425
    assert len(info.transport_endpoints) >= 1
    assert info.transport_endpoints[0].route_type == "overlay"
    assert info.transport_endpoints[0].purpose == "api"
    assert info.transport_endpoints[0].host == "26.97.29.153"


@pytest.mark.anyio
async def test_api_endpoint_info_defaults_data_port_to_api_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAI_PUBLIC_API_HOST", "85.137.164.250")
    monkeypatch.delenv("CAI_PUBLIC_DATA_HOST", raising=False)
    monkeypatch.delenv("CAI_PUBLIC_DATA_PORT", raising=False)

    async def fake_get_network_interfaces() -> list[NetworkInterfaceInfo]:
        return []

    monkeypatch.setattr(
        "cai.utils.info_gatherer.info_gatherer.get_network_interfaces",
        fake_get_network_interfaces,
    )

    info = await ApiEndpointInfo.gather(api_port=52425)

    assert info.host == "85.137.164.250"
    assert info.data_host == "85.137.164.250"
    assert info.data_port == 52425


@pytest.mark.anyio
@pytest.mark.parametrize("interface_name", ["sing-box", "v2rayN Tunnel", "Wintun"])
async def test_api_endpoint_info_does_not_promote_proxy_tunnel_to_api_host(
    monkeypatch: pytest.MonkeyPatch,
    interface_name: str,
) -> None:
    monkeypatch.delenv("CAI_PUBLIC_API_HOST", raising=False)
    monkeypatch.delenv("CAI_PUBLIC_DATA_HOST", raising=False)
    monkeypatch.delenv("CAI_AUTO_ADVERTISE_API_HOST", raising=False)
    monkeypatch.delenv("CAI_AUTO_ADVERTISE_OVERLAY_API_HOST", raising=False)

    async def fake_get_network_interfaces() -> list[NetworkInterfaceInfo]:
        return [
            NetworkInterfaceInfo(
                name="Ethernet",
                ip_address="192.168.1.45",
                interface_type="ethernet",
            ),
            NetworkInterfaceInfo(
                name=interface_name,
                ip_address="172.19.0.2",
                interface_type="unknown",
            ),
        ]

    monkeypatch.setattr(
        "cai.utils.info_gatherer.info_gatherer.get_network_interfaces",
        fake_get_network_interfaces,
    )

    info = await ApiEndpointInfo.gather(api_port=52425)

    assert info.host is None
    assert info.transport_endpoints[0].route_type == "overlay"


@pytest.mark.anyio
async def test_api_endpoint_info_can_explicitly_auto_advertise_overlay_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CAI_PUBLIC_API_HOST", raising=False)
    monkeypatch.delenv("CAI_PUBLIC_DATA_HOST", raising=False)
    monkeypatch.delenv("CAI_AUTO_ADVERTISE_API_HOST", raising=False)
    monkeypatch.setenv("CAI_AUTO_ADVERTISE_OVERLAY_API_HOST", "1")

    async def fake_get_network_interfaces() -> list[NetworkInterfaceInfo]:
        return [
            NetworkInterfaceInfo(
                name="Wi-Fi",
                ip_address="192.168.0.120",
                interface_type="wifi",
            ),
            NetworkInterfaceInfo(
                name="Radmin VPN",
                ip_address="26.97.29.153",
                interface_type="unknown",
            ),
        ]

    monkeypatch.setattr(
        "cai.utils.info_gatherer.info_gatherer.get_network_interfaces",
        fake_get_network_interfaces,
    )

    info = await ApiEndpointInfo.gather(api_port=52425)

    assert info.host == "26.97.29.153"
    assert info.data_host == "26.97.29.153"
    assert info.transport_endpoints[0].route_type == "overlay"


@pytest.mark.anyio
async def test_api_endpoint_info_respects_disabled_auto_advertise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CAI_PUBLIC_API_HOST", raising=False)
    monkeypatch.setenv("CAI_AUTO_ADVERTISE_API_HOST", "0")

    async def fake_get_network_interfaces() -> list[NetworkInterfaceInfo]:
        raise AssertionError("network interfaces should not be queried")

    monkeypatch.setattr(
        "cai.utils.info_gatherer.info_gatherer.get_network_interfaces",
        fake_get_network_interfaces,
    )

    info = await ApiEndpointInfo.gather(api_port=52425)

    assert info.host is None
    assert info.transport_endpoints == []

