# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cai.api.main import API
from cai.shared.types.common import NodeId
from cai.shared.types.profiling import NodeIdentity
from cai.shared.types.state import State


def _make_api(summary_local_only: bool) -> API:
    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api.port = 52415
    api.node_id = NodeId("node-local")
    api.state = {}
    api.summary_local_only = summary_local_only
    api.state_local_only = summary_local_only
    app.get("/v1/cai/summary")(api.get_cai_summary)
    app.get("/v1/cai/distributed-inference/diagnostics")(
        api.get_cai_distributed_inference_diagnostics
    )
    app.get("/state")(api.get_state)
    app.get("/state/{path:path}")(api.get_state)
    return api


def test_cai_summary_allows_public_requests_when_not_restricted() -> None:
    api = _make_api(summary_local_only=False)
    client = TestClient(api.app, client=("198.51.100.20", 40000))

    with patch("cai.api.main.load_cai_summary", return_value={"available": True}) as summary_mock:
        response = client.get("/v1/cai/summary")

    assert response.status_code == 200
    assert response.json() == {"available": True}
    summary_mock.assert_called_once()


def test_cai_summary_blocks_public_requests_when_local_only() -> None:
    api = _make_api(summary_local_only=True)
    client = TestClient(api.app, client=("198.51.100.20", 40000))

    with patch("cai.api.main.load_cai_summary") as summary_mock:
        response = client.get("/v1/cai/summary")

    assert response.status_code == 404
    summary_mock.assert_not_called()


def test_cai_summary_allows_local_requests_when_local_only() -> None:
    api = _make_api(summary_local_only=True)
    client = TestClient(api.app, client=("127.0.0.1", 40000))

    with patch("cai.api.main.load_cai_summary", return_value={"available": True}) as summary_mock:
        response = client.get("/v1/cai/summary")

    assert response.status_code == 200
    assert response.json() == {"available": True}
    summary_mock.assert_called_once()


def test_cai_distributed_inference_diagnostics_allows_local_requests() -> None:
    api = _make_api(summary_local_only=False)
    service = SimpleNamespace(
        distributed_inference_diagnostics=Mock(return_value={"status": "ready"})
    )
    api._get_cai_service = lambda: service
    client = TestClient(api.app, client=("127.0.0.1", 40000))

    response = client.get(
        "/v1/cai/distributed-inference/diagnostics?model_id=model-a"
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
    service.distributed_inference_diagnostics.assert_called_once_with(
        model_id="model-a"
    )


def test_cai_distributed_inference_diagnostics_blocks_public_requests() -> None:
    api = _make_api(summary_local_only=False)
    service = SimpleNamespace(distributed_inference_diagnostics=Mock())
    api._get_cai_service = lambda: service
    client = TestClient(api.app, client=("198.51.100.20", 40000))

    response = client.get("/v1/cai/distributed-inference/diagnostics")

    assert response.status_code == 404
    service.distributed_inference_diagnostics.assert_not_called()


def test_state_blocks_public_requests_when_local_only() -> None:
    api = _make_api(summary_local_only=True)
    client = TestClient(api.app, client=("198.51.100.20", 40000))

    response = client.get("/state")

    assert response.status_code == 404


def test_state_allows_local_requests_when_local_only() -> None:
    api = _make_api(summary_local_only=True)
    api.state = {"ok": True}
    client = TestClient(api.app, client=("127.0.0.1", 40000))

    response = client.get("/state")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_state_refreshes_local_worker_identity_from_node_config() -> None:
    api = _make_api(summary_local_only=False)
    api.node_id = NodeId("node-local")
    api.state = State(
        node_identities={
            NodeId("node-local"): NodeIdentity(
                worker_enabled=False,
                relay_enabled=False,
                worker_reward_address="old",
            )
        }
    )
    client = TestClient(api.app, client=("127.0.0.1", 40000))

    with (
        patch(
            "cai_compute_chain.node_config.load_or_create_node_config",
            return_value=type(
                "Config",
                (),
                {"worker_enabled": True, "relay_enabled": True},
            )(),
        ),
        patch(
            "cai_compute_chain.node_config.resolve_worker_reward_address",
            return_value="abc123",
        ),
        patch("cai_compute_chain.wallet.normalize_address", side_effect=lambda value: value),
    ):
        response = client.get("/state/nodeIdentities")

    assert response.status_code == 200
    assert response.json()["node-local"]["workerEnabled"] is True
    assert response.json()["node-local"]["relayEnabled"] is True
    assert response.json()["node-local"]["workerRewardAddress"] == "abc123"


def test_state_does_not_advertise_local_worker_without_reward_address() -> None:
    api = _make_api(summary_local_only=False)
    api.node_id = NodeId("node-local")
    api.state = State(
        node_identities={
            NodeId("node-local"): NodeIdentity(
                worker_enabled=True,
                relay_enabled=True,
            )
        }
    )
    client = TestClient(api.app, client=("127.0.0.1", 40000))

    with (
        patch(
            "cai_compute_chain.node_config.load_or_create_node_config",
            return_value=type(
                "Config",
                (),
                {"worker_enabled": True, "relay_enabled": True},
            )(),
        ),
        patch(
            "cai_compute_chain.node_config.resolve_worker_reward_address",
            return_value=None,
        ),
        patch("cai_compute_chain.wallet.get_active_wallet", return_value=None),
    ):
        response = client.get("/state/nodeIdentities")

    assert response.status_code == 200
    assert response.json()["node-local"]["workerEnabled"] is False
    assert response.json()["node-local"]["relayEnabled"] is True
    assert response.json()["node-local"]["workerRewardAddress"] is None
