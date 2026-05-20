# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

from cai_compute_chain import peer_payload as peer_payload_module
from cai_compute_chain.decentralized_compute import (
    CAI_OWNED_TRANSPORT_PROTOCOL,
    plan_llama_cpp_distributed_execution,
)
from cai_compute_chain.model import CaiNetworkConfig, MoneyPolicy, WalletPolicy
from cai_compute_chain.peer_payload import (
    validate_peer_payload_network,
    verify_peer_payload_signature,
)
from cai_compute_chain.route_health import llama_cpp_compute_cell_profile_for_path
from cai_compute_chain.wallet_signing import (
    encode_bytes,
    generate_signing_seed,
    public_key_b64_from_seed,
)
from cai.api.cai_bridge import (
    CaiBridgeService,
    _build_local_model_shard_inventory,
    _job_to_history_summary,
    _job_to_summary,
    _receipt_to_summary,
    _settlement_to_summary,
)


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_build_local_model_shard_inventory_from_completed_download() -> None:
    inventory = _build_local_model_shard_inventory(
        {
            "downloads": {
                "node-a": [
                    {
                        "DownloadCompleted": {
                            "nodeId": "node-a",
                            "shardMetadata": {
                                "PipelineShardMetadata": {
                                    "modelCard": {
                                        "modelId": "cai-network/Qwen3-0.6B-GGUF"
                                    },
                                    "deviceRank": 0,
                                    "worldSize": 2,
                                    "startLayer": 0,
                                    "endLayer": 14,
                                    "nLayers": 28,
                                }
                            },
                            "total": {"bytes": 1},
                        }
                    }
                ],
                "node-b": [
                    {
                        "DownloadCompleted": {
                            "nodeId": "node-b",
                            "shardMetadata": {
                                "PipelineShardMetadata": {
                                    "modelCard": {
                                        "modelId": "cai-network/Qwen3-0.6B-GGUF"
                                    },
                                    "deviceRank": 1,
                                    "worldSize": 2,
                                    "startLayer": 14,
                                    "endLayer": 28,
                                    "nLayers": 28,
                                }
                            },
                            "total": {"bytes": 1},
                        }
                    }
                ],
            }
        },
        "node-a",
    )

    model_inventory = inventory["cai-network/Qwen3-0.6B-GGUF"]

    assert model_inventory["status"] == "partial_ready"
    assert model_inventory["shards"] == [
        {
            "layerStart": 0,
            "layerEnd": 14,
            "status": "downloaded",
            "ready": True,
            "downloaded": True,
            "cached": True,
            "source": "state.downloads",
            "shardType": "PipelineShardMetadata",
            "deviceRank": 0,
            "worldSize": 2,
            "nLayers": 28,
        }
    ]


def test_build_local_model_shard_inventory_from_verified_chunk_cache() -> None:
    manifest = SimpleNamespace(
        model_id="cai-network/Qwen3-0.6B-GGUF",
        catalog_id="qwen3-06b",
        version="v1",
        chunks=[
            SimpleNamespace(
                chunk_id="chunk-a",
                layer_start=0,
                layer_end=14,
                encrypted_at_rest=False,
            )
        ],
        compute_default_chunk_coverage=lambda present: SimpleNamespace(
            ready=True,
            missing_chunk_ids=(),
        ),
    )
    model_distribution = SimpleNamespace(
        list_model_package_manifests=lambda policy=None: [manifest],
        list_cached_chunks=lambda policy=None: [
            SimpleNamespace(
                catalog_id="qwen3-06b",
                version="v1",
                chunk_id="chunk-a",
            )
        ],
    )

    inventory = _build_local_model_shard_inventory(
        {},
        "node-a",
        model_distribution=model_distribution,
    )

    shard = inventory["cai-network/Qwen3-0.6B-GGUF"]["shards"][0]
    assert inventory["cai-network/Qwen3-0.6B-GGUF"]["source"] == "chunk-cache"
    assert shard["source"] == "chunk-cache"
    assert shard["ready"] is True
    assert shard["chunkManifestVerified"] is True
    assert shard["cacheVerified"] is True


def test_build_local_model_shard_inventory_blocks_encrypted_cache_without_key() -> None:
    manifest = SimpleNamespace(
        model_id="cai-network/Qwen3-0.6B-GGUF",
        catalog_id="qwen3-06b",
        version="v1",
        chunks=[
            SimpleNamespace(
                chunk_id="chunk-a",
                layer_start=0,
                layer_end=14,
                encrypted_at_rest=True,
            )
        ],
        compute_default_chunk_coverage=lambda present: SimpleNamespace(
            ready=True,
            missing_chunk_ids=(),
        ),
    )
    model_distribution = SimpleNamespace(
        list_model_package_manifests=lambda policy=None: [manifest],
        list_cached_chunks=lambda policy=None: [
            SimpleNamespace(
                catalog_id="qwen3-06b",
                version="v1",
                chunk_id="chunk-a",
            )
        ],
    )

    inventory = _build_local_model_shard_inventory(
        {},
        "node-a",
        model_distribution=model_distribution,
    )

    shard = inventory["cai-network/Qwen3-0.6B-GGUF"]["shards"][0]
    assert shard["ready"] is False
    assert shard["status"] == "cached_blocked"
    assert shard["encryptedAtRest"] is True
    assert shard["decryptionKeyAvailable"] is False


def test_worker_binding_auto_binds_current_node_to_active_wallet() -> None:
    bound: list[tuple[str, str, object]] = []

    service = object.__new__(CaiBridgeService)
    service.state_url = "http://127.0.0.1:52415/state"
    service.cai_url = "http://127.0.0.1:52415"
    service.wallet_policy = object()
    service.modules = SimpleNamespace(
        node_config=SimpleNamespace(
            load_or_create_node_config=lambda policy=None: SimpleNamespace(
                worker_enabled=True
            ),
            assess_validator_network_status=lambda **kwargs: SimpleNamespace(
                current_node_id="node-local"
            ),
            resolve_worker_reward_address=lambda node_id, policy=None: None,
            bind_worker_reward_address=lambda node_id, address, policy=None: bound.append(
                (node_id, address, policy)
            ),
        ),
        wallet=SimpleNamespace(
            get_active_wallet=lambda policy=None: SimpleNamespace(
                address="abcd1234abcd1234abcd1234abcd1234"
            ),
            normalize_address=lambda address: str(address).lower(),
        ),
    )

    with patch("cai.api.cai_bridge._load_state_payload", return_value={}):
        result = service._ensure_local_worker_reward_binding()  # pyright: ignore[reportPrivateUsage]

    assert result == "node-local"
    assert bound == [
        ("node-local", "abcd1234abcd1234abcd1234abcd1234", service.wallet_policy)
    ]


def test_worker_reward_address_is_available_before_worker_is_enabled() -> None:
    bindings: dict[str, str] = {}

    service = object.__new__(CaiBridgeService)
    service.state_url = "http://127.0.0.1:52415/state"
    service.cai_url = "http://127.0.0.1:52415"
    service.wallet_policy = object()
    service.modules = SimpleNamespace(
        node_config=SimpleNamespace(
            assess_validator_network_status=lambda **kwargs: SimpleNamespace(
                current_node_id="node-local"
            ),
            resolve_worker_reward_address=lambda node_id, policy=None: bindings.get(
                node_id
            ),
            bind_worker_reward_address=lambda node_id, address, policy=None: bindings.__setitem__(
                node_id, address
            ),
        ),
        wallet=SimpleNamespace(
            get_active_wallet=lambda policy=None: SimpleNamespace(
                address="abcd1234abcd1234abcd1234abcd1234"
            ),
            normalize_address=lambda address: str(address).lower(),
        ),
    )

    with patch("cai.api.cai_bridge._load_state_payload", return_value={}):
        address = service._local_worker_reward_address()  # pyright: ignore[reportPrivateUsage]

    assert address == "abcd1234abcd1234abcd1234abcd1234"
    assert bindings == {"node-local": "abcd1234abcd1234abcd1234abcd1234"}


def test_set_worker_enabled_binds_after_worker_mode_is_enabled() -> None:
    config = SimpleNamespace(
        worker_enabled=False,
        validator_state="unbonded",
        worker_allowed_model_ids=[],
        worker_max_parallel_jobs=1,
        worker_max_memory_mb=None,
    )
    binding_checks: list[bool] = []

    def set_worker_mode(**kwargs):
        config.worker_enabled = bool(kwargs["enabled"])
        return config

    service = object.__new__(CaiBridgeService)
    service.wallet_policy = object()
    service.modules = SimpleNamespace(
        node_config=SimpleNamespace(
            load_or_create_node_config=lambda policy=None: config,
            set_worker_mode=set_worker_mode,
        ),
        wallet=SimpleNamespace(
            get_active_wallet=lambda policy=None: SimpleNamespace(
                address="abcd1234abcd1234abcd1234abcd1234"
            )
        ),
        settlement=SimpleNamespace(reconcile_worker_payouts=lambda policy=None: []),
    )
    service._ensure_local_worker_reward_binding = lambda: binding_checks.append(  # pyright: ignore[method-assign]
        config.worker_enabled
    ) or "node-local"

    result = service.set_worker_enabled(enabled=True)

    assert result["config"]["worker_enabled"] is True
    assert binding_checks == [True]


def test_set_worker_enabled_rejects_missing_reward_destination() -> None:
    config = SimpleNamespace(
        worker_enabled=False,
        validator_state="unbonded",
        worker_allowed_model_ids=[],
        worker_max_parallel_jobs=1,
        worker_max_memory_mb=None,
    )
    service = object.__new__(CaiBridgeService)
    service.state_url = "http://127.0.0.1:52415/state"
    service.cai_url = "http://127.0.0.1:52415"
    service.wallet_policy = object()
    service.modules = SimpleNamespace(
        node_config=SimpleNamespace(
            load_or_create_node_config=lambda policy=None: config,
            assess_validator_network_status=lambda **kwargs: SimpleNamespace(
                current_node_id="node-local"
            ),
            resolve_worker_reward_address=lambda node_id, policy=None: None,
            set_worker_mode=lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("set_worker_mode should not be called")
            ),
        ),
        wallet=SimpleNamespace(get_active_wallet=lambda policy=None: None),
    )

    with patch("cai.api.cai_bridge._load_state_payload", return_value={}):
        try:
            service.set_worker_enabled(enabled=True)
        except ValueError as exc:
            assert (
                str(exc)
                == "Select an active wallet or bind a worker reward address before enabling worker mode."
            )
        else:
            raise AssertionError("Expected worker enable to reject missing reward destination")


def test_set_relay_enabled_updates_local_config() -> None:
    config = SimpleNamespace(relay_enabled=False)

    def set_relay_mode(enabled, policy=None):
        config.relay_enabled = bool(enabled)
        return config

    service = object.__new__(CaiBridgeService)
    service.wallet_policy = object()
    service.modules = SimpleNamespace(
        node_config=SimpleNamespace(set_relay_mode=set_relay_mode),
    )

    result = service.set_relay_enabled(enabled=True)

    assert result["message"] == "Relay mode enabled."
    assert result["config"]["relay_enabled"] is True


def test_chat_completion_uses_execution_cai_url_for_jobs() -> None:
    calls: list[tuple[str, str | None, str | None, str | None]] = []

    service = object.__new__(CaiBridgeService)
    service.state_url = "http://127.0.0.1:52415/state"
    service.cai_url = "http://127.0.0.1:52415"
    service.execution_cai_url = "http://85.137.164.250:52425"
    service.local_node_id = "node-local"
    service.wallet_policy = object()
    service.money_policy = object()
    service.network_model_policy = SimpleNamespace(
        network_default_execution_model_id="Qwen/Qwen3-0.6B-GGUF"
    )
    service.modules = SimpleNamespace(
        model=SimpleNamespace(PaymentPreference=SimpleNamespace(AUTO="auto")),
        jobs=SimpleNamespace(
            create_job_intent=lambda **kwargs: calls.append(
                (
                    kwargs["cai_url"],
                    kwargs.get("execution_cai_url"),
                    kwargs.get("requester_node_id"),
                    kwargs.get("reserve_client_ip"),
                )
            )
            or SimpleNamespace(job_id="job-1"),
            execute_job_intent=lambda *args, **kwargs: (
                SimpleNamespace(job_id="job-1"),
                SimpleNamespace(raw_response={"id": "resp-1"}),
            ),
        ),
    )
    service._ensure_local_worker_reward_binding = lambda: None  # pyright: ignore[method-assign]

    result = service.chat_completion(
        {
            "model": "Qwen/Qwen3-0.6B-GGUF",
            "messages": [{"role": "user", "content": "2+3=?"}],
        },
        reserve_client_ip="203.0.113.42",
    )

    assert calls == [
        (
            "http://127.0.0.1:52415",
            "http://85.137.164.250:52425",
            "node-local",
            "203.0.113.42",
        )
    ]
    assert result["response"]["id"] == "resp-1"


def test_chat_completion_defaults_to_private_network_model() -> None:
    calls: list[str] = []

    service = object.__new__(CaiBridgeService)
    service.state_url = "http://127.0.0.1:52415/state"
    service.cai_url = "http://127.0.0.1:52415"
    service.execution_cai_url = "http://127.0.0.1:52415"
    service.local_node_id = "node-local"
    service.wallet_policy = object()
    service.money_policy = object()
    service.network_model_policy = SimpleNamespace(
        network_default_model_id="cai-network/Qwen3-0.6B-GGUF",
        network_default_execution_model_id="Qwen/Qwen3-0.6B-GGUF",
    )
    service.modules = SimpleNamespace(
        model=SimpleNamespace(PaymentPreference=SimpleNamespace(AUTO="auto")),
        jobs=SimpleNamespace(
            create_job_intent=lambda **kwargs: calls.append(kwargs["model_id"])
            or SimpleNamespace(job_id="job-1"),
            execute_job_intent=lambda *args, **kwargs: (
                SimpleNamespace(job_id="job-1"),
                SimpleNamespace(raw_response={"id": "resp-1"}),
            ),
        ),
    )
    service._ensure_local_worker_reward_binding = lambda: None  # pyright: ignore[method-assign]

    service.chat_completion({"messages": [{"role": "user", "content": "2+3=?"}]})

    assert calls == ["cai-network/Qwen3-0.6B-GGUF"]


def test_chat_completion_normalizes_execution_alias_to_private_network_model() -> None:
    calls: list[tuple[str, str]] = []

    service = object.__new__(CaiBridgeService)
    service.state_url = "http://127.0.0.1:52415/state"
    service.cai_url = "http://127.0.0.1:52415"
    service.execution_cai_url = "http://127.0.0.1:52415"
    service.local_node_id = "node-local"
    service.wallet_policy = object()
    service.money_policy = object()
    service.network_model_policy = SimpleNamespace(
        network_default_model_id="cai-network/Qwen3-0.6B-GGUF",
        network_default_execution_model_id="Qwen/Qwen3-0.6B-GGUF",
    )
    service.modules = SimpleNamespace(
        model=SimpleNamespace(PaymentPreference=SimpleNamespace(AUTO="auto")),
        jobs=SimpleNamespace(
            create_job_intent=lambda **kwargs: calls.append(
                (
                    kwargs["model_id"],
                    kwargs["request_payload_preview"]["model"],
                )
            )
            or SimpleNamespace(job_id="job-1"),
            execute_job_intent=lambda *args, **kwargs: (
                SimpleNamespace(job_id="job-1"),
                SimpleNamespace(raw_response={"id": "resp-1"}),
            ),
        ),
    )
    service._ensure_local_worker_reward_binding = lambda: None  # pyright: ignore[method-assign]

    service.chat_completion(
        {
            "model": "Qwen/Qwen3-0.6B-GGUF",
            "messages": [{"role": "user", "content": "2+3=?"}],
        }
    )

    assert calls == [
        (
            "cai-network/Qwen3-0.6B-GGUF",
            "cai-network/Qwen3-0.6B-GGUF",
        )
    ]


def test_chat_completion_keeps_private_model_when_private_default_lacks_workers() -> None:
    calls: list[str] = []

    service = object.__new__(CaiBridgeService)
    service.state_url = "http://127.0.0.1:52415/state"
    service.cai_url = "http://127.0.0.1:52415"
    service.execution_cai_url = "http://127.0.0.1:52415"
    service.local_node_id = "node-local"
    service.wallet_policy = object()
    service.money_policy = object()
    service.network_model_policy = SimpleNamespace(
        network_default_model_id="cai-network/Qwen3-0.6B-GGUF",
        network_default_execution_model_id="Qwen/Qwen3-0.6B-GGUF",
        minimum_worker_shards=1,
    )
    service.modules = SimpleNamespace(
        model=SimpleNamespace(PaymentPreference=SimpleNamespace(AUTO="auto")),
        jobs=SimpleNamespace(
            _resolve_worker_execution_node_audit=lambda cai_url, model_id: {
                "modelId": model_id,
                "checkedNodeCount": 2,
                "eligibleNodeIds": ["node-remote"],
                "nodes": [
                    {
                        "nodeId": "node-remote",
                        "eligible": True,
                        "reasons": [],
                    },
                    {
                        "nodeId": "node-local",
                        "eligible": False,
                        "reasons": ["worker mode is disabled"],
                    },
                ],
            },
            create_job_intent=lambda **kwargs: calls.append(kwargs["model_id"])
            or SimpleNamespace(job_id="job-1"),
            execute_job_intent=lambda *args, **kwargs: (
                SimpleNamespace(job_id="job-1"),
                SimpleNamespace(raw_response={"id": "resp-1"}),
            ),
        ),
    )
    service._ensure_local_worker_reward_binding = lambda: None  # pyright: ignore[method-assign]

    service.chat_completion({"messages": [{"role": "user", "content": "2+3=?"}]})

    assert calls == ["cai-network/Qwen3-0.6B-GGUF"]


def test_job_summaries_redact_prompt_text() -> None:
    job = SimpleNamespace(
        job_id="job-1",
        created_at="2026-04-26T00:00:00Z",
        status="completed",
        model_id="Qwen/Qwen3-0.6B-GGUF",
        requester_node_id="node-local",
        prompt="2+2=?",
        pricing_mode="auto",
        receipt_id="receipt-1",
        settlement_id="settlement-1",
    )

    summary = _job_to_summary(job)
    history = _job_to_history_summary(job)

    assert summary is not None
    assert history is not None
    assert "prompt" not in summary
    assert "prompt" not in history
    assert summary["requesterNodeId"] == "node-local"
    assert history["requesterNodeId"] == "node-local"
    assert summary["promptRedacted"] is True
    assert history["promptRedacted"] is True


def test_job_summaries_expose_execution_attempt_status() -> None:
    job = SimpleNamespace(
        job_id="job-1",
        created_at="2026-04-26T00:00:00Z",
        status="running",
        model_id="Qwen/Qwen3-0.6B-GGUF",
        requester_node_id="node-local",
        prompt="2+2=?",
        pricing_mode="auto",
        pricing_basis="llm_tokens",
        receipt_id=None,
        settlement_id=None,
        execution_attempts=[
            {
                "attempt": 1,
                "status": "retrying",
                "message": "timed out",
                "participantNodeIds": ["node-a"],
                "excludedNodeIds": [],
                "retryScheduled": True,
                "phase": "retry_scheduled",
                "phaseStartedAt": "2026-04-26T00:00:01Z",
                "phaseMessage": "Execution attempt failed; retrying.",
                "attemptDurationMs": 1200,
                "readinessDurationMs": 300,
                "responseDurationMs": 900,
            }
        ],
    )

    summary = _job_to_summary(job)
    history = _job_to_history_summary(job)

    assert summary is not None
    assert history is not None
    assert summary["executionAttemptCount"] == 1
    assert summary["executionAttemptStatus"]["attempt"] == 1
    assert summary["executionAttemptStatus"]["status"] == "retrying"
    assert summary["executionAttemptStatus"]["message"] == "timed out"
    assert summary["executionAttemptStatus"]["participantNodeIds"] == ["node-a"]
    assert summary["executionAttemptStatus"]["phase"] == "retry_scheduled"
    assert summary["executionAttemptStatus"]["phaseMessage"] == "Execution attempt failed; retrying."
    assert summary["executionAttemptStatus"]["attemptDurationMs"] == 1200
    assert summary["executionAttemptStatus"]["readinessDurationMs"] == 300
    assert summary["executionAttemptStatus"]["responseDurationMs"] == 900
    assert history["executionAttemptStatus"] == summary["executionAttemptStatus"]


def test_receipt_summary_exposes_network_audit() -> None:
    receipt = SimpleNamespace(
        receipt_id="receipt-1",
        job_id="job-1",
        finish_reason="stop",
        output_text="hello",
        instance_id="instance-1",
        pricing_mode="network_auto",
        pricing_basis="llm_tokens",
        prompt_tokens=12,
        completion_tokens=8,
        total_tokens=20,
        reserved_prompt_tokens=18,
        reserved_completion_tokens=64,
        reserved_compute_cost_atomic=150000,
        actual_compute_cost_atomic=100000,
        reservation_surplus_atomic=50000,
        usage_priced=True,
        worker_payouts=[{"node_id": "node-a"}, {"node_id": "node-b"}],
        network_audit={
            "transportMode": "multi_worker_direct",
            "participantCount": 2,
            "decentralizedExecution": True,
            "llamaCppExecutionStrategy": {
                "executionMode": "llama_cpp_rpc_low_latency",
            },
            "caiOwnedTransportExecuted": False,
            "caiOwnedTransportProofError": None,
        },
    )

    summary = _receipt_to_summary(receipt)

    assert summary is not None
    assert summary["pricingMode"] == "network_auto"
    assert summary["pricingBasis"] == "llm_tokens"
    assert summary["promptTokens"] == 12
    assert summary["completionTokens"] == 8
    assert summary["totalTokens"] == 20
    assert summary["usagePriced"] is True
    assert summary["transportMode"] == "multi_worker_direct"
    assert summary["participantCount"] == 2
    assert summary["decentralizedExecution"] is True
    assert summary["llamaCppExecutionMode"] == "llama_cpp_rpc_low_latency"
    assert summary["caiOwnedTransportExecuted"] is False
    assert summary["caiOwnedTransportProofError"] is None
    assert summary["networkAudit"] == receipt.network_audit


def test_receipt_summary_exposes_decentralized_chain_audit() -> None:
    receipt = SimpleNamespace(
        receipt_id="receipt-proof",
        job_id="job-proof",
        finish_reason="stop",
        output_text="ok",
        instance_id="instance-proof",
        pricing_mode="network_auto",
        pricing_basis="llm_tokens",
        prompt_tokens=4,
        completion_tokens=1,
        total_tokens=5,
        token_usage_source="cai_owned_transport_proof",
        reserved_prompt_tokens=8,
        reserved_completion_tokens=64,
        reserved_compute_cost_atomic=150000,
        actual_compute_cost_atomic=1800,
        reservation_surplus_atomic=148200,
        usage_priced=True,
        worker_payouts=[
            {"node_id": "node-a", "reward_atomic": 837},
            {"node_id": "node-b", "reward_atomic": 837},
        ],
        network_audit={
            "requesterNodeId": "node-user",
            "transportMode": "multi_worker_direct",
            "participantCount": 3,
            "participantNodeIds": ["node-user", "node-a", "node-b"],
            "decentralizedExecution": True,
            "directSocketLinkCount": 2,
            "directBidirectionalLinkCount": 1,
            "overlayLinkCount": 0,
            "relayHopsUsed": False,
            "relayBottleneckRisk": False,
            "checkedDirectSocketLinks": [{"source": "node-user", "target": "node-a"}],
            "checkedRelayRoutes": [],
            "relayRouteCandidateCount": 0,
            "relayCoordinatorCandidateCount": 0,
            "llamaCppExecutionStrategy": {
                "executionMode": "cai_owned_transport_required",
            },
            "caiOwnedTransportExecuted": True,
            "caiOwnedTransportProofError": None,
            "caiOwnedTransportExecutionProof": {
                "sessionId": "caiot-proof",
                "instanceId": "instance-proof",
                "executorNodeIds": ["node-a", "node-b"],
                "executionAudit": {
                    "verified": True,
                    "executionDag": {
                        "requesterNodeId": "node-user",
                        "coordinatorNodeId": "node-coord",
                        "executorNodeIds": ["node-a", "node-b"],
                        "participantNodeIds": [
                            "node-user",
                            "node-coord",
                            "node-a",
                            "node-b",
                        ],
                        "expectedStageIds": ["stage-a", "stage-b"],
                        "processedStageIds": ["stage-a", "stage-b"],
                        "missingStageIds": [],
                        "finalOutputBatchIds": ["batch-final"],
                    },
                },
                "shardReceipts": [
                    {
                        "nodeId": "node-a",
                        "metrics": {
                            "promptTokenCount": 4,
                            "completionTokenCount": 1,
                            "inputTokenCount": 5,
                            "outputTokenCount": 1,
                            "payloadSizeBytes": 128,
                            "outputPayloadSizeBytes": 64,
                        },
                    },
                    {
                        "nodeId": "node-b",
                        "metrics": {
                            "promptTokenCount": 4,
                            "completionTokenCount": 1,
                            "inputTokenCount": 5,
                            "outputTokenCount": 1,
                            "payloadSizeBytes": 128,
                            "outputPayloadSizeBytes": 64,
                        },
                    },
                ],
            },
        },
    )

    summary = _receipt_to_summary(receipt)

    assert summary is not None
    audit = summary["decentralizedChainAudit"]
    assert audit["requesterNodeId"] == "node-user"
    assert audit["coordinatorNodeId"] == "node-coord"
    assert audit["executorNodeIds"] == ["node-a", "node-b"]
    assert audit["executorCount"] == 2
    assert audit["route"]["directSocketLinkCount"] == 2
    assert audit["route"]["directBidirectionalLinkCount"] == 1
    assert audit["route"]["checkedDirectSocketLinkCount"] == 1
    assert audit["route"]["checkedRelayRouteCount"] == 0
    assert audit["route"]["relayRouteCandidateCount"] == 0
    assert audit["proof"]["executed"] is True
    assert audit["proof"]["verified"] is True
    assert audit["proof"]["stageCount"] == 2
    assert audit["proof"]["processedStageCount"] == 2
    assert audit["proof"]["finalOutputBatchCount"] == 1
    assert audit["tokens"]["source"] == "cai_owned_transport_proof"
    assert audit["tokens"]["promptTokens"] == 4
    assert audit["tokens"]["completionTokens"] == 1
    assert audit["tokens"]["proofPromptTokenCount"] == 8
    assert audit["tokens"]["proofCompletionTokenCount"] == 2
    assert audit["bytes"]["payloadSizeBytes"] == 256
    assert audit["bytes"]["outputPayloadSizeBytes"] == 128
    assert audit["reward"]["workerPayoutTotalAtomic"] == 1674
    assert audit["reward"]["payoutNodes"] == ["node-a", "node-b"]


def test_job_history_summary_exposes_route_audit_from_receipt() -> None:
    job = SimpleNamespace(
        job_id="job-1",
        created_at="2026-04-26T00:00:00Z",
        status="completed",
        model_id="cai-network/Qwen3-0.6B-GGUF",
        pricing_mode="network_auto",
        pricing_basis="llm_tokens",
        receipt_id="receipt-1",
        last_error=None,
    )
    receipt = SimpleNamespace(
        output_text="hello",
        worker_payouts=[{"node_id": "node-a", "reward_atomic": 10}],
        network_audit={
            "transportMode": "multi_worker_relay",
            "participantCount": 2,
            "participantNodeIds": ["node-a", "node-b"],
            "decentralizedExecution": True,
            "relayBottleneckRisk": False,
            "llamaCppExecutionStrategy": {
                "executionMode": "cai_owned_transport_required",
            },
            "caiOwnedTransportExecuted": True,
            "caiOwnedTransportProofError": None,
        },
    )

    history = _job_to_history_summary(job, receipt)

    assert history is not None
    assert history["transportMode"] == "multi_worker_relay"
    assert history["participantCount"] == 2
    assert history["decentralizedExecution"] is True
    assert history["relayBottleneckRisk"] is False
    assert history["llamaCppExecutionMode"] == "cai_owned_transport_required"
    assert history["caiOwnedTransportExecuted"] is True
    assert history["caiOwnedTransportProofError"] is None
    assert history["decentralizedChainAudit"]["executorNodeIds"] == [
        "node-a",
    ]
    assert history["decentralizedChainAudit"]["reward"]["workerPayoutTotalAtomic"] == 10
    assert history["networkAudit"] == receipt.network_audit


def test_job_history_summary_exposes_settlement_id() -> None:
    job = SimpleNamespace(
        job_id="job-1",
        created_at="2026-04-26T00:00:00Z",
        status="completed",
        model_id="cai-network/Qwen3-0.6B-GGUF",
        pricing_mode="network_auto",
        pricing_basis="llm_tokens",
        receipt_id="receipt-1",
        settlement_id="settlement-1",
        last_error=None,
    )

    history = _job_to_history_summary(job)

    assert history is not None
    assert history["settlementId"] == "settlement-1"


def test_settlement_summary_exposes_chain_transactions() -> None:
    settlement = SimpleNamespace(
        settlement_id="settlement-1",
        status="applied",
        funding_source="wallet",
        compute_cost_atomic=1000,
        tx_fee_atomic=10,
        settlement_fee_atomic=50,
        ai_development_fee_atomic=50,
        worker_reward_atomic=900,
        source_wallet_debit_atomic=1010,
        reserve_debit_atomic=0,
        accepted_attestations=1,
        rejected_attestations=0,
        accepted_bond_atomic=1000,
        committee_validator_ids=["validator-1"],
        applied_at="2026-04-26T00:00:00Z",
        balance_audit={"all_expected_deltas_match": True},
    )
    chain_transactions = [
        {
            "tx_id": "tx-1",
            "tx_type": "settlement_worker_reward",
            "address": "abcd1234abcd1234abcd1234abcd1234",
            "wallet_id": "wallet-1",
            "delta_atomic": 900,
            "balance_after_atomic": 1900,
            "job_id": "job-1",
            "receipt_id": "receipt-1",
            "settlement_id": "settlement-1",
            "payout_id": "payout-1",
            "block_height": 4,
            "block_hash": "block-4",
            "block_created_at": "2026-04-26T00:00:01Z",
        }
    ]

    summary = _settlement_to_summary(
        settlement,
        MoneyPolicy(),
        lambda value, policy=None: f"{value} coins",
        chain_transactions=chain_transactions,
    )

    assert summary is not None
    assert summary["chainRecorded"] is True
    assert summary["chainTransactionCount"] == 1
    assert summary["chainTransactions"][0]["txId"] == "tx-1"
    assert summary["chainTransactions"][0]["txType"] == "settlement_worker_reward"
    assert summary["chainTransactions"][0]["deltaAtomic"] == 900
    assert summary["chainTransactions"][0]["deltaCoins"] == "900 coins"
    assert summary["chainTransactions"][0]["balanceAfterAtomic"] == 1900
    assert summary["chainTransactions"][0]["payoutId"] == "payout-1"


def test_history_page_supports_settlements_with_chain_history() -> None:
    settlement = SimpleNamespace(
        settlement_id="settlement-1",
        status="applied",
        funding_source="wallet",
        compute_cost_atomic=1000,
        tx_fee_atomic=10,
        settlement_fee_atomic=50,
        ai_development_fee_atomic=50,
        worker_reward_atomic=900,
        source_wallet_debit_atomic=1010,
        reserve_debit_atomic=0,
        accepted_attestations=1,
        rejected_attestations=0,
        accepted_bond_atomic=1000,
        committee_validator_ids=["validator-1"],
        applied_at="2026-04-26T00:00:00Z",
        balance_audit={},
    )
    history_calls: list[tuple[str, object]] = []

    service = object.__new__(CaiBridgeService)
    service.wallet_policy = object()
    service.money_policy = MoneyPolicy()
    service.modules = SimpleNamespace(
        wallet=SimpleNamespace(
            get_active_wallet=lambda policy=None: None,
            atomic_to_coins=lambda value, policy=None: str(value),
        ),
        settlement=SimpleNamespace(
            list_settlements=lambda policy=None: [settlement],
        ),
        chain=SimpleNamespace(
            chain_settlement_history=lambda settlement_id, policy=None, **kwargs: history_calls.append(
                (settlement_id, policy)
            )
            or [
                {
                    "tx_id": "tx-1",
                    "tx_type": "settlement_worker_reward",
                    "address": "abcd1234abcd1234abcd1234abcd1234",
                    "delta_atomic": 900,
                    "settlement_id": settlement_id,
                }
            ],
        ),
    )

    result = service.history_page(section="settlements", limit=10)

    assert result["section"] == "settlements"
    assert result["hasMore"] is False
    assert result["items"][0]["settlementId"] == "settlement-1"
    assert result["items"][0]["chainTransactions"][0]["txId"] == "tx-1"
    assert history_calls == [("settlement-1", service.wallet_policy)]


def test_history_page_journal_uses_chain_address_history() -> None:
    wallet = SimpleNamespace(
        wallet_id="wallet-1",
        address="abcd1234abcd1234abcd1234abcd1234",
    )
    history_calls: list[tuple[str, object]] = []

    service = object.__new__(CaiBridgeService)
    service.wallet_policy = object()
    service.money_policy = MoneyPolicy()
    service.modules = SimpleNamespace(
        wallet=SimpleNamespace(
            get_active_wallet=lambda policy=None: wallet,
            atomic_to_coins=lambda value, policy=None: f"{value} coins",
            list_journal_entries=lambda **kwargs: [
                SimpleNamespace(
                    entry_id="local-1",
                    event_type="local_credit",
                    created_at="2026-04-26T00:00:00Z",
                    counterparty_address=None,
                    amount_atomic=1,
                    tx_fee_atomic=0,
                    note="local fallback",
                )
            ],
        ),
        chain=SimpleNamespace(
            chain_address_history=lambda address, policy=None, **kwargs: history_calls.append(
                (address, policy)
            )
            or [
                {
                    "tx_id": "tx-chain-1",
                    "tx_type": "worker_reward_credit",
                    "address": wallet.address,
                    "delta_atomic": 900,
                    "balance_after_atomic": 1900,
                    "note": "Worker reward for node node-a.",
                    "block_height": 4,
                    "block_hash": "block-4",
                    "block_created_at": "2026-04-26T00:00:01Z",
                }
            ],
        ),
    )

    result = service.history_page(section="journal", limit=10)

    assert result["section"] == "journal"
    assert result["items"][0]["source"] == "chain"
    assert result["items"][0]["entryId"] == "tx-chain-1"
    assert result["items"][0]["eventType"] == "worker_reward_credit"
    assert result["items"][0]["amountCoins"] == "900 coins"
    assert result["items"][0]["balanceAfterCoins"] == "1900 coins"
    assert result["items"][0]["blockHeight"] == 4
    assert history_calls == [(wallet.address, service.wallet_policy)]


def test_summary_includes_currency_metadata() -> None:
    service = object.__new__(CaiBridgeService)
    service.state_url = "http://127.0.0.1:52415/state"
    service.cai_url = "http://127.0.0.1:52415"
    service.execution_cai_url = "http://127.0.0.1:52415"
    service.local_node_id = "node-local"
    service.money_policy = MoneyPolicy()
    service.network_config = CaiNetworkConfig()
    service.network_model_policy = object()
    service.wallet_policy = object()
    service.modules = SimpleNamespace(
        ui_state=SimpleNamespace(
            build_interface_snapshot=lambda **kwargs: SimpleNamespace(
                to_dict=lambda: {
                    "wallet": {
                        "has_active_wallet": False,
                        "wallet_name": None,
                        "address": None,
                        "balance_coins": None,
                        "unlocked": False,
                    },
                    "validator": {},
                    "worker": {},
                    "reward": {
                        "pending_count": 0,
                        "finalized_count": 0,
                        "applied_count": 0,
                        "unbound_count": 0,
                    },
                    "compute": {},
                    "chain": {
                        "network": "mainnet",
                        "block_count": 1,
                        "transaction_count": 2,
                        "tip_height": 0,
                        "tip_hash": "abc123",
                        "finalized_height": 0,
                        "last_sync_at": "2026-01-01T00:00:00+00:00",
                        "valid": True,
                    },
                }
            )
        ),
        jobs=SimpleNamespace(
            repair_local_worker_reward_state=lambda **kwargs: {
                "peerSync": {"validatorSet": {"status": "ok"}}
            },
            reconcile_stale_running_job_intents=lambda policy=None: None,
            list_job_intents=lambda policy=None: [],
            list_execution_receipts=lambda policy=None: [],
        ),
        wallet=SimpleNamespace(
            atomic_to_coins=lambda value, policy=None: str(value),
            get_active_wallet=lambda policy=None: None,
            list_journal_entries=lambda **kwargs: [],
        ),
        settlement=SimpleNamespace(
            list_settlements=lambda **kwargs: [],
            list_worker_payouts=lambda **kwargs: [],
        ),
        chain=SimpleNamespace(
            ensure_chain_genesis=lambda **kwargs: 0,
            chain_summary=lambda policy=None: {},
        ),
        cai_owned_diagnostics=SimpleNamespace(
            build_cai_owned_worker_runtime_queue_snapshot=lambda **kwargs: {
                "localNodeId": kwargs.get("local_node_id"),
                "ready": True,
                "recordCount": 1,
                "receivedCount": 1,
                "processingCount": 0,
                "processedCount": 0,
                "failedCount": 0,
                "timedOutCount": 0,
                "deliveredCount": 0,
                "lastError": None,
                "currentBatch": None,
                "records": [],
            }
        ),
        update_channel=SimpleNamespace(
            build_local_update_summary=lambda: {"runtime": {}, "updates": {}}
        ),
        model_distribution=SimpleNamespace(
            list_model_package_manifests=lambda policy=None: [],
            list_cached_chunks=lambda **kwargs: [],
        ),
    )
    service.list_wallet_rows = lambda: []  # pyright: ignore[method-assign]
    service.validator_set = lambda: {}  # pyright: ignore[method-assign]
    service._local_worker_reward_address = lambda: None  # pyright: ignore[method-assign]

    with patch(
        "cai.api.cai_bridge._load_state_payload",
        return_value={
            "nodeIdentities": {
                "node-local": {
                    "resources": {
                        "vramBytes": 4 * 1024**3,
                        "cpuCores": 8,
                    },
                    "readiness": {
                        "caiOwnedTransport": {
                            "protocol": CAI_OWNED_TRANSPORT_PROTOCOL,
                            "protocolVersion": 1,
                            "implemented": True,
                            "runtimeReady": False,
                            "status": "test_adapter_ready",
                        }
                    },
                }
            },
            "nodeMemory": {
                "node-local": {
                    "ramTotal": {"inBytes": 16 * 1024**3},
                    "ramAvailable": {"inBytes": 10 * 1024**3},
                }
            },
        },
    ):
        result = service.summary()

    assert result["currency"] == {
        "code": "CAICN",
        "name": "CAI Network Credit",
        "decimals": 8,
    }
    assert result["economics"]["rewardTokenCode"] == "CAICN"
    assert result["economics"]["dailyIpReserveLimitEnabled"] is True
    assert result["economics"]["dailyIpReserveLimitCoins"] == "1.00000000"
    assert result["economics"]["automaticTokenPricingEnabled"] is True
    assert result["economics"]["inputTokenPriceCoins"] == service.money_policy.automatic_price_per_input_token_coins
    assert result["economics"]["outputTokenPriceCoins"] == service.money_policy.automatic_price_per_output_token_coins
    assert result["economics"]["developerTreasuryWalletId"] == service.money_policy.developer_treasury_wallet_id
    assert result["economics"]["aiDevelopmentWalletId"] == service.money_policy.ai_development_wallet_id
    assert result["economics"]["aiDevelopmentAddress"] == service.money_policy.ai_development_address
    assert result["economics"]["aiDevelopmentFeeBps"] == service.money_policy.ai_development_fee_bps
    assert result["networkConfig"]["chainNetwork"] == service.money_policy.chain_network.value
    assert result["chainStatus"]["tip_height"] == 0
    assert result["chainStatus"]["tip_hash"] == "abc123"
    assert result["safety"]["mode"] == "mainnet_alpha"
    assert {item["code"] for item in result["safety"]["warnings"]} == {
        "mainnet_alpha",
        "single_validator_guarded_alpha",
    }
    assert result["reward"]["applied_count"] == 0
    assert result["worker"]["runtimeQueue"]["localNodeId"] == "node-local"
    assert result["worker"]["runtimeQueue"]["receivedCount"] == 1
    assert result["worker"]["runtime_queue"] == result["worker"]["runtimeQueue"]
    assert result["worker"]["resourceSummary"]["ramBytes"] == 16 * 1024**3
    assert result["worker"]["resourceSummary"]["ramAvailableBytes"] == 10 * 1024**3
    assert result["worker"]["resourceSummary"]["vramBytes"] == 4 * 1024**3
    assert result["worker"]["resources"] == result["worker"]["resourceSummary"]
    assert (
        result["worker"]["readiness"]["caiOwnedTransport"]["status"]
        == "test_adapter_ready"
    )
    assert result["worker"]["caiOwnedTransport"]["runtimeReady"] is False
    assert result["diagnostics"]["maintenanceStatus"] == "ok"
    assert result["diagnostics"]["maintenanceErrors"] == []
    assert result["diagnostics"]["maintenanceResults"] == {
        "repairLocalWorkerRewardState": {
            "peerSync": {"validatorSet": {"status": "ok"}}
        }
    }


def test_summary_reports_non_blocking_maintenance_errors() -> None:
    service = object.__new__(CaiBridgeService)
    service.state_url = "http://127.0.0.1:52415/state"
    service.cai_url = "http://127.0.0.1:52415"
    service.execution_cai_url = "http://127.0.0.1:52415"
    service.local_node_id = "node-local"
    service.money_policy = MoneyPolicy()
    service.network_config = CaiNetworkConfig()
    service.network_model_policy = object()
    service.wallet_policy = object()
    service.modules = SimpleNamespace(
        ui_state=SimpleNamespace(
            build_interface_snapshot=lambda **kwargs: SimpleNamespace(
                to_dict=lambda: {
                    "wallet": {},
                    "validator": {},
                    "worker": {},
                    "reward": {},
                    "compute": {},
                    "chain": {},
                }
            )
        ),
        jobs=SimpleNamespace(
            repair_local_worker_reward_state=lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("reward repair unavailable")
            ),
            reconcile_stale_running_job_intents=lambda policy=None: (
                _ for _ in ()
            ).throw(OSError("stale job reconcile unavailable")),
            list_job_intents=lambda policy=None: [],
            list_execution_receipts=lambda policy=None: [],
        ),
        wallet=SimpleNamespace(
            atomic_to_coins=lambda value, policy=None: str(value),
            get_active_wallet=lambda policy=None: None,
            list_journal_entries=lambda **kwargs: [],
        ),
        settlement=SimpleNamespace(
            list_settlements=lambda **kwargs: [],
            list_worker_payouts=lambda **kwargs: [],
        ),
        chain=SimpleNamespace(
            ensure_chain_genesis=lambda **kwargs: (_ for _ in ()).throw(
                ValueError("genesis unavailable")
            ),
            chain_summary=lambda policy=None: {},
        ),
        cai_owned_diagnostics=SimpleNamespace(
            build_cai_owned_worker_runtime_queue_snapshot=lambda **kwargs: {}
        ),
        update_channel=SimpleNamespace(
            build_local_update_summary=lambda: {"runtime": {}, "updates": {}}
        ),
        model_distribution=SimpleNamespace(
            list_model_package_manifests=lambda policy=None: [],
            list_cached_chunks=lambda **kwargs: [],
        ),
    )
    service.list_wallet_rows = lambda: []  # pyright: ignore[method-assign]
    service.validator_set = lambda: {}  # pyright: ignore[method-assign]
    service._local_worker_reward_address = lambda: None  # pyright: ignore[method-assign]

    with patch(
        "cai.api.cai_bridge._load_state_payload",
        side_effect=OSError("state unavailable"),
    ):
        result = service.summary()

    diagnostics = result["diagnostics"]
    assert diagnostics["maintenanceStatus"] == "degraded"
    assert diagnostics["statePayloadAvailable"] is False
    assert {
        item["operation"] for item in diagnostics["maintenanceErrors"]
    } == {
        "load_state_payload",
        "repair_local_worker_reward_state",
        "ensure_chain_genesis",
        "reconcile_stale_running_job_intents",
    }
    assert {
        item["errorType"] for item in diagnostics["maintenanceErrors"]
    } == {"OSError", "RuntimeError", "ValueError"}


def test_sync_chain_rejects_other_network_before_merge() -> None:
    merge_calls: list[dict[str, object]] = []

    service = object.__new__(CaiBridgeService)
    service.wallet_policy = WalletPolicy()
    service.modules = SimpleNamespace(
        chain=SimpleNamespace(
            merge_remote_chain_payload=lambda payload, policy=None: merge_calls.append(
                payload
            )
            or (0, 0),
            chain_summary=lambda policy=None: {},
        ),
        peer_payload=SimpleNamespace(
            peer_payload_signatures_required=lambda: False,
            validate_peer_payload_network=validate_peer_payload_network,
            verify_peer_payload_signature=verify_peer_payload_signature,
        ),
    )

    try:
        service.sync_chain(
            {
                "chain": {
                    "network": "testnet",
                    "chain_id": "testnet",
                    "schema_version": 1,
                    "blocks": [],
                }
            }
        )
    except ValueError as exc:
        assert "Refusing chain sync payload for network 'testnet'" in str(exc)
    else:
        raise AssertionError("Expected wrong-network chain sync to fail.")

    assert merge_calls == []


def test_sync_validator_set_imports_direct_payload() -> None:
    merge_calls: list[tuple[dict[str, object], str, object]] = []

    service = object.__new__(CaiBridgeService)
    service.cai_url = "http://127.0.0.1:52425"
    service.wallet_policy = object()
    service.validator_set = lambda: {"validators": ["imported"]}  # pyright: ignore[method-assign]
    service.modules = SimpleNamespace(
        validators=SimpleNamespace(
            merge_remote_validator_set_payload=lambda payload, source_url, policy=None: merge_calls.append(
                (payload, source_url, policy)
            )
            or 1,
        )
    )

    payload = {
        "sourceUrl": "http://85.137.164.250:52415/v1/cai/validators",
        "validatorSet": {"validators": []},
    }
    result = service.sync_validator_set(payload)

    assert result["message"] == "Validator set payload imported."
    assert result["importedRecords"] == 1
    assert merge_calls == [
        (
            {"validators": []},
            "http://85.137.164.250:52415/v1/cai/validators",
            service.wallet_policy,
        )
    ]


def test_chain_export_signs_when_active_wallet_is_unlocked() -> None:
    signing_seed = generate_signing_seed()
    public_key_b64 = public_key_b64_from_seed(signing_seed)
    wallet = SimpleNamespace(
        wallet_id="wallet-a",
        address="abcd1234abcd1234abcd1234abcd1234",
    )

    service = object.__new__(CaiBridgeService)
    service.wallet_policy = WalletPolicy()
    service.modules = SimpleNamespace(
        chain=SimpleNamespace(
            export_chain_payload=lambda policy=None: {
                "exported_at": "2026-05-02T00:00:00+00:00",
                "chain": {
                    "network": "mainnet",
                    "chain_id": "mainnet",
                    "schema_version": 1,
                    "blocks": [],
                },
            }
        ),
        peer_payload=peer_payload_module,
        wallet=SimpleNamespace(
            get_active_wallet=lambda policy=None: wallet,
            load_unlocked_wallet_signing_material=lambda wallet, policy=None: {
                "public_key_b64": public_key_b64,
                "signing_seed_b64": encode_bytes(signing_seed),
            },
        ),
    )

    payload = service.chain()

    assert payload["network"] == "mainnet"
    assert payload["chain_id"] == "mainnet"
    assert payload["signature"]["signer_wallet_id"] == "wallet-a"
    ok, error = verify_peer_payload_signature(payload, payload_name="chain")
    assert ok, error


def test_chain_export_reports_unsigned_status_when_wallet_is_locked() -> None:
    wallet = SimpleNamespace(
        wallet_id="wallet-a",
        address="abcd1234abcd1234abcd1234abcd1234",
    )

    service = object.__new__(CaiBridgeService)
    service.wallet_policy = WalletPolicy()
    service.modules = SimpleNamespace(
        chain=SimpleNamespace(
            export_chain_payload=lambda policy=None: {
                "network": "mainnet",
                "chain_id": "mainnet",
                "schema_version": 1,
                "chain": {"blocks": []},
            }
        ),
        peer_payload=peer_payload_module,
        wallet=SimpleNamespace(
            get_active_wallet=lambda policy=None: wallet,
            load_unlocked_wallet_signing_material=lambda wallet, policy=None: None,
        ),
    )

    payload = service.chain()

    assert "signature" not in payload
    assert payload["signatureStatus"] == {
        "signed": False,
        "reason": "wallet_locked",
    }


def test_chain_export_reports_signing_failure_status() -> None:
    wallet = SimpleNamespace(
        wallet_id="wallet-a",
        address="abcd1234abcd1234abcd1234abcd1234",
    )

    service = object.__new__(CaiBridgeService)
    service.wallet_policy = WalletPolicy()
    service.modules = SimpleNamespace(
        chain=SimpleNamespace(
            export_chain_payload=lambda policy=None: {
                "network": "mainnet",
                "chain_id": "mainnet",
                "schema_version": 1,
                "chain": {"blocks": []},
            }
        ),
        peer_payload=peer_payload_module,
        wallet=SimpleNamespace(
            get_active_wallet=lambda policy=None: wallet,
            load_unlocked_wallet_signing_material=lambda wallet, policy=None: {
                "public_key_b64": "",
                "signing_seed_b64": "",
            },
        ),
    )

    payload = service.chain()

    assert "signature" not in payload
    assert payload["signatureStatus"]["signed"] is False
    assert payload["signatureStatus"]["reason"] == "signing_failed"
    assert payload["signatureStatus"]["errorType"] == "ValueError"
    assert "public key is required" in payload["signatureStatus"]["message"]


def test_chunk_inventory_uses_local_node_id_and_source_kind() -> None:
    service = object.__new__(CaiBridgeService)
    service.local_node_id = "node-local"
    service.cai_url = "http://198.51.100.10:52415"
    service.wallet_policy = object()
    service.modules = SimpleNamespace(
        model_distribution=SimpleNamespace(
            export_chunk_inventory_payload=lambda source_id, source_kind, endpoint_base_url=None, policy=None: SimpleNamespace(
                to_dict=lambda: {
                    "source_id": source_id,
                    "source_kind": source_kind,
                    "endpoint_base_url": endpoint_base_url,
                    "records": [],
                }
            )
        )
    )

    result = service.chunk_inventory(source_kind="peer_cache")

    assert result == {
        "source_id": "node-local",
        "source_kind": "peer_cache",
        "endpoint_base_url": "http://198.51.100.10:52415",
        "records": [],
    }


def test_sync_chunk_inventory_uses_state_and_cai_url() -> None:
    calls: list[tuple[dict[str, object], str, str, object, str | None]] = []

    service = object.__new__(CaiBridgeService)
    service.local_node_id = "node-local"
    service.state_url = "http://127.0.0.1:52415/state"
    service.cai_url = "http://127.0.0.1:52415"
    service.wallet_policy = object()
    service.modules = SimpleNamespace(
        model_distribution=SimpleNamespace(
            sync_chunk_inventory_from_cai_peers=lambda *, state_payload, cai_url, source_kind, policy=None, local_node_id=None: calls.append(
                (state_payload, cai_url, source_kind, policy, local_node_id)
            )
            or SimpleNamespace(
                attempted_peers=2,
                successful_peers=1,
                imported_payloads=1,
                pruned_payloads=0,
                peer_urls=["http://198.51.100.10:52415/v1/cai/chunk-inventory?source_kind=peer_cache"],
            )
        )
    )

    with patch("cai.api.cai_bridge._load_state_payload", return_value={"nodeIdentities": {}}):
        result = service.sync_chunk_inventory(source_kind="peer_cache")

    assert calls == [
        (
            {"nodeIdentities": {}},
            "http://127.0.0.1:52415",
            "peer_cache",
            service.wallet_policy,
            "node-local",
        )
    ]
    assert result["attemptedPeers"] == 2
    assert result["successfulPeers"] == 1
    assert result["importedPayloads"] == 1
    assert result["prunedPayloads"] == 0


def test_node_capabilities_exports_state_payload() -> None:
    refresh_calls: list[tuple[dict[str, object], str, object, str | None]] = []

    service = object.__new__(CaiBridgeService)
    service.local_node_id = "node-local"
    service.state_url = "http://127.0.0.1:52415/state"
    service.cai_url = "http://127.0.0.1:52415"
    service.wallet_policy = object()
    service.modules = SimpleNamespace(
        peer_payload=SimpleNamespace(
            add_peer_payload_metadata=lambda payload, policy=None: {
                "network": "mainnet",
                **payload,
            }
        ),
        node_capabilities=SimpleNamespace(
            export_node_capabilities_payload=lambda *, state_payload, cai_url, local_node_id, policy=None: {
                "records": [
                    {
                        "node_id": local_node_id,
                        "api_urls": [cai_url],
                        "state_seen": state_payload,
                    }
                ]
            },
            refresh_local_node_capabilities=lambda *, state_payload, cai_url, local_node_id, policy=None: refresh_calls.append(
                (state_payload, cai_url, policy, local_node_id)
            )
            or [],
        )
    )

    with patch("cai.api.cai_bridge._load_state_payload", return_value={"nodeIdentities": {}}):
        result = service.node_capabilities()

    assert result["records"][0]["node_id"] == "node-local"
    assert result["records"][0]["api_urls"] == ["http://127.0.0.1:52415"]
    assert refresh_calls == [
        (
            {"nodeIdentities": {}},
            "http://127.0.0.1:52415",
            service.wallet_policy,
            "node-local",
        )
    ]


def test_attest_worker_capability_fetches_source_and_records_attestation() -> None:
    attest_calls: list[dict[str, object]] = []
    merge_calls: list[dict[str, object]] = []
    record = SimpleNamespace(
        node_id="node-worker",
        worker_verified=True,
        payload_signature_valid=True,
        worker_reward_address="worker-address",
        node_public_key_address="worker-key",
        worker_allowed_model_ids=[],
        model_ids=[],
        resource_summary={"vramBytes": 8_000},
        readiness={"caiOwnedTransport": {"runtimeReady": True}},
    )
    service = object.__new__(CaiBridgeService)
    service.state_url = "http://127.0.0.1:52415/state"
    service.cai_url = "http://127.0.0.1:52415"
    service.wallet_policy = object()
    service.modules = SimpleNamespace(
        node_config=SimpleNamespace(
            get_validator_attestation_status=lambda **kwargs: SimpleNamespace(
                can_attest=True,
                validator_id="validator-a",
                reason="ok",
            ),
            load_or_create_node_config=lambda policy=None: SimpleNamespace(
                validator_wallet_id="validator-wallet",
                validator_address="validator-a",
            ),
        ),
        node_capabilities=SimpleNamespace(
            verified_node_capability_records_from_payload=(
                lambda *args, **kwargs: [record]
            ),
            merge_remote_node_capabilities_payload=(
                lambda payload, source_url, policy=None, only_node_id=None: (
                    merge_calls.append(
                        {
                            "source_url": source_url,
                            "only_node_id": only_node_id,
                        }
                    )
                    or 1
                )
            ),
            list_node_capabilities=lambda policy=None: [record],
        ),
        wallet=SimpleNamespace(
            find_wallet_by_id=lambda wallet_id, policy=None: SimpleNamespace(
                wallet_id=wallet_id,
            ),
            load_unlocked_wallet_signing_material=lambda wallet, policy=None: {
                "public_key_b64": "validator-public-key",
                "signing_seed_b64": "validator-signing-seed",
            },
        ),
        worker_capability_attestations=SimpleNamespace(
            create_worker_capability_challenge=lambda *args, **kwargs: {
                "challenge_id": "challenge-1",
                "worker_node_id": "node-worker",
            },
            verify_worker_capability_challenge_receipt=lambda *args, **kwargs: (
                True,
                None,
            ),
            record_worker_capability_attestation=lambda *args, **kwargs: attest_calls.append(
                kwargs
            )
            or SimpleNamespace(attestation_id="attestation-1")
        ),
    )

    with (
        patch("cai.api.cai_bridge._load_state_payload", return_value={"nodeIdentities": {}}),
        patch(
            "cai.api.cai_bridge.urlopen",
            side_effect=[
                _FakeResponse(b'{"records": []}'),
                _FakeResponse(b'{"receipt": {"challenge_id": "challenge-1"}}'),
            ],
        ),
    ):
        result = service.attest_worker_capability(
            {"sourceUrl": "http://198.51.100.10:52415", "nodeId": "node-worker"}
        )

    assert result["accepted"] is True
    assert result["workerNodeId"] == "node-worker"
    assert merge_calls[0]["only_node_id"] == "node-worker"
    assert attest_calls
    assert attest_calls[0]["validator_id"] == "validator-a"
    assert attest_calls[0]["probe_result"]["sourceUrl"] == (
        "http://198.51.100.10:52415/v1/cai/node-capabilities"
    )
    assert attest_calls[0]["probe_result"]["challengeVerified"] is True


def test_attest_worker_capability_worker_submitted_payload_issues_challenge() -> None:
    merge_calls: list[dict[str, object]] = []
    record = SimpleNamespace(
        node_id="node-worker",
        worker_verified=True,
        payload_signature_valid=True,
        worker_reward_address="worker-address",
        node_public_key_address="worker-key",
        worker_allowed_model_ids=[],
        model_ids=[],
        resource_summary={"vramBytes": 8_000},
        readiness={"caiOwnedTransport": {"runtimeReady": True}},
    )
    service = object.__new__(CaiBridgeService)
    service.state_url = "http://127.0.0.1:52415/state"
    service.cai_url = "http://127.0.0.1:52415"
    service.wallet_policy = object()
    service.modules = SimpleNamespace(
        node_config=SimpleNamespace(
            get_validator_attestation_status=lambda **kwargs: SimpleNamespace(
                can_attest=True,
                validator_id="validator-a",
                reason="ok",
            ),
            load_or_create_node_config=lambda policy=None: SimpleNamespace(
                validator_wallet_id="validator-wallet",
                validator_address="validator-a",
            ),
        ),
        node_capabilities=SimpleNamespace(
            verified_node_capability_records_from_payload=(
                lambda *args, **kwargs: [record]
            ),
            merge_remote_node_capabilities_payload=(
                lambda payload, source_url, policy=None, only_node_id=None: (
                    merge_calls.append(
                        {
                            "source_url": source_url,
                            "only_node_id": only_node_id,
                        }
                    )
                    or 1
                )
            ),
        ),
        wallet=SimpleNamespace(
            find_wallet_by_id=lambda wallet_id, policy=None: SimpleNamespace(
                wallet_id=wallet_id,
            ),
            load_unlocked_wallet_signing_material=lambda wallet, policy=None: {
                "public_key_b64": "validator-public-key",
                "signing_seed_b64": "validator-signing-seed",
            },
        ),
        worker_capability_attestations=SimpleNamespace(
            create_worker_capability_challenge=lambda *args, **kwargs: {
                "challenge_id": "challenge-1",
                "validator_id": kwargs["validator_id"],
                "worker_node_id": "node-worker",
            }
        ),
    )

    with (
        patch("cai.api.cai_bridge._load_state_payload", return_value={"nodeIdentities": {}}),
        patch("cai.api.cai_bridge.urlopen") as urlopen_mock,
    ):
        result = service.attest_worker_capability(
            {
                "capabilityPayload": {"records": []},
                "nodeId": "node-worker",
            }
        )

    assert result["accepted"] is False
    assert result["challengeRequired"] is True
    assert result["challenge"]["challenge_id"] == "challenge-1"
    assert result["probe"]["attestationMode"] == "worker_submitted"
    assert merge_calls == [
        {
            "source_url": "worker-submitted",
            "only_node_id": "node-worker",
        }
    ]
    urlopen_mock.assert_not_called()


def test_attest_worker_capability_worker_submitted_payload_records_receipt() -> None:
    attest_calls: list[dict[str, object]] = []
    record = SimpleNamespace(
        node_id="node-worker",
        worker_verified=True,
        payload_signature_valid=True,
        worker_reward_address="worker-address",
        node_public_key_address="worker-key",
        worker_allowed_model_ids=[],
        model_ids=[],
        resource_summary={"vramBytes": 8_000},
        readiness={"caiOwnedTransport": {"runtimeReady": True}},
    )
    service = object.__new__(CaiBridgeService)
    service.state_url = "http://127.0.0.1:52415/state"
    service.cai_url = "http://127.0.0.1:52415"
    service.wallet_policy = object()
    service.modules = SimpleNamespace(
        node_config=SimpleNamespace(
            get_validator_attestation_status=lambda **kwargs: SimpleNamespace(
                can_attest=True,
                validator_id="validator-a",
                reason="ok",
            ),
            load_or_create_node_config=lambda policy=None: SimpleNamespace(
                validator_wallet_id="validator-wallet",
                validator_address="validator-a",
            ),
        ),
        node_capabilities=SimpleNamespace(
            verified_node_capability_records_from_payload=(
                lambda *args, **kwargs: [record]
            ),
            merge_remote_node_capabilities_payload=lambda *args, **kwargs: 1,
        ),
        wallet=SimpleNamespace(
            find_wallet_by_id=lambda wallet_id, policy=None: SimpleNamespace(
                wallet_id=wallet_id,
            ),
            load_unlocked_wallet_signing_material=lambda wallet, policy=None: {
                "public_key_b64": "validator-public-key",
                "signing_seed_b64": "validator-signing-seed",
            },
        ),
        worker_capability_attestations=SimpleNamespace(
            verify_worker_capability_challenge=lambda *args, **kwargs: (True, None),
            verify_worker_capability_challenge_receipt=lambda *args, **kwargs: (
                True,
                None,
            ),
            record_worker_capability_attestation=lambda *args, **kwargs: attest_calls.append(
                kwargs
            )
            or SimpleNamespace(attestation_id="attestation-1"),
        ),
    )

    with (
        patch("cai.api.cai_bridge._load_state_payload", return_value={"nodeIdentities": {}}),
        patch("cai.api.cai_bridge.urlopen") as urlopen_mock,
    ):
        result = service.attest_worker_capability(
            {
                "capabilityPayload": {"records": []},
                "nodeId": "node-worker",
                "challenge": {
                    "challenge_id": "challenge-1",
                    "validator_id": "validator-a",
                    "worker_node_id": "node-worker",
                },
                "challengeReceipt": {"challenge_id": "challenge-1"},
            }
        )

    assert result["accepted"] is True
    assert result["workerNodeId"] == "node-worker"
    assert attest_calls
    assert attest_calls[0]["note"] == (
        "Validator accepted worker-submitted capability challenge."
    )
    assert attest_calls[0]["probe_result"]["attestationMode"] == "worker_submitted"
    assert attest_calls[0]["probe_result"]["challengeVerified"] is True
    urlopen_mock.assert_not_called()


def test_worker_capability_challenge_returns_signed_receipt() -> None:
    record = SimpleNamespace(
        node_id="node-local",
        worker_enabled=True,
        worker_reward_address="worker-key",
        node_public_key_b64=None,
        node_public_key_address=None,
        payload_public_key_address=None,
    )
    service = object.__new__(CaiBridgeService)
    service.state_url = "http://127.0.0.1:52415/state"
    service.cai_url = "http://127.0.0.1:52415"
    service.local_node_id = "node-local"
    service.wallet_policy = object()
    service.modules = SimpleNamespace(
        node_capabilities=SimpleNamespace(
            refresh_local_node_capabilities=lambda **kwargs: [record],
        ),
        wallet=SimpleNamespace(
            get_active_wallet=lambda policy=None: SimpleNamespace(
                address="worker-key"
            ),
            load_unlocked_wallet_signing_material=lambda wallet, policy=None: {
                "public_key_b64": "worker-public-key",
                "signing_seed_b64": "worker-signing-seed",
            },
            normalize_address=lambda address: str(address).strip().lower(),
        ),
        wallet_signing=SimpleNamespace(
            address_from_public_key_b64=lambda public_key_b64: "worker-key",
        ),
        worker_capability_attestations=SimpleNamespace(
            verify_worker_capability_challenge=lambda *args, **kwargs: (
                True,
                None,
            ),
            worker_capability_fingerprint_from_record=lambda item: "fingerprint-1",
            create_worker_capability_challenge_receipt=lambda *args, **kwargs: {
                "challenge_id": kwargs["challenge"]["challenge_id"],
                "worker_public_key_b64": kwargs["worker_public_key_b64"],
            },
        ),
    )

    with patch("cai.api.cai_bridge._load_state_payload", return_value={}):
        result = service.worker_capability_challenge(
            {
                "challenge_id": "challenge-1",
                "worker_node_id": "node-local",
                "capability_fingerprint": "fingerprint-1",
            }
        )

    assert result["accepted"] is True
    assert result["receipt"]["challenge_id"] == "challenge-1"
    assert result["receipt"]["worker_public_key_b64"] == "worker-public-key"
    assert record.node_public_key_address == "worker-key"


def test_sync_node_capabilities_uses_state_and_cai_url() -> None:
    sync_calls: list[tuple[dict[str, object], str, object, str | None]] = []

    service = object.__new__(CaiBridgeService)
    service.local_node_id = "node-local"
    service.state_url = "http://127.0.0.1:52415/state"
    service.cai_url = "http://127.0.0.1:52415"
    service.wallet_policy = object()
    service.modules = SimpleNamespace(
        node_capabilities=SimpleNamespace(
            refresh_local_node_capabilities=lambda **kwargs: [],
            sync_node_capabilities_from_cai_peers=lambda *, state_payload, cai_url, policy=None, local_node_id=None: sync_calls.append(
                (state_payload, cai_url, policy, local_node_id)
            )
            or SimpleNamespace(
                attempted_peers=2,
                successful_peers=1,
                imported_records=1,
                pruned_records=0,
                peer_urls=["http://198.51.100.10:52415/v1/cai/node-capabilities"],
            ),
            list_node_capabilities=lambda policy=None: [
                SimpleNamespace(node_id="node-peer", relay_enabled=True)
            ],
        )
    )

    with patch("cai.api.cai_bridge._load_state_payload", return_value={"nodeIdentities": {}}):
        result = service.sync_node_capabilities()

    assert result["attemptedPeers"] == 2
    assert result["successfulPeers"] == 1
    assert result["importedRecords"] == 1
    assert result["prunedRecords"] == 0
    assert result["recordCount"] == 1
    assert result["records"][0]["node_id"] == "node-peer"
    assert sync_calls == [
        (
            {"nodeIdentities": {}},
            "http://127.0.0.1:52415",
            service.wallet_policy,
            "node-local",
        )
    ]


def test_route_health_summary_and_probe_use_route_health_module() -> None:
    probe_calls: list[tuple[dict[str, object], object, str | None]] = []
    data_probe_calls: list[tuple[dict[str, object], object, str | None]] = []
    overlay_calls: list[tuple[dict[str, object], object]] = []
    relay_score_calls: list[tuple[dict[str, object], object]] = []

    service = object.__new__(CaiBridgeService)
    service.local_node_id = "node-local"
    service.state_url = "http://127.0.0.1:52415/state"
    service.wallet_policy = object()
    record = SimpleNamespace(route_id="route-1", reachable=True)
    data_record = SimpleNamespace(route_id="data-1", reachable=True)
    overlay_record = SimpleNamespace(route_id="overlay-1", reachable=True)
    service.modules = SimpleNamespace(
        route_health=SimpleNamespace(
            list_route_health_records=lambda policy=None: [record],
            probe_direct_api_routes=lambda *, state_payload, local_node_id=None, policy=None: probe_calls.append(
                (state_payload, policy, local_node_id)
            )
            or [record],
            probe_direct_data_routes=lambda *, state_payload, local_node_id=None, policy=None: data_probe_calls.append(
                (state_payload, policy, local_node_id)
            )
            or [data_record],
            record_overlay_routes_from_state=lambda *, state_payload, policy=None: overlay_calls.append(
                (state_payload, policy)
            )
            or [overlay_record],
            prune_stale_route_health_records=lambda *, policy=None: 3,
            score_relay_route_candidates=lambda *, state_payload, route_health_records=None: relay_score_calls.append(
                (state_payload, route_health_records)
            )
            or {"candidateCount": 0, "bottleneckRisk": False},
        )
    )

    summary = service.route_health()
    with patch("cai.api.cai_bridge._load_state_payload", return_value={"nodeIdentities": {}}):
        probe = service.probe_route_health()

    assert summary["recordCount"] == 1
    assert summary["records"][0]["route_id"] == "route-1"
    assert probe["probedRecords"] == 3
    assert probe["directProbedRecords"] == 1
    assert probe["directDataProbedRecords"] == 1
    assert probe["overlayObservedRecords"] == 1
    assert probe["prunedRecords"] == 3
    assert probe["relayScore"]["candidateCount"] == 0
    assert probe_calls == [
        (
            {"nodeIdentities": {}},
            service.wallet_policy,
            "node-local",
        )
    ]
    assert data_probe_calls == [
        (
            {"nodeIdentities": {}},
            service.wallet_policy,
            "node-local",
        )
    ]
    assert overlay_calls == [({"nodeIdentities": {}}, service.wallet_policy)]
    assert relay_score_calls == [({"nodeIdentities": {}}, [record])]


def test_compute_cells_reports_low_latency_worker_cell() -> None:
    service = object.__new__(CaiBridgeService)
    service.state_url = "http://127.0.0.1:52415/state"
    service.wallet_policy = object()
    record = SimpleNamespace(
        source_node_id="node-a",
        sink_node_id="node-b",
        route_type="llama_cpp_rpc_direct",
        reachable=True,
        endpoint_url="llama-cpp-rpc://198.51.100.11:52435",
        checked_at="2026-05-03T00:00:00+00:00",
        latency_ms=7.0,
    )
    service.modules = SimpleNamespace(
        decentralized_compute=SimpleNamespace(
            plan_llama_cpp_distributed_execution=plan_llama_cpp_distributed_execution,
        ),
        route_health=SimpleNamespace(
            list_route_health_records=lambda policy=None: [record],
            llama_cpp_compute_cell_profile_for_path=llama_cpp_compute_cell_profile_for_path,
        )
    )
    state_payload = {
        "nodeIdentities": {
            "node-a": {"workerEnabled": True},
            "node-b": {"workerEnabled": True},
            "node-relay": {"relayEnabled": True},
        }
    }

    with patch("cai.api.cai_bridge._load_state_payload", return_value=state_payload):
        payload = service.compute_cells()

    assert payload["workerNodeIds"] == ["node-a", "node-b"]
    assert payload["cellCount"] == 2
    first_cell = payload["cells"][0]
    assert first_cell["sourceNodeId"] == "node-a"
    assert first_cell["profile"] == "low_latency_sharded_cell"
    assert first_cell["readyForLlamaCppRpc"] is True
    assert first_cell["executionMode"] == "llama_cpp_rpc_low_latency"
    assert first_cell["requiresCaiOwnedTransport"] is False


def test_compute_cells_marks_unproven_cell_as_cai_owned_transport_required() -> None:
    service = object.__new__(CaiBridgeService)
    service.state_url = "http://127.0.0.1:52415/state"
    service.wallet_policy = object()
    service.modules = SimpleNamespace(
        decentralized_compute=SimpleNamespace(
            plan_llama_cpp_distributed_execution=plan_llama_cpp_distributed_execution,
        ),
        route_health=SimpleNamespace(
            list_route_health_records=lambda policy=None: [],
            llama_cpp_compute_cell_profile_for_path=llama_cpp_compute_cell_profile_for_path,
        ),
    )
    state_payload = {
        "nodeIdentities": {
            "node-a": {"workerEnabled": True},
            "node-b": {"workerEnabled": True},
        }
    }

    with patch("cai.api.cai_bridge._load_state_payload", return_value=state_payload):
        payload = service.compute_cells()

    assert payload["workerNodeIds"] == ["node-a", "node-b"]
    assert payload["cellCount"] == 2
    assert payload["readyCellCount"] == 0
    assert payload["caiOwnedTransportRequiredCellCount"] == 2
    first_cell = payload["cells"][0]
    assert first_cell["profile"] == "unproven_sharded_cell"
    assert first_cell["executionMode"] == "cai_owned_transport_required"
    assert first_cell["requiresCaiOwnedTransport"] is True
    route_policy = first_cell["caiOwnedTransport"]["routePolicy"]
    assert route_policy["manualTunnelRequired"] is False


def test_chunk_payload_reads_public_shared_chunk_bytes() -> None:
    service = object.__new__(CaiBridgeService)
    service.wallet_policy = object()
    service.modules = SimpleNamespace(
        model_distribution=SimpleNamespace(
            load_model_package_manifest=lambda catalog_id, version, policy=None: SimpleNamespace(
                package_kind="public_shared"
            ),
            ModelPackageKind=SimpleNamespace(PUBLIC_SHARED="public_shared"),
            _read_cached_chunk_payload=lambda chunk_id, *, catalog_id, version, policy=None: b"chunk-bytes",
        )
    )

    payload = service.chunk_payload(catalog_id="demo", version="v1", chunk_id="chunk-1")

    assert payload == b"chunk-bytes"


def test_chunk_payload_rejects_private_package_bytes() -> None:
    service = object.__new__(CaiBridgeService)
    service.wallet_policy = object()
    service.modules = SimpleNamespace(
        model_distribution=SimpleNamespace(
            load_model_package_manifest=lambda catalog_id, version, policy=None: SimpleNamespace(
                package_kind="private_curated"
            ),
            ModelPackageKind=SimpleNamespace(PUBLIC_SHARED="public_shared"),
            _read_cached_chunk_payload=lambda chunk_id, *, catalog_id, version, policy=None: b"private-bytes",
        )
    )

    try:
        service.chunk_payload(catalog_id="private", version="v1", chunk_id="chunk-1")
    except ValueError as exc:
        assert "disabled for non-public package private@v1" in str(exc)
    else:
        raise AssertionError("Expected private package chunk payload to be blocked.")

