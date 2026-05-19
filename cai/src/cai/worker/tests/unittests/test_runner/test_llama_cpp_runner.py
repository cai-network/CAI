# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import os
import asyncio
import subprocess
from contextlib import suppress
from pathlib import Path

from aiohttp import WSMsgType
from cai_compute_chain.decentralized_compute import (
    CAI_OWNED_TRANSPORT_PROTOCOL,
    deterministic_cai_owned_transport_session_id,
)
from cai.shared.types.chunks import ErrorChunk, TokenChunk
from cai.shared.models.model_cards import InferenceBackend, ModelCard, ModelId, ModelTask
from cai.shared.types.common import CommandId, Host, NodeId
from cai.shared.types.events import ChunkGenerated
from cai.shared.types.memory import Memory
from cai.shared.types.tasks import LoadModel, StartWarmup, TextGeneration
from cai.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from cai.shared.types.worker.instances import BoundInstance, MlxRingInstance
from cai.shared.types.worker.instances import LlamaCppRelayRoute
from cai.shared.types.worker.runners import (
    RunnerId,
    RunnerIdle,
    RunnerLoaded,
    RunnerReady,
    ShardAssignments,
)
from cai.shared.types.worker.shards import PipelineShardMetadata
from cai.worker.runner.llama_cpp.runner import (
    DEFAULT_DISTRIBUTED_READY_TIMEOUT_SECONDS,
    DEFAULT_READY_TIMEOUT_SECONDS,
    Runner,
    _apply_qwen3_message_directives,
    _cai_owned_transport_generation_require_data_plane_route,
    _cai_owned_transport_generation_require_executor_readiness,
    _cai_owned_transport_generation_require_proven_data_plane_route,
    _cai_owned_transport_generation_require_runtime_ready,
    _cai_owned_transport_generation_require_shard_readiness,
    _cai_owned_transport_skip_local_llama_server,
    _find_llama_rpc_server_binary,
    _find_llama_server_binary,
    _windows_subprocess_flags,
)
from cai.worker.runner.llama_cpp.relay_tunnel import (
    DEFAULT_RELAY_STREAM_CHUNK_SIZE,
    LlamaCppRelayTunnelManager,
    LlamaCppReverseRelayManager,
    _RELAY_EOF_MESSAGE,
    _RELAY_TARGET_CONNECTED_MESSAGE,
    _llama_cpp_rpc_hello_payload,
    _probe_llama_cpp_rpc_hello_stream,
    _relay_ws_connect_kwargs,
)


class EventCollector:
    def __init__(self) -> None:
        self.events = []

    def send(self, event) -> None:
        self.events.append(event)


class RelayTunnelStub:
    def __init__(
        self,
        sink_node_id: NodeId,
        endpoint: Host,
        *,
        selected_mode: str | None = None,
        selected_route: LlamaCppRelayRoute | None = None,
    ) -> None:
        self.sink_node_id = sink_node_id
        self.endpoint = endpoint
        self.selected_mode = selected_mode
        self.selected_route = selected_route
        self.llama_cpp_rpc_probe_calls: list[NodeId] = []

    def local_endpoint_for_sink(self, sink_node_id: NodeId) -> Host | None:
        if sink_node_id == self.sink_node_id:
            return self.endpoint
        return None

    def probe_route(self, sink_node_id: NodeId, *, timeout: float) -> None:
        raise AssertionError("runner should use llama.cpp RPC HELLO probe")

    def probe_llama_cpp_rpc_route(self, sink_node_id: NodeId, *, timeout: float) -> str:
        assert sink_node_id == self.sink_node_id
        assert timeout > 0
        self.llama_cpp_rpc_probe_calls.append(sink_node_id)
        return "1.0.0"

    def selected_route_for_sink(
        self,
        sink_node_id: NodeId,
    ) -> tuple[str, LlamaCppRelayRoute] | None:
        if sink_node_id != self.sink_node_id or self.selected_route is None:
            return None
        return (self.selected_mode or "relay", self.selected_route)


def _task() -> TextGeneration:
    return TextGeneration(
        task_id="task-a",
        instance_id="instance-a",
        command_id=CommandId("command-a"),
        task_params=TextGenerationTaskParams(
            model="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
            stream=False,
            max_output_tokens=8,
            temperature=0.0,
        ),
    )


def _qwen3_task(**overrides) -> TextGeneration:
    params = TextGenerationTaskParams(
        model="Qwen/Qwen3-0.6B-GGUF",
        input=[InputMessage(role="user", content=InputMessageContent("2+2=?"))],
        stream=False,
        max_output_tokens=8,
    )
    if overrides:
        params = params.model_copy(update=overrides)
    return TextGeneration(
        task_id="task-qwen3",
        instance_id="instance-qwen3",
        command_id=CommandId("command-qwen3"),
        task_params=params,
    )


def _bound_instance(*, device_rank: int, world_size: int) -> BoundInstance:
    node_ids = [NodeId("node-a"), NodeId("node-b")]
    runner_ids = [RunnerId("runner-a"), RunnerId("runner-b")]
    model_card = ModelCard(
        model_id=ModelId("Qwen/Qwen2.5-0.5B-Instruct-GGUF"),
        storage_size=Memory.from_mb(1),
        n_layers=8,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        inference_backend=InferenceBackend.LlamaCpp,
    )
    shards = {
        runner_ids[idx]: PipelineShardMetadata(
            model_card=model_card,
            device_rank=idx,
            world_size=world_size,
            start_layer=idx * 4,
            end_layer=(idx + 1) * 4,
            n_layers=8,
        )
        for idx in range(world_size)
    }
    instance = MlxRingInstance(
        instance_id="instance-a",
        shard_assignments=ShardAssignments(
            model_id=model_card.model_id,
            node_to_runner={
                node_ids[idx]: runner_ids[idx] for idx in range(world_size)
            },
            runner_to_shard=shards,
        ),
        hosts_by_node={
            node_ids[0]: [
                {"ip": "0.0.0.0", "port": 50052},
                {"ip": "10.0.0.2", "port": 50052},
            ],
            node_ids[1]: [
                {"ip": "10.0.0.1", "port": 50052},
                {"ip": "0.0.0.0", "port": 50052},
            ],
        },
        ephemeral_port=50052,
    )
    return BoundInstance(
        instance=instance,
        bound_runner_id=runner_ids[device_rank],
        bound_node_id=node_ids[device_rank],
    )


def test_run_text_generation_wraps_success_in_chunk_generated() -> None:
    runner = object.__new__(Runner)
    runner.model_id = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    runner.event_sender = EventCollector()
    runner._is_rpc_worker = lambda: False  # type: ignore[attr-defined]
    runner._request_chat_completion = lambda payload: {  # type: ignore[attr-defined]
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "4"},
            }
        ],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "total_tokens": 4,
        },
    }

    runner._run_text_generation(_task())

    assert len(runner.event_sender.events) == 1
    event = runner.event_sender.events[0]
    assert isinstance(event, ChunkGenerated)
    assert event.command_id == "command-a"
    assert isinstance(event.chunk, TokenChunk)
    assert event.chunk.text == "4"
    assert event.chunk.finish_reason == "stop"


def test_run_text_generation_defaults_terminal_finish_reason_and_preserves_reasoning() -> None:
    runner = object.__new__(Runner)
    runner.model_id = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    runner.event_sender = EventCollector()
    runner._is_rpc_worker = lambda: False  # type: ignore[attr-defined]
    runner._request_chat_completion = lambda payload: {  # type: ignore[attr-defined]
        "choices": [
            {
                "message": {
                    "reasoning_content": "thinking",
                    "content": "4",
                },
            }
        ],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "total_tokens": 4,
        },
    }

    runner._run_text_generation(_task())

    assert len(runner.event_sender.events) == 2
    first_event = runner.event_sender.events[0]
    second_event = runner.event_sender.events[1]

    assert isinstance(first_event, ChunkGenerated)
    assert isinstance(first_event.chunk, TokenChunk)
    assert first_event.chunk.text == "thinking"
    assert first_event.chunk.is_thinking is True
    assert first_event.chunk.finish_reason is None
    assert first_event.chunk.usage is None

    assert isinstance(second_event, ChunkGenerated)
    assert isinstance(second_event.chunk, TokenChunk)
    assert second_event.chunk.text == "4"
    assert second_event.chunk.is_thinking is False
    assert second_event.chunk.finish_reason == "stop"
    assert second_event.chunk.usage is not None


def test_run_text_generation_wraps_errors_in_chunk_generated() -> None:
    runner = object.__new__(Runner)
    runner.model_id = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    runner.event_sender = EventCollector()
    runner._is_rpc_worker = lambda: False  # type: ignore[attr-defined]

    def _raise(_payload):
        raise RuntimeError("boom")

    runner._request_chat_completion = _raise  # type: ignore[attr-defined]

    runner._run_text_generation(_task())

    assert len(runner.event_sender.events) == 1
    event = runner.event_sender.events[0]
    assert isinstance(event, ChunkGenerated)
    assert event.command_id == "command-a"
    assert isinstance(event.chunk, ErrorChunk)
    assert event.chunk.error_message == "boom"


def test_distributed_rpc_worker_skips_generation_chunks() -> None:
    runner = object.__new__(Runner)
    runner.bound_instance = _bound_instance(device_rank=1, world_size=2)
    runner.instance = runner.bound_instance.instance
    runner.shard_metadata = runner.bound_instance.bound_shard
    runner.model_id = runner.shard_metadata.model_card.model_id
    runner.event_sender = EventCollector()

    runner._run_text_generation(_task())

    assert runner.event_sender.events == []


def test_remote_rpc_servers_excludes_self_for_coordinator() -> None:
    runner = object.__new__(Runner)
    runner.bound_instance = _bound_instance(device_rank=0, world_size=2)
    runner.instance = runner.bound_instance.instance
    runner.shard_metadata = runner.bound_instance.bound_shard

    endpoints = runner._remote_rpc_servers()

    assert endpoints == ["10.0.0.2:50052"]


def test_cai_owned_transport_session_payload_is_rank_stable() -> None:
    coordinator = object.__new__(Runner)
    coordinator.bound_instance = _bound_instance(device_rank=0, world_size=2)
    coordinator.instance = coordinator.bound_instance.instance
    coordinator.shard_metadata = coordinator.bound_instance.bound_shard
    coordinator.model_id = coordinator.shard_metadata.model_card.model_id

    worker = object.__new__(Runner)
    worker.bound_instance = _bound_instance(device_rank=1, world_size=2)
    worker.instance = worker.bound_instance.instance
    worker.shard_metadata = worker.bound_instance.bound_shard
    worker.model_id = worker.shard_metadata.model_card.model_id

    expected_session_id = deterministic_cai_owned_transport_session_id(
        "instance-a",
        ["node-a", "node-b"],
        task_id="task-a",
    )

    payload = coordinator._cai_owned_transport_session_payload("task-a")

    assert payload is not None
    assert payload["schemaVersion"] == 1
    assert payload["protocol"] == CAI_OWNED_TRANSPORT_PROTOCOL
    assert payload["sessionId"] == expected_session_id
    assert payload["instanceId"] == "instance-a"
    assert payload["modelId"] == "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    assert payload["taskId"] == "task-a"
    assert payload["sourceNodeId"] == "node-a"
    assert payload["participantNodeIds"] == ["node-a", "node-b"]
    assert payload["executionMode"] == "cai_owned_transport_required"
    assert payload["routePolicy"] == {
        "runtime": "llama.cpp",
        "journalOnly": True,
        "dataPlane": "standard_llama_cpp_rpc",
    }
    assert worker._cai_owned_transport_session_id("task-a") == expected_session_id


def test_cai_owned_transport_journal_hook_is_env_guarded(monkeypatch) -> None:
    runner = object.__new__(Runner)
    runner.bound_instance = _bound_instance(device_rank=0, world_size=2)
    runner.instance = runner.bound_instance.instance
    runner.shard_metadata = runner.bound_instance.bound_shard
    runner.model_id = runner.shard_metadata.model_card.model_id
    calls = []

    def _create_cai_owned_transport_session(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "cai_compute_chain.decentralized_compute.create_cai_owned_transport_session",
        _create_cai_owned_transport_session,
    )

    runner._maybe_create_cai_owned_transport_session("task-a")
    assert calls == []

    monkeypatch.setenv("CAI_OWNED_TRANSPORT_JOURNAL_ENABLED", "1")
    runner._maybe_create_cai_owned_transport_session("task-a")

    assert calls == [
        {
            "session_id": deterministic_cai_owned_transport_session_id(
                "instance-a",
                ["node-a", "node-b"],
                task_id="task-a",
            ),
            "instance_id": "instance-a",
            "participant_node_ids": ["node-a", "node-b"],
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            "task_id": "task-a",
            "source_node_id": "node-a",
            "execution_mode": "cai_owned_transport_required",
            "route_policy": {
                "runtime": "llama.cpp",
                "journalOnly": True,
                "dataPlane": "standard_llama_cpp_rpc",
            },
        }
    ]


def test_cai_owned_transport_generation_dispatches_and_emits_final_output(
    monkeypatch,
) -> None:
    runner = object.__new__(Runner)
    runner.bound_instance = _bound_instance(device_rank=0, world_size=2)
    runner.instance = runner.bound_instance.instance.model_copy(
        update={
            "cai_api_urls_by_node": {
                NodeId("node-a"): ["http://node-a:52415"],
                NodeId("node-b"): [
                    "cai-overlay:http://relay:52415?targetNodeId=node-b"
                ],
            }
        }
    )
    runner.shard_metadata = runner.bound_instance.bound_shard
    runner.model_id = runner.shard_metadata.model_card.model_id
    runner.event_sender = EventCollector()
    calls = {}

    def _dispatch(**kwargs):
        calls["dispatch"] = kwargs
        return {"sessionId": "caiot_generation"}

    def _await(session_id, **kwargs):
        calls["await"] = {"session_id": session_id, **kwargs}
        return {
            "status": "completed",
            "proofVerified": True,
            "finalOutput": {"payload": b"distributed answer"},
        }

    monkeypatch.setattr(
        "cai_compute_chain.decentralized_compute.dispatch_cai_owned_transport_execution_dag",
        _dispatch,
    )
    monkeypatch.setattr(
        "cai_compute_chain.decentralized_compute.await_cai_owned_transport_session_final_result",
        _await,
    )
    monkeypatch.setenv("CAI_OWNED_TRANSPORT_GENERATION_ENABLED", "1")
    monkeypatch.setenv("CAI_OWNED_TRANSPORT_PAYLOAD_COMPRESSION", "gzip")
    monkeypatch.setenv("CAI_OWNED_TRANSPORT_PAYLOAD_CHUNK_SIZE_BYTES", "64")
    monkeypatch.setenv(
        "CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_EXECUTOR_READINESS",
        "1",
    )
    monkeypatch.setenv("CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_RUNTIME_READY", "1")
    monkeypatch.setenv("CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_SHARD_READINESS", "1")
    monkeypatch.setenv("CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_DATA_PLANE_ROUTE", "1")

    handled = runner._run_cai_owned_transport_generation_if_enabled(_task())

    assert handled is True
    assert calls["dispatch"]["requester_node_id"] == "node-a"
    assert calls["dispatch"]["executor_node_ids"] == ["node-a", "node-b"]
    assert calls["dispatch"]["peer_cai_urls_by_node"] == {
        "node-a": ["http://node-a:52415"],
        "node-b": ["cai-overlay:http://relay:52415?targetNodeId=node-b"],
    }
    assert calls["dispatch"]["total_layer_count"] == 8
    assert calls["dispatch"]["route_policy"] == {
        "runtime": "llama.cpp",
        "journalOnly": False,
        "dataPlane": "cai_owned_transport_execution_dag",
        "coordinator": "llama_cpp_runner",
        "avoidSingleTransitBottleneck": True,
        "requireProvenDataPlaneRoute": True,
        "minimumRelayQuorum": 0,
    }
    assert calls["dispatch"]["payload_compression"] == "gzip"
    assert calls["dispatch"]["payload_chunk_size_bytes"] == 64
    assert calls["dispatch"]["llm_runtime_metadata"] == {
        "modelId": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "totalLayerCount": 8,
        "hiddenSize": 1,
        "activationDtype": "f16",
        "tensorEncoding": "ggml-tensor-v1",
        "tokenizerConfigHash": runner._cai_owned_transport_tokenizer_config_hash(),
        "backend": "llama.cpp-patched",
        "backendVersion": "llama.cpp/cai-shard-0.1",
        "metadataSource": "cai.llama_cpp.runner",
        "ggufArchitecture": "qwen2",
        "shardCompatibility": "layer_range_supported",
        "layerRangeSupported": True,
        "layerRangeProbeAbi": "cai-layer-range-v1",
        "layerRangeProbeReport": (
            "docs/reports/qwen2.5-production-binary-conformance-2026-05-11.json"
        ),
        "layerRangeEquivalenceProbeReport": (
            "docs/reports/qwen2.5-layer-range-equivalence-probe-2026-05-11.json"
        ),
        "stateFormat": "ggml-tensor-v1/layer-range-activation-v1",
        "activationStateFormat": "ggml-tensor-v1/layer-range-activation-v1",
        "decodeStateFormat": "ggml-kv-cache-v1/token-step-kv-cache-v1",
    }
    assert calls["dispatch"]["require_executor_readiness"] is True
    assert calls["dispatch"]["require_cai_owned_runtime_ready"] is True
    assert calls["dispatch"]["require_executor_shard_readiness"] is True
    assert calls["dispatch"]["require_data_plane_route"] is True
    assert calls["dispatch"]["require_proven_data_plane_route"] is True
    assert calls["await"]["session_id"] == "caiot_generation"
    assert calls["await"]["requester_node_id"] == "node-a"
    assert len(runner.event_sender.events) == 1
    event = runner.event_sender.events[0]
    assert isinstance(event, ChunkGenerated)
    assert isinstance(event.chunk, TokenChunk)
    assert event.chunk.text == "distributed answer"
    assert event.chunk.finish_reason == "stop"


def test_cai_owned_transport_generation_preflight_defaults_are_safe(
    monkeypatch,
) -> None:
    for key in (
        "CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_EXECUTOR_READINESS",
        "CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_RUNTIME_READY",
        "CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_SHARD_READINESS",
        "CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_DATA_PLANE_ROUTE",
        "CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_PROVEN_DATA_PLANE_ROUTE",
    ):
        monkeypatch.delenv(key, raising=False)

    assert _cai_owned_transport_generation_require_executor_readiness() is True
    assert _cai_owned_transport_generation_require_shard_readiness() is True
    assert _cai_owned_transport_generation_require_data_plane_route() is True
    assert _cai_owned_transport_generation_require_proven_data_plane_route() is True
    assert _cai_owned_transport_generation_require_runtime_ready() is False

    monkeypatch.setenv("CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_DATA_PLANE_ROUTE", "0")
    assert _cai_owned_transport_generation_require_data_plane_route() is False
    assert _cai_owned_transport_generation_require_proven_data_plane_route() is False


def test_cai_owned_transport_llm_runtime_metadata_uses_model_card_and_env(
    monkeypatch,
) -> None:
    runner = object.__new__(Runner)
    runner.bound_instance = _bound_instance(device_rank=0, world_size=2)
    runner.instance = runner.bound_instance.instance
    runner.shard_metadata = runner.bound_instance.bound_shard
    runner.model_id = runner.shard_metadata.model_card.model_id
    monkeypatch.setenv("CAI_LLM_SHARD_ACTIVATION_DTYPE", "bf16")
    monkeypatch.setenv("CAI_LLM_SHARD_TENSOR_ENCODING", "raw-le")
    monkeypatch.setenv("CAI_LLM_SHARD_BACKEND_VERSION", "llama.cpp/cai-prod-test")

    metadata = runner._cai_owned_transport_llm_runtime_metadata()

    assert metadata["modelId"] == "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    assert metadata["totalLayerCount"] == 8
    assert metadata["hiddenSize"] == 1
    assert metadata["activationDtype"] == "bf16"
    assert metadata["tensorEncoding"] == "raw-le"
    assert metadata["backend"] == "llama.cpp-patched"
    assert metadata["backendVersion"] == "llama.cpp/cai-prod-test"
    assert metadata["ggufArchitecture"] == "qwen2"
    assert metadata["shardCompatibility"] == "layer_range_supported"
    assert metadata["layerRangeSupported"] is True
    assert metadata["layerRangeProbeAbi"] == "cai-layer-range-v1"
    assert str(metadata["layerRangeProbeReport"]).endswith(
        "qwen2.5-production-binary-conformance-2026-05-11.json"
    )
    assert str(metadata["layerRangeEquivalenceProbeReport"]).endswith(
        "qwen2.5-layer-range-equivalence-probe-2026-05-11.json"
    )
    assert metadata["stateFormat"] == "ggml-tensor-v1/layer-range-activation-v1"
    assert metadata["activationStateFormat"] == (
        "ggml-tensor-v1/layer-range-activation-v1"
    )
    assert metadata["decodeStateFormat"] == "ggml-kv-cache-v1/token-step-kv-cache-v1"
    assert metadata["tokenizerConfigHash"] == (
        runner._cai_owned_transport_tokenizer_config_hash()
    )


def test_cai_owned_transport_generation_required_reports_error(monkeypatch) -> None:
    runner = object.__new__(Runner)
    runner.bound_instance = _bound_instance(device_rank=0, world_size=2)
    runner.instance = runner.bound_instance.instance
    runner.shard_metadata = runner.bound_instance.bound_shard
    runner.model_id = runner.shard_metadata.model_card.model_id
    runner.event_sender = EventCollector()

    monkeypatch.setenv("CAI_OWNED_TRANSPORT_GENERATION_ENABLED", "1")
    monkeypatch.setenv("CAI_OWNED_TRANSPORT_GENERATION_REQUIRED", "1")

    handled = runner._run_cai_owned_transport_generation_if_enabled(_task())

    assert handled is True
    assert len(runner.event_sender.events) == 1
    event = runner.event_sender.events[0]
    assert isinstance(event, ChunkGenerated)
    assert isinstance(event.chunk, ErrorChunk)
    assert "missing CAI API URLs" in event.chunk.error_message


def test_cai_owned_transport_required_generation_skips_local_server_and_warmup(
    monkeypatch,
) -> None:
    runner = object.__new__(Runner)
    runner.bound_instance = _bound_instance(device_rank=0, world_size=2)
    runner.instance = runner.bound_instance.instance
    runner.shard_metadata = runner.bound_instance.bound_shard
    runner.model_id = runner.shard_metadata.model_card.model_id
    runner.runner_id = RunnerId("runner-a")
    runner.current_status = RunnerIdle()
    runner.event_sender = EventCollector()
    calls: list[str] = []

    def _record_start_server() -> None:
        calls.append("start_server")

    def _record_warmup() -> None:
        calls.append("warmup")

    runner._start_server = _record_start_server  # type: ignore[attr-defined]
    runner._warmup = _record_warmup  # type: ignore[attr-defined]
    monkeypatch.setenv("CAI_OWNED_TRANSPORT_GENERATION_ENABLED", "1")
    monkeypatch.setenv("CAI_OWNED_TRANSPORT_GENERATION_REQUIRED", "1")

    runner.handle_task(
        LoadModel(
            task_id="load-cai-owned",
            instance_id=runner.instance.instance_id,
        )
    )
    assert isinstance(runner.current_status, RunnerLoaded)

    runner.handle_task(
        StartWarmup(
            task_id="warmup-cai-owned",
            instance_id=runner.instance.instance_id,
        )
    )

    assert calls == []
    assert isinstance(runner.current_status, RunnerReady)
    assert _cai_owned_transport_skip_local_llama_server() is True


def test_cai_owned_transport_skip_local_server_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("CAI_OWNED_TRANSPORT_GENERATION_ENABLED", "1")
    monkeypatch.setenv("CAI_OWNED_TRANSPORT_GENERATION_REQUIRED", "1")
    monkeypatch.setenv("CAI_OWNED_TRANSPORT_SKIP_LOCAL_LLAMA_SERVER", "0")

    assert _cai_owned_transport_skip_local_llama_server() is False


def test_cai_owned_transport_offer_submit_uses_instance_api_urls(monkeypatch) -> None:
    runner = object.__new__(Runner)
    runner.bound_instance = _bound_instance(device_rank=0, world_size=2)
    runner.instance = runner.bound_instance.instance.model_copy(
        update={
            "cai_api_urls_by_node": {
                NodeId("node-b"): [
                    "http://node-b-bad:52415",
                    "http://node-b-good:52415",
                ],
            }
        }
    )
    runner.shard_metadata = runner.bound_instance.bound_shard
    runner.model_id = runner.shard_metadata.model_card.model_id
    create_calls = []
    offer_calls = []

    def _create_cai_owned_transport_session(**kwargs):
        create_calls.append(kwargs)

    def _submit_cai_owned_transport_session_offer(url, payload, **_kwargs):
        offer_calls.append((url, payload))
        if "bad" in url:
            raise OSError("bad route")
        return {"status": "created"}

    monkeypatch.setattr(
        "cai_compute_chain.decentralized_compute.create_cai_owned_transport_session",
        _create_cai_owned_transport_session,
    )
    monkeypatch.setattr(
        "cai_compute_chain.decentralized_compute.submit_cai_owned_transport_session_offer",
        _submit_cai_owned_transport_session_offer,
    )
    monkeypatch.setenv("CAI_OWNED_TRANSPORT_JOURNAL_ENABLED", "1")
    monkeypatch.setenv("CAI_OWNED_TRANSPORT_OFFER_SUBMIT_ENABLED", "1")

    runner._maybe_create_cai_owned_transport_session("task-a")

    assert len(create_calls) == 1
    assert len(offer_calls) == 2
    assert offer_calls[0][0] == "http://node-b-bad:52415"
    assert offer_calls[1][0] == "http://node-b-good:52415"
    assert offer_calls[1][1]["sessionId"] == create_calls[0]["session_id"]
    assert offer_calls[1][1]["participantNodeIds"] == ["node-a", "node-b"]


def test_distributed_runner_records_llama_cpp_rpc_protocol_failure(monkeypatch) -> None:
    runner = object.__new__(Runner)
    runner.bound_instance = _bound_instance(device_rank=0, world_size=2)
    runner.instance = runner.bound_instance.instance
    runner.shard_metadata = runner.bound_instance.bound_shard
    runner.relay_tunnel_manager = None

    calls = []

    def _record_llama_cpp_rpc_result(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "cai_compute_chain.route_health.record_llama_cpp_rpc_result",
        _record_llama_cpp_rpc_result,
    )

    runner._record_remote_rpc_protocol_failure(
        "llama.cpp server exited before becoming ready\n"
        "ggml-rpc.cpp:354: Remote RPC server crashed or returned malformed response"
    )

    assert calls == [
        {
            "source_node_id": "node-a",
            "sink_node_id": "node-b",
            "transit_node_id": None,
            "endpoint_url": "llama-cpp-rpc://10.0.0.2:50052",
            "reachable": False,
            "error": (
                "llama.cpp server exited before becoming ready\n"
                "ggml-rpc.cpp:354: Remote RPC server crashed or returned malformed response"
            ),
        }
    ]


def test_distributed_generation_failure_records_llama_cpp_rpc_protocol_failure(
    monkeypatch,
) -> None:
    runner = object.__new__(Runner)
    runner.bound_instance = _bound_instance(device_rank=0, world_size=2)
    runner.instance = runner.bound_instance.instance
    runner.shard_metadata = runner.bound_instance.bound_shard
    runner.model_id = runner.shard_metadata.model_card.model_id
    runner.event_sender = EventCollector()
    runner.relay_tunnel_manager = None

    def _raise_rpc_failure(_payload):
        raise RuntimeError(
            "ggml-rpc.cpp:354: Remote RPC server crashed or returned malformed response"
        )

    calls = []
    runner._request_chat_completion = _raise_rpc_failure  # type: ignore[attr-defined]

    def _record_llama_cpp_rpc_result(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "cai_compute_chain.route_health.record_llama_cpp_rpc_result",
        _record_llama_cpp_rpc_result,
    )

    runner._run_text_generation(_task())

    assert len(runner.event_sender.events) == 1
    event = runner.event_sender.events[0]
    assert isinstance(event, ChunkGenerated)
    assert isinstance(event.chunk, ErrorChunk)
    assert calls == [
        {
            "source_node_id": "node-a",
            "sink_node_id": "node-b",
            "transit_node_id": None,
            "endpoint_url": "llama-cpp-rpc://10.0.0.2:50052",
            "reachable": False,
            "error": (
                "ggml-rpc.cpp:354: Remote RPC server crashed or returned malformed response"
            ),
        }
    ]


def test_distributed_warmup_failure_records_llama_cpp_rpc_protocol_failure(
    monkeypatch,
) -> None:
    runner = object.__new__(Runner)
    runner.bound_instance = _bound_instance(device_rank=0, world_size=2)
    runner.instance = runner.bound_instance.instance
    runner.shard_metadata = runner.bound_instance.bound_shard
    runner.model_id = runner.shard_metadata.model_card.model_id
    runner.relay_tunnel_manager = None

    def _raise_rpc_failure(*, payload):
        raise RuntimeError("ggml-rpc remote rpc server crashed")

    calls = []
    runner._request_chat_completion = _raise_rpc_failure  # type: ignore[attr-defined]

    def _record_llama_cpp_rpc_result(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "cai_compute_chain.route_health.record_llama_cpp_rpc_result",
        _record_llama_cpp_rpc_result,
    )

    try:
        runner._warmup()
    except RuntimeError:
        pass

    assert calls == [
        {
            "source_node_id": "node-a",
            "sink_node_id": "node-b",
            "transit_node_id": None,
            "endpoint_url": "llama-cpp-rpc://10.0.0.2:50052",
            "reachable": False,
            "error": "ggml-rpc remote rpc server crashed",
        }
    ]


def test_distributed_runner_records_llama_cpp_rpc_protocol_success(monkeypatch) -> None:
    runner = object.__new__(Runner)
    runner.bound_instance = _bound_instance(device_rank=0, world_size=2)
    runner.instance = runner.bound_instance.instance
    runner.shard_metadata = runner.bound_instance.bound_shard
    runner.relay_tunnel_manager = None

    calls = []

    def _record_llama_cpp_rpc_result(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "cai_compute_chain.route_health.record_llama_cpp_rpc_result",
        _record_llama_cpp_rpc_result,
    )

    runner._record_remote_rpc_protocol_success()

    assert calls == [
        {
            "source_node_id": "node-a",
            "sink_node_id": "node-b",
            "transit_node_id": None,
            "endpoint_url": "llama-cpp-rpc://10.0.0.2:50052",
            "reachable": True,
            "error": None,
        }
    ]


def test_distributed_runner_records_selected_relay_tunnel_direct_mode(
    monkeypatch,
) -> None:
    runner = object.__new__(Runner)
    runner.bound_instance = _bound_instance(device_rank=0, world_size=2)
    relay_route = LlamaCppRelayRoute(
        source_node_id=NodeId("node-a"),
        transit_node_id=NodeId("node-relay"),
        sink_node_id=NodeId("node-b"),
        relay_api_host="203.0.113.10",
        relay_api_port=52415,
        target_host="203.0.113.11",
        target_port=52435,
        source_segment_type="overlay",
        sink_segment_type="overlay",
    )
    runner.instance = runner.bound_instance.instance.model_copy(
        update={
            "relay_routes_by_node": {
                NodeId("node-a"): [relay_route],
            }
        }
    )
    runner.shard_metadata = runner.bound_instance.bound_shard
    runner.relay_tunnel_manager = RelayTunnelStub(
        NodeId("node-b"),
        Host(ip="127.0.0.1", port=60123),
        selected_mode="direct",
        selected_route=relay_route,
    )
    calls = []

    def _record_llama_cpp_rpc_result(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "cai_compute_chain.route_health.record_llama_cpp_rpc_result",
        _record_llama_cpp_rpc_result,
    )

    runner._record_remote_rpc_protocol_success()

    assert calls == [
        {
            "source_node_id": "node-a",
            "sink_node_id": "node-b",
            "transit_node_id": None,
            "endpoint_url": "llama-cpp-rpc://203.0.113.11:52435",
            "reachable": True,
            "error": None,
        }
    ]


def test_remote_rpc_servers_use_local_relay_tunnel_for_routed_sink() -> None:
    runner = object.__new__(Runner)
    runner.bound_instance = _bound_instance(device_rank=0, world_size=2)
    runner.instance = runner.bound_instance.instance.model_copy(
        update={
            "hosts_by_node": {
                NodeId("node-a"): [
                    Host(ip="0.0.0.0", port=50052),
                    Host(ip="198.51.100.1", port=0),
                ],
                NodeId("node-b"): [
                    Host(ip="0.0.0.0", port=50052),
                    Host(ip="0.0.0.0", port=50052),
                ],
            },
            "relay_routes_by_node": {
                NodeId("node-a"): [
                    LlamaCppRelayRoute(
                        source_node_id=NodeId("node-a"),
                        transit_node_id=NodeId("node-relay"),
                        sink_node_id=NodeId("node-b"),
                        relay_api_host="203.0.113.10",
                        relay_api_port=52415,
                        target_host="203.0.113.11",
                        target_port=52435,
                        source_segment_type="overlay",
                        sink_segment_type="overlay",
                    )
                ]
            },
        }
    )
    runner.shard_metadata = runner.bound_instance.bound_shard
    runner.relay_tunnel_manager = RelayTunnelStub(
        NodeId("node-b"),
        Host(ip="127.0.0.1", port=60123),
    )

    endpoints = runner._remote_rpc_servers()

    assert endpoints == ["127.0.0.1:60123"]


def test_wait_for_remote_rpc_servers_uses_direct_llama_cpp_rpc_hello(monkeypatch) -> None:
    sent_payloads: list[bytes] = []

    class Socket:
        def __init__(self) -> None:
            response = bytes([1, 2, 3, 0]) + bytes(24)
            self.chunks = [len(response).to_bytes(8, "little"), response]

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def settimeout(self, _timeout: float) -> None:
            return None

        def sendall(self, payload: bytes) -> None:
            sent_payloads.append(payload)

        def recv(self, _size: int) -> bytes:
            return self.chunks.pop(0)

    def fake_create_connection(address, *, timeout: float):
        assert address == ("10.0.0.2", 50052)
        assert timeout == 1
        return Socket()

    runner = object.__new__(Runner)
    runner.bound_instance = _bound_instance(device_rank=0, world_size=2)
    runner.instance = runner.bound_instance.instance
    runner.shard_metadata = runner.bound_instance.bound_shard
    runner.relay_tunnel_manager = None
    monkeypatch.setattr(
        "cai.worker.runner.llama_cpp.runner.socket.create_connection",
        fake_create_connection,
    )

    runner._wait_for_remote_rpc_servers()

    assert sent_payloads == [_llama_cpp_rpc_hello_payload()]


def test_wait_for_remote_rpc_servers_uses_relay_llama_cpp_rpc_probe(
    monkeypatch,
) -> None:
    runner = object.__new__(Runner)
    runner.bound_instance = _bound_instance(device_rank=0, world_size=2)
    runner.instance = runner.bound_instance.instance
    runner.shard_metadata = runner.bound_instance.bound_shard
    runner.relay_tunnel_manager = RelayTunnelStub(
        NodeId("node-b"),
        Host(ip="127.0.0.1", port=60123),
    )
    monkeypatch.setattr(
        "cai.worker.runner.llama_cpp.runner.socket.create_connection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("relay route should not use direct socket probe")
        ),
    )

    runner._wait_for_remote_rpc_servers()

    assert runner.relay_tunnel_manager.llama_cpp_rpc_probe_calls == [NodeId("node-b")]


def test_relay_probe_uses_non_stream_probe_endpoint() -> None:
    route = LlamaCppRelayRoute(
        source_node_id=NodeId("node-a"),
        transit_node_id=NodeId("node-relay"),
        sink_node_id=NodeId("node-b"),
        relay_api_host="203.0.113.10",
        relay_api_port=52415,
        target_host="203.0.113.11",
        target_port=52435,
        source_segment_type="overlay",
        sink_segment_type="overlay",
    )
    manager = LlamaCppRelayTunnelManager([route])

    url = manager._relay_probe_url(route)

    assert url.startswith("http://203.0.113.10:52415/v1/cai/relay/rpc/probe?")
    assert "/relay/rpc/ws" not in url
    assert "sink_node_id=node-b" in url


def test_relay_probe_url_can_request_llama_cpp_rpc_protocol() -> None:
    route = LlamaCppRelayRoute(
        source_node_id=NodeId("node-a"),
        transit_node_id=NodeId("node-relay"),
        sink_node_id=NodeId("node-b"),
        relay_api_host="203.0.113.10",
        relay_api_port=52415,
        target_host="203.0.113.11",
        target_port=52435,
        source_segment_type="overlay",
        sink_segment_type="overlay",
    )
    manager = LlamaCppRelayTunnelManager([route])

    url = manager._relay_probe_url(route, protocol="llama_cpp_rpc")

    assert "protocol=llama_cpp_rpc" in url


def test_relay_tunnel_manager_keeps_fallback_routes_per_sink() -> None:
    primary_route = LlamaCppRelayRoute(
        source_node_id=NodeId("node-a"),
        transit_node_id=NodeId("node-relay-a"),
        sink_node_id=NodeId("node-b"),
        relay_api_host="203.0.113.10",
        relay_api_port=52415,
        target_host="203.0.113.11",
        target_port=52435,
        source_segment_type="overlay",
        sink_segment_type="overlay",
    )
    fallback_route = primary_route.model_copy(
        update={
            "transit_node_id": NodeId("node-relay-b"),
            "relay_api_host": "203.0.113.12",
        }
    )

    manager = LlamaCppRelayTunnelManager([primary_route, fallback_route])

    assert [
        route.transit_node_id
        for route in manager._routes_by_sink[NodeId("node-b")]
    ] == [NodeId("node-relay-a"), NodeId("node-relay-b")]


def test_relay_tunnel_uses_direct_target_when_reachable() -> None:
    async def run() -> None:
        async def handle_remote(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            payload = await reader.readexactly(4)
            writer.write(payload + b"-ok")
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        remote_server = await asyncio.start_server(handle_remote, "127.0.0.1", 0)
        remote_port = int((remote_server.sockets or [])[0].getsockname()[1])
        route = LlamaCppRelayRoute(
            source_node_id=NodeId("node-a"),
            transit_node_id=NodeId("node-relay"),
            sink_node_id=NodeId("node-b"),
            relay_api_host="203.0.113.10",
            relay_api_port=52415,
            target_host="127.0.0.1",
            target_port=remote_port,
            source_segment_type="overlay",
            sink_segment_type="overlay",
        )
        manager = LlamaCppRelayTunnelManager([route])
        writer: asyncio.StreamWriter | None = None
        try:
            manager.start()
            endpoint = manager.local_endpoint_for_sink(NodeId("node-b"))
            assert endpoint is not None
            reader, writer = await asyncio.open_connection(endpoint.ip, endpoint.port)
            writer.write(b"ping")
            await writer.drain()

            assert await asyncio.wait_for(reader.readexactly(7), timeout=3) == b"ping-ok"
        finally:
            if writer is not None:
                writer.close()
                with suppress(Exception):
                    await writer.wait_closed()
            manager.stop()
            remote_server.close()
            await remote_server.wait_closed()

    asyncio.run(run())


def test_relay_tunnel_llama_cpp_rpc_probe_uses_actual_local_path() -> None:
    async def run() -> None:
        seen_payloads: list[bytes] = []

        async def handle_remote(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            seen_payloads.append(
                await reader.readexactly(len(_llama_cpp_rpc_hello_payload()))
            )
            response = bytes([1, 4, 2, 0]) + bytes(24)
            writer.write(len(response).to_bytes(8, "little") + response)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        remote_server = await asyncio.start_server(handle_remote, "127.0.0.1", 0)
        remote_port = int((remote_server.sockets or [])[0].getsockname()[1])
        route = LlamaCppRelayRoute(
            source_node_id=NodeId("node-a"),
            transit_node_id=NodeId("node-relay"),
            sink_node_id=NodeId("node-b"),
            relay_api_host="203.0.113.10",
            relay_api_port=52415,
            target_host="127.0.0.1",
            target_port=remote_port,
            source_segment_type="overlay",
            sink_segment_type="overlay",
        )
        manager = LlamaCppRelayTunnelManager([route])
        try:
            manager.start()
            version = await asyncio.to_thread(
                lambda: manager.probe_llama_cpp_rpc_route(
                    NodeId("node-b"),
                    timeout=3,
                )
            )

            assert version == "1.4.2"
            assert seen_payloads == [
                _llama_cpp_rpc_hello_payload(),
                _llama_cpp_rpc_hello_payload(),
            ]
        finally:
            manager.stop()
            remote_server.close()
            await remote_server.wait_closed()

    asyncio.run(run())


def test_relay_llama_cpp_rpc_hello_stream_parses_version() -> None:
    class Reader:
        def __init__(self) -> None:
            response = bytes([2, 1, 5, 0]) + bytes(24)
            self.chunks = [len(response).to_bytes(8, "little"), response]

        async def readexactly(self, _size: int) -> bytes:
            return self.chunks.pop(0)

    class Writer:
        def __init__(self) -> None:
            self.payloads: list[bytes] = []

        def write(self, payload: bytes) -> None:
            self.payloads.append(payload)

        async def drain(self) -> None:
            return None

    writer = Writer()
    version = asyncio.run(
        _probe_llama_cpp_rpc_hello_stream(
            Reader(),
            writer,  # type: ignore[arg-type]
            timeout=1,
        )
    )

    assert version == "2.1.5"
    assert writer.payloads == [_llama_cpp_rpc_hello_payload()]


def test_relay_tunnel_selects_relay_when_direct_rpc_protocol_fails() -> None:
    async def run() -> None:
        route = LlamaCppRelayRoute(
            source_node_id=NodeId("node-a"),
            transit_node_id=NodeId("node-relay"),
            sink_node_id=NodeId("node-b"),
            relay_api_host="203.0.113.10",
            relay_api_port=52415,
            target_host="127.0.0.1",
            target_port=52435,
            source_segment_type="overlay",
            sink_segment_type="overlay",
        )
        manager = LlamaCppRelayTunnelManager([route])

        async def fake_connect_direct_stream(_route, *, timeout: float):
            raise RuntimeError("malformed response")

        async def fake_probe_relay_route(_route, *, timeout: float) -> str:
            return "1.7.0"

        manager._connect_direct_stream = fake_connect_direct_stream  # type: ignore[method-assign]
        manager._async_probe_relay_llama_cpp_rpc_route = fake_probe_relay_route  # type: ignore[method-assign]

        version = await manager._async_select_llama_cpp_rpc_route(
            NodeId("node-b"),
            timeout=1,
        )

        assert version == "1.7.0"
        assert manager._preferred_route_by_sink[NodeId("node-b")] == ("relay", route)

    asyncio.run(run())


def test_relay_pipe_local_eof_sends_half_close_marker() -> None:
    class Reader:
        async def read(self, _size: int) -> bytes:
            return b""

    class WebSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []
            self.closed = False

        async def send_str(self, payload: str) -> None:
            self.sent.append(payload)

        async def close(self) -> None:
            self.closed = True

    manager = object.__new__(LlamaCppRelayTunnelManager)
    websocket = WebSocket()

    asyncio.run(manager._pipe_local_to_websocket(Reader(), websocket))

    assert websocket.sent == [_RELAY_EOF_MESSAGE]
    assert websocket.closed is False


def test_relay_pipe_uses_configured_stream_chunk_size() -> None:
    class Reader:
        def __init__(self) -> None:
            self.sizes: list[int] = []
            self._chunks = [b"abc", b""]

        async def read(self, size: int) -> bytes:
            self.sizes.append(size)
            return self._chunks.pop(0)

    class WebSocket:
        def __init__(self) -> None:
            self.sent_bytes: list[bytes] = []
            self.sent_text: list[str] = []

        async def send_bytes(self, payload: bytes) -> None:
            self.sent_bytes.append(payload)

        async def send_str(self, payload: str) -> None:
            self.sent_text.append(payload)

    manager = object.__new__(LlamaCppRelayTunnelManager)
    reader = Reader()
    websocket = WebSocket()

    asyncio.run(manager._pipe_local_to_websocket(reader, websocket))

    assert reader.sizes == [
        DEFAULT_RELAY_STREAM_CHUNK_SIZE,
        DEFAULT_RELAY_STREAM_CHUNK_SIZE,
    ]
    assert websocket.sent_bytes == [b"abc"]
    assert websocket.sent_text == [_RELAY_EOF_MESSAGE]


def test_relay_ws_connect_options_disable_message_limit_and_compression() -> None:
    assert _relay_ws_connect_kwargs() == {
        "heartbeat": 30,
        "max_msg_size": 0,
        "compress": 0,
    }


def test_relay_pipe_websocket_eof_half_closes_local_writer() -> None:
    class WebSocket:
        async def receive(self):
            return type(
                "Message",
                (),
                {"type": WSMsgType.TEXT, "data": _RELAY_EOF_MESSAGE},
            )()

    class Writer:
        def __init__(self) -> None:
            self.eof = False
            self.closed = False

        def can_write_eof(self) -> bool:
            return True

        def write_eof(self) -> None:
            self.eof = True

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    manager = object.__new__(LlamaCppRelayTunnelManager)
    writer = Writer()

    asyncio.run(manager._pipe_websocket_to_local(WebSocket(), writer))

    assert writer.eof is True
    assert writer.closed is False


def test_reverse_relay_confirms_target_after_local_socket_opens(monkeypatch) -> None:
    class Message:
        def __init__(self, message_type, data=None) -> None:
            self.type = message_type
            self.data = data

    class WebSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []
            self.messages = [
                Message(WSMsgType.TEXT, "registered"),
                Message(WSMsgType.TEXT, "connected"),
                Message(WSMsgType.TEXT, _RELAY_EOF_MESSAGE),
            ]

        async def send_str(self, payload: str) -> None:
            self.sent.append(payload)

        async def receive(self):
            return self.messages.pop(0)

    class WebSocketContext:
        def __init__(self, websocket: WebSocket) -> None:
            self.websocket = websocket

        async def __aenter__(self) -> WebSocket:
            return self.websocket

        async def __aexit__(self, *_args) -> None:
            return None

    class Session:
        def __init__(self, websocket: WebSocket) -> None:
            self.websocket = websocket

        def ws_connect(self, *_args, **_kwargs) -> WebSocketContext:
            return WebSocketContext(self.websocket)

    class Reader:
        async def read(self, _size: int) -> bytes:
            return b""

    class Writer:
        def __init__(self) -> None:
            self.eof = False
            self.closed = False

        def can_write_eof(self) -> bool:
            return True

        def write_eof(self) -> None:
            self.eof = True

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            return None

    async def fake_open_connection(host: str, port: int):
        assert host == "127.0.0.1"
        assert port == 52435
        return Reader(), writer

    route = LlamaCppRelayRoute(
        source_node_id=NodeId("node-a"),
        transit_node_id=NodeId("node-relay"),
        sink_node_id=NodeId("node-b"),
        relay_api_host="203.0.113.10",
        relay_api_port=52415,
        target_host="203.0.113.11",
        target_port=52435,
        source_segment_type="overlay",
        sink_segment_type="overlay",
    )
    websocket = WebSocket()
    writer = Writer()
    manager = object.__new__(LlamaCppReverseRelayManager)
    manager._require_session = lambda: Session(websocket)  # type: ignore[attr-defined]
    manager._reverse_relay_ws_url = lambda _route: "ws://relay"  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "cai.worker.runner.llama_cpp.relay_tunnel.asyncio.open_connection",
        fake_open_connection,
    )

    asyncio.run(manager._serve_reverse_route(route))

    assert websocket.sent[0] == _RELAY_TARGET_CONNECTED_MESSAGE
    assert writer.eof is True


def test_rpc_worker_collects_incoming_reverse_relay_routes() -> None:
    runner = object.__new__(Runner)
    runner.bound_instance = _bound_instance(device_rank=1, world_size=2)
    incoming_route = LlamaCppRelayRoute(
        source_node_id=NodeId("node-a"),
        transit_node_id=NodeId("node-relay"),
        sink_node_id=NodeId("node-b"),
        relay_api_host="203.0.113.10",
        relay_api_port=52415,
        target_host="203.0.113.11",
        target_port=52435,
        source_segment_type="overlay",
        sink_segment_type="overlay",
    )
    outbound_route = LlamaCppRelayRoute(
        source_node_id=NodeId("node-b"),
        transit_node_id=NodeId("node-relay"),
        sink_node_id=NodeId("node-a"),
        relay_api_host="203.0.113.10",
        relay_api_port=52415,
        target_host="203.0.113.12",
        target_port=52435,
        source_segment_type="overlay",
        sink_segment_type="overlay",
    )
    runner.instance = runner.bound_instance.instance.model_copy(
        update={
            "relay_routes_by_node": {
                NodeId("node-a"): [incoming_route],
                NodeId("node-b"): [outbound_route],
            }
        }
    )

    routes = runner._instance_incoming_relay_routes()

    assert routes == [incoming_route]


def test_rpc_worker_starts_reverse_relay_manager_for_incoming_routes(monkeypatch) -> None:
    started: dict[str, object] = {}

    class ReverseRelayStub:
        def __init__(self, routes) -> None:
            started["routes"] = list(routes)

        def start(self) -> None:
            started["started"] = True

    runner = object.__new__(Runner)
    runner.bound_instance = _bound_instance(device_rank=1, world_size=2)
    incoming_route = LlamaCppRelayRoute(
        source_node_id=NodeId("node-a"),
        transit_node_id=NodeId("node-relay"),
        sink_node_id=NodeId("node-b"),
        relay_api_host="203.0.113.10",
        relay_api_port=52415,
        target_host="203.0.113.11",
        target_port=52435,
        source_segment_type="overlay",
        sink_segment_type="overlay",
    )
    runner.instance = runner.bound_instance.instance.model_copy(
        update={"relay_routes_by_node": {NodeId("node-a"): [incoming_route]}}
    )
    runner.reverse_relay_manager = None
    monkeypatch.setattr(
        "cai.worker.runner.llama_cpp.runner.LlamaCppReverseRelayManager",
        ReverseRelayStub,
    )

    runner._start_reverse_relay_tunnels()

    assert started["routes"] == [incoming_route]
    assert started["started"] is True
    assert runner.reverse_relay_manager is not None


def test_rpc_bind_port_prefers_bound_host_port() -> None:
    runner = object.__new__(Runner)
    runner.bound_instance = _bound_instance(device_rank=0, world_size=2)
    runner.instance = runner.bound_instance.instance.model_copy(
        update={
            "hosts_by_node": {
                NodeId("node-a"): [
                    Host(ip="0.0.0.0", port=52435),
                    Host(ip="10.0.0.2", port=50052),
                ],
                NodeId("node-b"): [
                    Host(ip="10.0.0.1", port=52435),
                    Host(ip="0.0.0.0", port=50052),
                ],
            }
        }
    )
    runner.shard_metadata = runner.bound_instance.bound_shard

    assert runner._rpc_bind_port() == 52435


def test_build_server_args_for_distributed_rpc_uses_explicit_split_settings() -> None:
    runner = object.__new__(Runner)
    runner.bound_instance = _bound_instance(device_rank=0, world_size=2)
    runner.instance = runner.bound_instance.instance
    runner.shard_metadata = runner.bound_instance.bound_shard

    args = runner._build_server_args(
        model_path="C:/models/test.gguf",  # type: ignore[arg-type]
        server_binary="C:/llama-server.exe",  # type: ignore[arg-type]
        server_port=62600,
    )

    assert "--rpc" in args
    assert "10.0.0.2:50052" in args
    assert "--fit" in args
    assert args[args.index("--fit") + 1] == "off"
    assert "--split-mode" in args
    assert args[args.index("--split-mode") + 1] == "layer"
    assert "--no-cache-prompt" in args
    assert "--cache-ram" in args
    assert args[args.index("--cache-ram") + 1] == "0"
    assert "--device" in args
    assert args[args.index("--device") + 1] == "CUDA0,RPC0"
    assert "--tensor-split" in args
    assert args[args.index("--tensor-split") + 1] == "4,4"


def test_build_server_args_caps_default_context_size_for_runtime() -> None:
    runner = object.__new__(Runner)
    runner.bound_instance = _bound_instance(device_rank=0, world_size=1)
    runner.instance = runner.bound_instance.instance
    runner.shard_metadata = runner.bound_instance.bound_shard

    args = runner._build_server_args(
        model_path="C:/models/test.gguf",  # type: ignore[arg-type]
        server_binary="C:/llama-server.exe",  # type: ignore[arg-type]
        server_port=62600,
    )

    assert "-c" in args
    assert args[args.index("-c") + 1] == "4096"


def test_build_rpc_server_args_disables_cache_by_default(monkeypatch) -> None:
    runner = object.__new__(Runner)

    monkeypatch.delenv("CAI_LLAMA_CPP_RPC_CACHE", raising=False)

    args = runner._build_rpc_server_args(
        rpc_server_binary="C:/rpc-server.exe",  # type: ignore[arg-type]
        rpc_port=50052,
    )

    assert "--cache" not in args


def test_build_rpc_server_args_allows_cache_override(monkeypatch) -> None:
    runner = object.__new__(Runner)

    monkeypatch.setenv("CAI_LLAMA_CPP_RPC_CACHE", "true")

    args = runner._build_rpc_server_args(
        rpc_server_binary="C:/rpc-server.exe",  # type: ignore[arg-type]
        rpc_port=50052,
    )

    assert "--cache" in args


def test_find_llama_server_binary_uses_shared_runtime_payload_layout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    binary_name = "llama-server.exe" if os.name == "nt" else "llama-server"
    runtime_binary = repo_root / "data" / "runtime" / "llama.cpp" / binary_name
    runtime_binary.parent.mkdir(parents=True, exist_ok=True)
    runtime_binary.write_bytes(b"server")
    monkeypatch.delenv("CAI_LLAMA_CPP_SERVER", raising=False)
    monkeypatch.setattr(
        "cai.worker.runner.llama_cpp.runner._repo_root",
        lambda: repo_root,
    )

    resolved = _find_llama_server_binary()

    assert resolved == runtime_binary.resolve()


def test_find_llama_rpc_server_binary_prefers_explicit_env_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    binary_name = "rpc-server.exe" if os.name == "nt" else "rpc-server"
    explicit_binary = tmp_path / binary_name
    explicit_binary.write_bytes(b"rpc")
    monkeypatch.setenv("CAI_LLAMA_CPP_RPC_SERVER", str(explicit_binary))

    resolved = _find_llama_rpc_server_binary()

    assert resolved == explicit_binary.resolve()


def test_build_server_args_allows_distributed_prompt_cache_override(
    monkeypatch,
) -> None:
    runner = object.__new__(Runner)
    runner.bound_instance = _bound_instance(device_rank=0, world_size=2)
    runner.instance = runner.bound_instance.instance
    runner.shard_metadata = runner.bound_instance.bound_shard

    monkeypatch.setenv("CAI_LLAMA_CPP_DISTRIBUTED_PROMPT_CACHE", "true")

    args = runner._build_server_args(
        model_path="C:/models/test.gguf",  # type: ignore[arg-type]
        server_binary="C:/llama-server.exe",  # type: ignore[arg-type]
        server_port=62600,
    )

    assert "--no-cache-prompt" not in args
    assert "--cache-ram" not in args


def test_apply_qwen3_message_directives_injects_no_think_for_user_message() -> None:
    task = _qwen3_task(enable_thinking=False)
    messages = [{"role": "user", "content": "2+2=?"}]

    updated = _apply_qwen3_message_directives(
        task.task_params.model, task.task_params, messages
    )

    assert updated[0]["content"] == "/no_think\n2+2=?"


def test_apply_qwen3_message_directives_injects_think_for_user_message() -> None:
    task = _qwen3_task(enable_thinking=True)
    messages = [{"role": "user", "content": "2+2=?"}]

    updated = _apply_qwen3_message_directives(
        task.task_params.model, task.task_params, messages
    )

    assert updated[0]["content"] == "/think\n2+2=?"


def test_run_text_generation_forwards_extended_sampling_and_qwen3_defaults() -> None:
    runner = object.__new__(Runner)
    runner.model_id = "Qwen/Qwen3-0.6B-GGUF"
    runner.event_sender = EventCollector()
    runner._is_rpc_worker = lambda: False  # type: ignore[attr-defined]
    seen_payload: dict[str, object] = {}

    def _capture(payload):
        seen_payload.update(payload)
        return {
            "choices": [{"finish_reason": "stop", "message": {"content": "4"}}],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "total_tokens": 4,
            },
        }

    runner._request_chat_completion = _capture  # type: ignore[attr-defined]

    runner._run_text_generation(
        _qwen3_task(
            enable_thinking=False,
            top_k=21,
            min_p=0.12,
            repetition_penalty=1.07,
            repetition_context_size=48,
        )
    )

    assert seen_payload["messages"] == [{"role": "user", "content": "/no_think\n2+2=?"}]
    assert seen_payload["temperature"] == 0.7
    assert seen_payload["top_p"] == 0.8
    assert seen_payload["top_k"] == 21
    assert seen_payload["min_p"] == 0.12
    assert seen_payload["repeat_penalty"] == 1.07
    assert seen_payload["repeat_last_n"] == 48


def test_windows_subprocess_flags_detach_console_processes() -> None:
    expected = 0
    if os.name == "nt":
        expected = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
            subprocess, "DETACHED_PROCESS", 0x00000008
        )

    assert _windows_subprocess_flags() == expected


def test_distributed_coordinator_uses_extended_ready_timeout() -> None:
    runner = object.__new__(Runner)
    runner.bound_instance = _bound_instance(device_rank=0, world_size=2)
    runner.instance = runner.bound_instance.instance
    runner.shard_metadata = runner.bound_instance.bound_shard

    assert runner._server_ready_timeout_seconds() == max(
        DEFAULT_READY_TIMEOUT_SECONDS,
        DEFAULT_DISTRIBUTED_READY_TIMEOUT_SECONDS,
    )

