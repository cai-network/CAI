# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import base64
import asyncio
import copy
import contextlib
import hmac
import hashlib
import json
import os
import random
import struct
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn, cast
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from uuid import uuid4

import anyio
from anyio import BrokenResourceError, ClosedResourceError
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from hypercorn.asyncio import serve  # pyright: ignore[reportUnknownVariableType]
from hypercorn.config import Config
from hypercorn.typing import ASGIFramework
from loguru import logger

from cai_compute_chain.gguf_shard_policy import gguf_shard_compatibility
from cai_compute_chain.model import curated_model_registry

from cai.api.adapters.chat_completions import (
    chat_request_to_text_generation,
    collect_chat_response,
    collect_chat_response_payload,
    generate_chat_stream,
)
from cai.api.adapters.claude import (
    claude_request_to_text_generation,
    collect_claude_response,
    generate_claude_stream,
)
from cai.api.adapters.ollama import (
    collect_ollama_chat_response,
    collect_ollama_generate_response,
    generate_ollama_chat_stream,
    generate_ollama_generate_stream,
    ollama_generate_request_to_text_generation,
    ollama_request_to_text_generation,
)
from cai.api.adapters.responses import (
    collect_responses_response,
    generate_responses_stream,
    responses_request_to_text_generation,
)
from cai.api.audit import safe_audit_event
from cai.api.cai_bridge import load_cai_summary, make_cai_service
from cai.api.cai_transport_errors import build_cai_transport_error_detail
from cai.api.dashboard_state import build_dashboard_state
from cai.api.endpoint_policy import EndpointAccess, lookup_endpoint_policy
from cai.api.keepalive import with_sse_keepalive
from cai.api.node_capability_adapter import (
    capability_record_node_identity as _capability_record_node_identity,
    capability_record_node_memory as _capability_record_node_memory,
    capability_record_route_peers as _capability_record_route_peers,
    worker_identity_state as _worker_identity_state,
)
from cai.api.peer_http import (
    bootstrap_api_base_url_for_node as _bootstrap_api_base_url_for_node,
    cai_summary_urls_by_node_id as _cai_summary_urls_by_node_id,
)
from cai.api.rate_limit import InMemoryFixedWindowRateLimiter
from cai.api.text_generation_failures import (
    runner_failure_message_for_model,
    text_generation_failure_detail,
)
from cai.routing.cai_owned_transport_message import CaiOwnedTransportOverlayMessage
from cai.api.types import (
    AddCustomModelParams,
    AdvancedImageParams,
    BenchChatCompletionRequest,
    BenchChatCompletionResponse,
    BenchImageGenerationResponse,
    BenchImageGenerationTaskParams,
    CancelCommandResponse,
    CancelDownloadParams,
    CancelDownloadResponse,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CreateInstanceParams,
    CreateInstanceResponse,
    DeleteDownloadResponse,
    DeleteInstanceResponse,
    DeleteTracesRequest,
    DeleteTracesResponse,
    ErrorInfo,
    ErrorResponse,
    FinishReason,
    GenerationStats,
    HuggingFaceSearchResult,
    ImageData,
    ImageEditsTaskParams,
    ImageGenerationResponse,
    ImageGenerationStats,
    ImageGenerationTaskParams,
    ImageListItem,
    ImageListResponse,
    ImageSize,
    ModelList,
    ModelListModel,
    PlaceInstanceParams,
    PlacementPreview,
    PlacementPreviewResponse,
    StartDownloadParams,
    StartDownloadResponse,
    StreamingChoiceResponse,
    ToolCall,
    TraceCategoryStats,
    TraceEventResponse,
    TraceListItem,
    TraceListResponse,
    TraceRankStats,
    TraceResponse,
    TraceStatsResponse,
    normalize_image_size,
)
from cai.api.types.claude_api import (
    ClaudeMessagesRequest,
    ClaudeMessagesResponse,
)
from cai.api.types.ollama_api import (
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OllamaModelDetails,
    OllamaModelTag,
    OllamaPsModel,
    OllamaPsResponse,
    OllamaShowRequest,
    OllamaShowResponse,
    OllamaTagsResponse,
)
from cai.api.types.openai_responses import (
    ResponsesRequest,
    ResponsesResponse,
)
from cai.master.image_store import ImageStore
from cai.master.placement import place_instance as get_instance_placements
from cai.shared.apply import apply
from cai.shared.constants import (
    DASHBOARD_DIR,
    CAI_CACHE_HOME,
    CAI_EVENT_LOG_DIR,
    CAI_IMAGE_CACHE_DIR,
    CAI_MAX_CHUNK_SIZE,
    CAI_TRACING_CACHE_DIR,
)
from cai.shared.network_model_policy import (
    enforce_private_network_model_request,
    get_private_network_model_policy,
    is_private_network_model,
    private_network_model_effective_min_nodes,
    validate_private_network_instance,
)
from cai.shared.election import ElectionMessage
from cai.shared.logging import InterceptLogger
from cai.shared.models.model_cards import (
    InferenceBackend,
    ModelCard,
    ModelId,
    add_to_card_cache,
    derive_custom_model_id_from_local_path,
    get_card,
    get_model_cards,
    set_custom_model_local_path,
)
from cai.shared.tracing import TraceEvent, compute_stats, export_trace, load_trace_file
from cai.shared.types.chunks import (
    ErrorChunk,
    ImageChunk,
    InputImageChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from cai.shared.types.commands import (
    AddCustomModelCard,
    CancelDownload,
    Command,
    CreateInstance,
    DeleteCustomModelCard,
    DeleteDownload,
    DeleteInstance,
    DownloadCommand,
    ForwarderCommand,
    ForwarderDownloadCommand,
    ImageEdits,
    ImageGeneration,
    PlaceInstance,
    SendInputChunk,
    StartDownload,
    TaskCancelled,
    TaskFinished,
    TextGeneration,
)
from cai.shared.types.common import CommandId, Id, NodeId, SystemId
from cai.shared.types.events import (
    ChunkGenerated,
    Event,
    IndexedEvent,
    InstanceDeleted,
    TracesMerged,
)
from cai.shared.types.memory import Memory
from cai.shared.types.profiling import MemoryUsage, NodeIdentity, NodeNetworkInfo
from cai.shared.types.state import State
from cai.shared.types.tasks import (
    ImageEdits as ImageEditsTask,
)
from cai.shared.types.tasks import (
    ImageGeneration as ImageGenerationTask,
)
from cai.shared.types.tasks import (
    TextGeneration as TextGenerationTask,
)
from cai.shared.types.text_generation import Base64Image, TextGenerationTaskParams
from cai.shared.types.worker.downloads import DownloadCompleted
from cai.shared.types.worker.downloads import DownloadProgress
from cai.shared.types.worker.instances import Instance, InstanceId, InstanceMeta, MlxRingInstance
from cai.shared.types.worker.shards import Sharding
from cai.shared.topology import Topology
from cai.utils.banner import print_startup_banner
from cai.utils.channels import Receiver, Sender, channel
from cai.utils.disk_event_log import DiskEventLog
from cai.utils.power_sampler import PowerSampler
from cai.utils.task_group import TaskGroup

from cai_compute_chain.cai_desktop_app import (
    SUPPORTED_DESKTOP_LANGUAGE_SET,
    resolve_language,
    save_desktop_language,
)
from cai_compute_chain.node_capabilities import (
    NodeCapabilityRecord,
    list_node_capabilities,
    list_verified_worker_node_ids,
    worker_capability_verification_required,
)
from cai_compute_chain.update_channel import (
    build_update_manifest,
    build_update_package,
    record_portable_update_activity,
    update_server_enabled,
)


def _model_card_supported_for_cai_gguf_compute(card: ModelCard) -> bool:
    return (
        card.inference_backend == InferenceBackend.LlamaCpp
        and card.layer_range_supported
        and card.shard_compatibility == "layer_range_supported"
    )


def _unsupported_gguf_model_detail(card: ModelCard) -> str:
    architecture = card.gguf_architecture or card.family or "unknown"
    reason = card.shard_compatibility_reason or (
        "No checked CAI layer-range proof is registered for this GGUF architecture."
    )
    return (
        f"Model '{card.model_id}' is not supported for CAI distributed GGUF "
        f"compute yet. Architecture: {architecture}. {reason}"
    )


def _model_info_is_gguf(model: Any) -> bool:
    model_id = str(getattr(model, "id", "") or "").lower()
    tags = [str(tag).lower() for tag in getattr(model, "tags", []) or []]
    return "gguf" in model_id or any("gguf" in tag for tag in tags)


def _model_info_supported_for_cai_gguf_compute(model: Any) -> bool:
    if not _model_info_is_gguf(model):
        return False
    model_id = str(getattr(model, "id", "") or "").strip()
    tags = [str(tag) for tag in getattr(model, "tags", []) or []]
    compatibility = gguf_shard_compatibility(
        model_id=model_id,
        family=" ".join(tags),
        filename=model_id,
        allow_full_model_local=False,
    )
    return (
        compatibility.layer_range_supported
        and compatibility.shard_compatibility == "layer_range_supported"
    )

_API_EVENT_LOG_DIR = CAI_EVENT_LOG_DIR / "api"
ONBOARDING_COMPLETE_FILE = CAI_CACHE_HOME / "onboarding_complete"
_RELAY_TARGET_CONNECT_TIMEOUT_SECONDS = float(
    os.getenv("CAI_RELAY_TARGET_CONNECT_TIMEOUT_SECONDS", "1") or "1"
)
_REVERSE_RELAY_WAIT_TIMEOUT_SECONDS = float(
    os.getenv("CAI_REVERSE_RELAY_WAIT_TIMEOUT_SECONDS", "4") or "4"
)
_RELAY_STREAM_CHUNK_SIZE = max(
    int(os.getenv("CAI_RELAY_STREAM_CHUNK_SIZE", "16384") or "16384"),
    1024,
)
_RELAY_EOF_MESSAGE = "__cai_relay_eof__"
_RELAY_TARGET_CONNECTED_MESSAGE = "__cai_relay_target_connected__"
_REVERSE_RELAY_TARGET_READY_TIMEOUT_SECONDS = float(
    os.getenv("CAI_REVERSE_RELAY_TARGET_READY_TIMEOUT_SECONDS", "4") or "4"
)
_LLAMA_CPP_RPC_CMD_HELLO = 14
_LLAMA_CPP_RPC_CONN_CAPS_SIZE = 24
_LLAMA_CPP_RPC_HELLO_RESPONSE_SIZE = 4 + _LLAMA_CPP_RPC_CONN_CAPS_SIZE


@dataclass
class _ReverseRelaySession:
    websocket: WebSocket
    done: asyncio.Event = field(default_factory=asyncio.Event)


class NoStoreStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: Mapping[str, Any]) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


def _format_to_content_type(image_format: Literal["png", "jpeg", "webp"] | None) -> str:
    return f"image/{image_format or 'png'}"


def _ensure_seed(params: AdvancedImageParams | None) -> AdvancedImageParams:
    """Ensure advanced params has a seed set for distributed consistency."""
    if params is None:
        return AdvancedImageParams(seed=random.randint(0, 2**32 - 1))
    if params.seed is None:
        return params.model_copy(update={"seed": random.randint(0, 2**32 - 1)})
    return params


def _load_json_url(url: str, *, timeout: int = 5) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _http_error_detail(exc: HTTPError) -> str | None:
    try:
        raw = exc.read()
    except Exception:
        return None
    if not raw:
        return None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return raw.decode("utf-8", errors="replace").strip() or None
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if detail is not None:
            return str(detail)
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message") is not None:
            return str(error["message"])
    return str(payload)


def _api_command_send_timeout_seconds() -> float:
    raw = os.getenv("CAI_API_COMMAND_SEND_TIMEOUT_SECONDS", "30")
    try:
        timeout = float(str(raw).strip() or "30")
    except ValueError:
        timeout = 30.0
    return max(0.1, timeout)


def _raise_cai_transport_http_error(
    exc: BaseException,
    *,
    status_code: int = 400,
    operation: str | None = None,
) -> NoReturn:
    raise HTTPException(
        status_code=status_code,
        detail=build_cai_transport_error_detail(
            exc,
            operation=operation,
            status_code=status_code,
        ),
    ) from exc


def _execution_cai_base_url(local_port: int) -> str:
    configured = str(os.getenv("CAI_EXECUTION_CAI_URL") or "").strip().rstrip("/")
    if configured:
        return configured
    return f"http://127.0.0.1:{local_port}"


def _resolve_cai_repo_root() -> Path:
    configured = str(
        os.getenv("CAI_REPO_ROOT")
        or os.getenv("CAI_RUNTIME_REPO")
        or ""
    ).strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


_CAI_UPDATE_ARCHIVE_STREAM_CHUNK_BYTES = 1024 * 1024


def _stream_cai_update_archive_response(
    archive_path: Path,
    *,
    range_header: str | None,
) -> Response | StreamingResponse:
    resolved = archive_path.expanduser().resolve()
    size = resolved.stat().st_size
    filename = resolved.name.replace('"', "")
    base_headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": f'attachment; filename="{filename}"',
    }

    try:
        requested_range = _parse_cai_update_archive_range(range_header, size=size)
    except ValueError:
        return Response(
            status_code=HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
            headers={**base_headers, "Content-Range": f"bytes */{size}"},
        )

    if requested_range is None:
        start = 0
        end = size - 1
        return StreamingResponse(
            _iter_cai_update_archive_range(resolved, start=start, end=end),
            media_type="application/zip",
            headers={**base_headers, "Content-Length": str(size)},
        )

    start, end = requested_range
    content_length = max(0, end - start + 1)
    return StreamingResponse(
        _iter_cai_update_archive_range(resolved, start=start, end=end),
        status_code=HTTPStatus.PARTIAL_CONTENT,
        media_type="application/zip",
        headers={
            **base_headers,
            "Content-Length": str(content_length),
            "Content-Range": f"bytes {start}-{end}/{size}",
        },
    )


def _parse_cai_update_archive_range(
    range_header: str | None,
    *,
    size: int,
) -> tuple[int, int] | None:
    header = str(range_header or "").strip()
    if not header:
        return None
    if size <= 0:
        raise ValueError("Cannot serve ranges for an empty archive.")
    if not header.lower().startswith("bytes="):
        raise ValueError("Unsupported range unit.")
    spec = header.split("=", 1)[1].strip()
    if not spec or "," in spec:
        raise ValueError("Only a single byte range is supported.")
    start_text, separator, end_text = spec.partition("-")
    if separator != "-":
        raise ValueError("Invalid byte range.")

    if not start_text:
        suffix_length = int(end_text)
        if suffix_length <= 0:
            raise ValueError("Invalid suffix byte range.")
        start = max(size - suffix_length, 0)
        end = size - 1
    else:
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
    if start < 0 or start >= size or end < start:
        raise ValueError("Requested range is not satisfiable.")
    return start, min(end, size - 1)


def _iter_cai_update_archive_range(
    archive_path: Path,
    *,
    start: int,
    end: int,
) -> Iterable[bytes]:
    remaining = max(0, end - start + 1)
    with archive_path.open("rb") as handle:
        handle.seek(start)
        while remaining > 0:
            chunk = handle.read(
                min(_CAI_UPDATE_ARCHIVE_STREAM_CHUNK_BYTES, remaining)
            )
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


class API:
    def __init__(
        self,
        node_id: NodeId,
        *,
        port: int,
        event_receiver: Receiver[IndexedEvent],
        command_sender: Sender[ForwarderCommand],
        download_command_sender: Sender[ForwarderDownloadCommand],
        # This lets us pause the API if an election is running
        election_receiver: Receiver[ElectionMessage],
        cai_owned_transport_message_sender: Sender[CaiOwnedTransportOverlayMessage]
        | None = None,
        cai_owned_transport_message_receiver: Receiver[CaiOwnedTransportOverlayMessage]
        | None = None,
    ) -> None:
        self.state = State()
        self._local_state_cache = State()
        self._event_log = DiskEventLog(_API_EVENT_LOG_DIR)
        self._system_id = SystemId()
        self.command_sender = command_sender
        self.download_command_sender = download_command_sender
        self.event_receiver = event_receiver
        self.election_receiver = election_receiver
        self.cai_owned_transport_message_sender = cai_owned_transport_message_sender
        self.cai_owned_transport_message_receiver = cai_owned_transport_message_receiver
        self.node_id: NodeId = node_id
        self.current_master_node_id: NodeId = node_id
        self.last_completed_election: int = 0
        self.port = port
        self._worker_required_nodes_cache: set[NodeId] | None = None
        self._worker_required_nodes_cache_initialized = False
        self._worker_required_nodes_cache_at = 0.0

        self.paused: bool = False
        self.paused_ev: anyio.Event = anyio.Event()

        self.app = FastAPI()
        self.dashboard_disabled = str(
            os.getenv("CAI_DISABLE_DASHBOARD") or ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.state_local_only = str(
            os.getenv("CAI_STATE_LOCAL_ONLY") or ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.summary_local_only = str(
            os.getenv("CAI_SUMMARY_LOCAL_ONLY") or ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.cai_api_bearer_token = str(os.getenv("CAI_API_BEARER_TOKEN") or "").strip()
        self.update_server_enabled = update_server_enabled()
        self._public_rate_limiter = InMemoryFixedWindowRateLimiter.from_env()
        self._reverse_relay_queues: dict[
            tuple[str, str, int],
            asyncio.Queue[_ReverseRelaySession],
        ] = {}
        self._reverse_relay_queues_lock = asyncio.Lock()

        @self.app.middleware("http")
        async def _log_requests(  # pyright: ignore[reportUnusedFunction]
            request: Request,
            call_next: Callable[[Request], Awaitable[Response]],
        ) -> Response:
            rate_limit_response = self._maybe_rate_limit_public_request(request)
            if rate_limit_response is not None:
                return rate_limit_response
            logger.debug(f"API request: {request.method} {request.url.path}")
            return await call_next(request)

        self._setup_exception_handlers()
        self._setup_cors()
        self._setup_routes()

        if not self.dashboard_disabled:
            self.app.mount(
                "/",
                NoStoreStaticFiles(
                    directory=DASHBOARD_DIR,
                    html=True,
                ),
                name="dashboard",
            )

        self._text_generation_queues: dict[
            CommandId,
            Sender[TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk],
        ] = {}
        self._image_generation_queues: dict[
            CommandId, Sender[ImageChunk | ErrorChunk]
        ] = {}
        self._image_store = ImageStore(CAI_IMAGE_CACHE_DIR)
        self._tg: TaskGroup = TaskGroup()

    def reset(
        self,
        result_clock: int,
        event_receiver: Receiver[IndexedEvent],
        master_node_id: NodeId | None = None,
    ):
        logger.info("Resetting API State")
        self._cache_local_node_state_from(self.state)
        self._event_log.close()
        self._event_log = DiskEventLog(_API_EVENT_LOG_DIR)
        self.state = State()
        self._system_id = SystemId()
        self._text_generation_queues = {}
        self._image_generation_queues = {}
        self._worker_required_nodes_cache = None
        self._worker_required_nodes_cache_initialized = False
        self._worker_required_nodes_cache_at = 0.0
        self._reverse_relay_queues = {}
        self.unpause(result_clock, master_node_id=master_node_id)
        self.state = self._overlay_local_node_state(self.state)
        self.event_receiver.close()
        self.event_receiver = event_receiver
        self._tg.start_soon(self._apply_state)

    def _cache_local_node_state_from(self, state: State) -> None:
        local_node_id = self.node_id
        update: dict[str, Any] = {}
        has_local_data = False

        for field_name in (
            "last_seen",
            "node_identities",
            "node_memory",
            "node_disk",
            "node_system",
            "node_network",
            "node_thunderbolt",
            "node_thunderbolt_bridge",
            "node_rdma_ctl",
        ):
            mapping = getattr(state, field_name, None)
            if not isinstance(mapping, Mapping) or local_node_id not in mapping:
                continue
            cached_mapping = dict(getattr(self._local_state_cache, field_name, {}))
            cached_mapping[local_node_id] = mapping[local_node_id]
            update[field_name] = cached_mapping
            has_local_data = True

        if has_local_data:
            topology = Topology()
            topology.add_node(local_node_id)
            update["topology"] = topology
            self._local_state_cache = self._local_state_cache.model_copy(update=update)

    def _overlay_local_node_state(self, state: State) -> State:
        local_node_id = self.node_id
        update: dict[str, Any] = {}
        has_overlay = False

        for field_name in (
            "last_seen",
            "node_identities",
            "node_memory",
            "node_disk",
            "node_system",
            "node_network",
            "node_thunderbolt",
            "node_thunderbolt_bridge",
            "node_rdma_ctl",
        ):
            cached_mapping = getattr(self._local_state_cache, field_name, None)
            if not isinstance(cached_mapping, Mapping) or local_node_id not in cached_mapping:
                continue
            current_mapping = getattr(state, field_name, None)
            merged_mapping = dict(current_mapping) if isinstance(current_mapping, Mapping) else {}
            merged_mapping[local_node_id] = cached_mapping[local_node_id]
            update[field_name] = merged_mapping
            has_overlay = True

        if not has_overlay:
            return state

        topology = copy.deepcopy(state.topology)
        topology.add_node(local_node_id)
        update["topology"] = topology
        return state.model_copy(update=update)

    def _resolve_execution_cai_url(self) -> str:
        configured = str(os.getenv("CAI_EXECUTION_CAI_URL") or "").strip().rstrip("/")
        if configured:
            return configured

        # Keep the request control plane local. The local API forwards commands
        # through the event router/P2P layer, while advertised master HTTP URLs
        # are only reachability hints and may be unusable behind NAT/proxies.
        return f"http://127.0.0.1:{self.port}"

    def _worker_node_last_seen_is_fresh(self, node_id: str) -> bool:
        normalized_node_id = str(node_id).strip()
        if not normalized_node_id:
            return False
        if normalized_node_id == str(self.node_id).strip():
            return True

        last_seen = getattr(self.state, "last_seen", None)
        if not isinstance(last_seen, Mapping):
            return True

        raw_value = last_seen.get(NodeId(normalized_node_id))
        if raw_value is None:
            raw_value = last_seen.get(normalized_node_id)
        if raw_value is None:
            return True

        if isinstance(raw_value, datetime):
            parsed = raw_value
        elif isinstance(raw_value, str):
            raw_text = raw_value.strip()
            if not raw_text:
                return True
            try:
                parsed = datetime.fromisoformat(raw_text.replace("Z", "+00:00"))
            except ValueError:
                return True
        else:
            return True

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        try:
            stale_seconds = max(
                int(str(os.getenv("CAI_WORKER_NODE_STALE_SECONDS", "90") or "90")),
                0,
            )
        except ValueError:
            stale_seconds = 90
        stale_after = timedelta(seconds=stale_seconds)
        return datetime.now(tz=timezone.utc) - parsed.astimezone(timezone.utc) <= stale_after

    def _resolve_worker_required_nodes(self) -> set[NodeId] | None:
        if (
            getattr(self, "_worker_required_nodes_cache_initialized", False)
            and (
                time.monotonic()
                - float(getattr(self, "_worker_required_nodes_cache_at", 0.0))
            )
            < 5.0
        ):
            cached_nodes = getattr(self, "_worker_required_nodes_cache", None)
            if cached_nodes is None:
                return None
            return set(cached_nodes)

        state = getattr(self, "state", None)
        if state is None:
            return None

        node_identities = state.node_identities
        if not isinstance(node_identities, dict) or not node_identities:
            return None

        summary_urls = _cai_summary_urls_by_node_id(
            node_identities=node_identities,
            local_port=self.port,
            local_node_id=str(self.node_id),
        )
        if not summary_urls:
            return None

        eligible_nodes: set[NodeId] = set()
        checked_nodes = 0
        verified_worker_nodes = self._load_verified_worker_capability_node_ids()
        require_verified_worker_capabilities = (
            worker_capability_verification_required()
        )
        unresolved_summary_urls: dict[str, str] = {}
        local_node_id = str(self.node_id).strip()
        local_worker_enabled = self._load_local_worker_enabled()
        if local_worker_enabled is not None:
            checked_nodes += 1
            if local_worker_enabled and self._load_local_worker_reward_address():
                eligible_nodes.add(self.node_id)
        if verified_worker_nodes:
            active_verified_worker_nodes = {
                str(node_id).strip()
                for node_id in verified_worker_nodes
                if str(node_id).strip() in summary_urls
                and self._worker_node_last_seen_is_fresh(str(node_id))
            }
            checked_nodes += len(active_verified_worker_nodes)
            for node_id in active_verified_worker_nodes:
                if str(node_id).strip() != local_node_id:
                    eligible_nodes.add(NodeId(node_id))
        if require_verified_worker_capabilities:
            self._worker_required_nodes_cache_initialized = True
            self._worker_required_nodes_cache_at = time.monotonic()
            self._worker_required_nodes_cache = set(eligible_nodes)
            return set(eligible_nodes)
        for node_id, identity in node_identities.items():
            normalized_node_id = str(node_id).strip()
            if not normalized_node_id:
                continue
            if normalized_node_id == local_node_id:
                continue
            if not self._worker_node_last_seen_is_fresh(normalized_node_id):
                continue

            worker_enabled, reward_address = _worker_identity_state(identity)
            if worker_enabled is None:
                summary_url = summary_urls.get(normalized_node_id)
                if summary_url:
                    unresolved_summary_urls[normalized_node_id] = summary_url
                continue
            checked_nodes += 1
            if worker_enabled and reward_address:
                eligible_nodes.add(NodeId(node_id))

        for node_id, summary_url in unresolved_summary_urls.items():
            try:
                summary_payload = _load_json_url(summary_url, timeout=5)
            except Exception:
                continue
            checked_nodes += 1
            worker = summary_payload.get("worker") or {}
            reward_address = str(worker.get("worker_reward_address") or "").strip()
            if bool(worker.get("worker_enabled")) and reward_address:
                eligible_nodes.add(NodeId(node_id))

        self._worker_required_nodes_cache_initialized = True
        self._worker_required_nodes_cache_at = time.monotonic()
        if eligible_nodes:
            self._worker_required_nodes_cache = set(eligible_nodes)
            return set(eligible_nodes)
        if checked_nodes > 0:
            self._worker_required_nodes_cache = set()
            return set()
        self._worker_required_nodes_cache = None
        return None

    def _load_verified_worker_capability_node_ids(self) -> set[str]:
        try:
            cai_service = make_cai_service(
                state_url=f"http://127.0.0.1:{self.port}/state",
                cai_url=f"http://127.0.0.1:{self.port}",
                local_node_id=str(self.node_id),
            )
            return set(
                list_verified_worker_node_ids(
                    cai_service.wallet_policy,
                    max_age_seconds=300,
                )
            )
        except Exception:
            return set()

    def _load_verified_worker_capability_records(
        self,
    ) -> dict[NodeId, NodeCapabilityRecord]:
        try:
            cai_service = make_cai_service(
                state_url=f"http://127.0.0.1:{self.port}/state",
                cai_url=f"http://127.0.0.1:{self.port}",
                local_node_id=str(self.node_id),
            )
            verified_node_ids = set(
                list_verified_worker_node_ids(
                    cai_service.wallet_policy,
                    max_age_seconds=300,
                )
            )
            if not verified_node_ids:
                return {}
            records: dict[NodeId, NodeCapabilityRecord] = {}
            for record in list_node_capabilities(cai_service.wallet_policy):
                node_id = str(record.node_id or "").strip()
                if node_id and node_id in verified_node_ids:
                    records[NodeId(node_id)] = record
            return records
        except Exception:
            return {}

    def _load_local_worker_enabled(self) -> bool | None:
        try:
            cai_service = make_cai_service(
                state_url=f"http://127.0.0.1:{self.port}/state",
                cai_url=f"http://127.0.0.1:{self.port}",
                local_node_id=str(self.node_id),
            )
            config = cai_service.modules.node_config.load_or_create_node_config(
                cai_service.wallet_policy
            )
        except Exception:
            return None
        return bool(getattr(config, "worker_enabled", False))

    def _load_local_worker_reward_address(self) -> str | None:
        try:
            cai_service = make_cai_service(
                state_url=f"http://127.0.0.1:{self.port}/state",
                cai_url=f"http://127.0.0.1:{self.port}",
                local_node_id=str(self.node_id),
            )
            reward_address = cai_service.modules.node_config.resolve_worker_reward_address(
                str(self.node_id),
                cai_service.wallet_policy,
            )
            if not reward_address:
                active_wallet = cai_service.modules.wallet.get_active_wallet(
                    cai_service.wallet_policy
                )
                if active_wallet is not None:
                    reward_address = getattr(active_wallet, "address", None)
            if not reward_address:
                return None
            return cai_service.modules.wallet.normalize_address(str(reward_address))
        except Exception:
            return None

    def _load_route_health_records(self) -> list[Any]:
        try:
            cai_service = make_cai_service(
                state_url=f"http://127.0.0.1:{self.port}/state",
                cai_url=f"http://127.0.0.1:{self.port}",
                local_node_id=str(self.node_id),
            )
            return list(
                cai_service.modules.route_health.list_route_health_records(
                    cai_service.wallet_policy
                )
            )
        except Exception:
            return []

    def _probe_llama_cpp_rpc_route_health_before_placement(
        self,
        *,
        model_card: ModelCard,
        min_nodes: int,
    ) -> None:
        if (
            model_card.inference_backend != InferenceBackend.LlamaCpp
            or min_nodes <= 1
        ):
            return
        try:
            cai_service = make_cai_service(
                state_url=f"http://127.0.0.1:{self.port}/state",
                cai_url=f"http://127.0.0.1:{self.port}",
                local_node_id=str(self.node_id),
            )
            probe = getattr(
                cai_service.modules.route_health,
                "probe_llama_cpp_rpc_routes",
                None,
            )
            if not callable(probe):
                return
            timeout_sec = float(
                os.environ.get("CAI_LLAMA_CPP_RPC_PREFLIGHT_TIMEOUT_SECONDS", "0.75")
                or "0.75"
            )
            probe(
                state_payload=self._state_payload_with_current_local_identity(),
                local_node_id=str(self.node_id),
                timeout_sec=max(0.1, timeout_sec),
                policy=cai_service.wallet_policy,
            )
        except Exception:
            logger.debug("Unable to probe llama.cpp RPC route health before placement")

    def _state_payload_with_current_local_identity(self) -> Any:
        if not hasattr(self.state, "model_dump"):
            return self.state

        payload = self.state.model_dump(by_alias=True)
        node_identities = payload.setdefault("nodeIdentities", {})
        if not isinstance(node_identities, dict):
            return payload

        local_node_id = str(self.node_id).strip()
        if not local_node_id:
            return payload

        identity = node_identities.get(local_node_id)
        if not isinstance(identity, dict):
            identity = {}

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
                load_or_create_node_config,
                resolve_worker_reward_address,
            )
            from cai_compute_chain.wallet import (
                get_active_wallet,
                load_unlocked_wallet_signing_material,
                normalize_address,
            )

            wallet_policy = WalletPolicy()
            config = load_or_create_node_config(wallet_policy)
            worker_enabled = bool(getattr(config, "worker_enabled", False))
            relay_enabled = bool(getattr(config, "relay_enabled", False))
            cai_owned_runtime_enabled = str(
                os.getenv("CAI_OWNED_TRANSPORT_RUNTIME_ENABLED", "1")
            ).strip().lower() not in {"0", "false", "no", "off", "disabled"}
            runtime_ready_raw = os.getenv("CAI_OWNED_TRANSPORT_RUNTIME_READY")
            cai_owned_runtime_ready_claim = (
                bool(cai_owned_runtime_enabled)
                if runtime_ready_raw is None or not runtime_ready_raw.strip()
                else runtime_ready_raw.strip().lower()
                not in {"0", "false", "no", "off", "disabled"}
            )
            reward_address = resolve_worker_reward_address(
                local_node_id,
                wallet_policy,
            )
            if not reward_address:
                active_wallet = get_active_wallet(wallet_policy)
                if active_wallet is not None:
                    reward_address = active_wallet.address
            normalized_reward_address = (
                normalize_address(reward_address) if reward_address else None
            )
            node_public_key_b64 = None
            node_public_key_address = None
            active_wallet = get_active_wallet(wallet_policy)
            if active_wallet is not None:
                signing_material = load_unlocked_wallet_signing_material(
                    active_wallet,
                    wallet_policy,
                )
                if isinstance(signing_material, dict):
                    node_public_key_b64 = str(
                        signing_material.get("public_key_b64") or ""
                    ).strip() or None
                    node_public_key_address = str(
                        signing_material.get("address") or ""
                    ).strip().lower() or None
            effective_worker_enabled = bool(worker_enabled and normalized_reward_address)
            llm_self_test = (
                load_cai_owned_llm_shard_self_test_result(policy=wallet_policy)
                if effective_worker_enabled
                else None
            )
            live_proof = load_cai_owned_transport_live_proof_result(
                policy=wallet_policy,
            )
        except Exception:
            return payload

        identity = {
            **identity,
            "workerEnabled": effective_worker_enabled,
            "relayEnabled": relay_enabled,
            "readiness": {
                **(
                    identity.get("readiness")
                    if isinstance(identity.get("readiness"), dict)
                    else {}
                ),
                "caiOwnedTransport": cai_owned_transport_runtime_readiness(
                    runtime_ready=bool(
                        effective_worker_enabled
                        and cai_owned_runtime_enabled
                        and cai_owned_runtime_ready_claim
                    ),
                    implemented=True,
                    proof_kind="deterministic_bytes_shard_runtime_loop",
                    status=(
                        "ready"
                        if (
                            effective_worker_enabled
                            and cai_owned_runtime_enabled
                            and cai_owned_runtime_ready_claim
                        )
                        else "test_adapter_ready"
                        if effective_worker_enabled and cai_owned_runtime_enabled
                        else "disabled"
                    ),
                    llm_shard_self_test=llm_self_test,
                    runtime_ready_proof=live_proof,
                    require_runtime_ready_proof=str(
                        os.getenv("CAI_OWNED_TRANSPORT_REQUIRE_LIVE_PROOF", "1")
                    ).strip().lower()
                    not in {"0", "false", "no", "off", "disabled"},
                ),
            },
        }
        if normalized_reward_address:
            identity["workerRewardAddress"] = normalized_reward_address
        if node_public_key_b64:
            identity["nodePublicKeyB64"] = node_public_key_b64
        if node_public_key_address:
            identity["nodePublicKeyAddress"] = node_public_key_address
        node_identities[local_node_id] = identity
        return payload

    def _resolve_execution_node_scope(
        self,
        requested_nodes: set[NodeId] | None = None,
    ) -> set[NodeId] | None:
        worker_nodes = self._resolve_worker_required_nodes()
        if worker_nodes is None:
            return requested_nodes
        if not worker_nodes:
            raise HTTPException(
                status_code=400,
                detail="No worker-enabled CAI nodes are currently available.",
            )
        if requested_nodes is None:
            return worker_nodes

        disallowed_nodes = sorted(
            str(node_id) for node_id in requested_nodes.difference(worker_nodes)
        )
        if disallowed_nodes:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Execution is restricted to worker-enabled CAI nodes. "
                    f"Disallowed nodes: {disallowed_nodes}"
                ),
            )
        return requested_nodes

    def _build_execution_view(
        self,
        node_scope: set[NodeId] | None,
    ) -> tuple[
        Topology,
        Mapping[NodeId, MemoryUsage],
        Mapping[NodeId, NodeNetworkInfo],
        Mapping[NodeId, NodeIdentity] | None,
        Mapping[NodeId, Sequence[NodeId]],
        Mapping[InstanceId, Instance],
        Mapping[NodeId, Sequence[DownloadProgress]],
    ]:
        state = getattr(self, "state", None)
        if state is None:
            return (Topology(), {}, {}, {}, {}, {}, {})

        overlay_peers = getattr(state, "overlay_peers", {}) or {}
        if node_scope is None:
            return (
                state.topology,
                state.node_memory,
                state.node_network,
                state.node_identities,
                overlay_peers,
                state.instances,
                state.downloads,
            )

        state_node_memory = getattr(state, "node_memory", {}) or {}
        state_node_network = getattr(state, "node_network", {}) or {}
        state_node_identities = getattr(state, "node_identities", {}) or {}
        capability_records = self._load_verified_worker_capability_records()
        capability_identities: dict[NodeId, NodeIdentity] = {}
        capability_memory: dict[NodeId, MemoryUsage] = {}
        capability_overlay_peers: dict[NodeId, set[NodeId]] = {}
        for node_id in node_scope:
            record = capability_records.get(node_id)
            if record is None:
                continue
            if node_id not in state_node_identities:
                capability_identities[node_id] = _capability_record_node_identity(
                    record
                )
            if node_id not in state_node_memory:
                memory = _capability_record_node_memory(record)
                if memory is not None:
                    capability_memory[node_id] = memory
            route_peers = _capability_record_route_peers(record)
            if route_peers:
                capability_overlay_peers.setdefault(node_id, set()).update(route_peers)

        route_scope = set(node_scope)
        route_scope.update(capability_identities.keys())
        for peers in capability_overlay_peers.values():
            route_scope.update(peers)
        if state_node_identities is not None:
            for node_id, identity in state_node_identities.items():
                if bool(getattr(identity, "relay_enabled", False)):
                    route_scope.add(node_id)
                    continue
                transport_endpoints = getattr(identity, "transport_endpoints", None)
                if transport_endpoints and any(
                    str(getattr(endpoint, "route_type", "")).strip() == "relay"
                    for endpoint in transport_endpoints
                ):
                    route_scope.add(node_id)

        for node_id, peers in overlay_peers.items():
            peer_set = set(peers)
            if node_id in route_scope or peer_set.intersection(route_scope):
                route_scope.add(node_id)
                route_scope.update(peer_set)

        relay_nodes = {
            node_id
            for node_id, identity in {
                **dict(state_node_identities or {}),
                **capability_identities,
            }.items()
            if bool(getattr(identity, "relay_enabled", False))
            or any(
                str(getattr(endpoint, "route_type", "")).strip() == "relay"
                for endpoint in (getattr(identity, "transport_endpoints", None) or [])
            )
        }
        for node_id, peers in list(capability_overlay_peers.items()):
            for peer_node_id in list(peers):
                if peer_node_id not in route_scope:
                    continue
                if peer_node_id in relay_nodes:
                    capability_overlay_peers.setdefault(peer_node_id, set()).add(
                        node_id
                    )

        topology = state.topology.get_subgraph_from_nodes(list(route_scope))
        node_memory = {
            node_id: usage
            for node_id, usage in state_node_memory.items()
            if node_id in node_scope
        }
        for node_id, usage in capability_memory.items():
            if node_id in node_scope and node_id not in node_memory:
                node_memory[node_id] = usage
        node_network = {
            node_id: info
            for node_id, info in state_node_network.items()
            if node_id in route_scope
        }
        node_identities = (
            {
                node_id: identity
                for node_id, identity in state_node_identities.items()
                if node_id in route_scope
            }
            if state_node_identities is not None
            else None
        )
        if capability_identities:
            if node_identities is None:
                node_identities = {}
            for node_id, identity in capability_identities.items():
                if node_id in route_scope and node_id not in node_identities:
                    node_identities[node_id] = identity
        current_instances = {
            instance_id: instance
            for instance_id, instance in state.instances.items()
            if set(instance.shard_assignments.node_to_runner.keys()).issubset(node_scope)
        }
        scoped_overlay_peers = {
            node_id: [peer for peer in peers if peer in route_scope]
            for node_id, peers in overlay_peers.items()
            if node_id in route_scope
        }
        for node_id, peers in capability_overlay_peers.items():
            if node_id not in route_scope:
                continue
            merged_peers = list(scoped_overlay_peers.get(node_id, []))
            for peer in sorted(peers, key=str):
                if peer in route_scope and peer not in merged_peers:
                    merged_peers.append(peer)
            scoped_overlay_peers[node_id] = merged_peers
        download_status = {
            node_id: progress
            for node_id, progress in state.downloads.items()
            if node_id in node_scope
        }
        return (
            topology,
            node_memory,
            node_network,
            node_identities,
            scoped_overlay_peers,
            current_instances,
            download_status,
        )

    def _validate_worker_only_instance(self, instance: Instance) -> None:
        node_scope = self._resolve_execution_node_scope()
        if node_scope is None:
            return

        instance_nodes = set(instance.shard_assignments.node_to_runner.keys())
        disallowed_nodes = sorted(
            str(node_id) for node_id in instance_nodes.difference(node_scope)
        )
        if disallowed_nodes:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Execution is restricted to worker-enabled CAI nodes. "
                    f"Disallowed nodes: {disallowed_nodes}"
                ),
            )

    def unpause(self, result_clock: int, master_node_id: NodeId | None = None):
        logger.info("Unpausing API")
        if master_node_id is not None:
            self.current_master_node_id = master_node_id
        self.last_completed_election = result_clock
        self.paused = False
        self.paused_ev.set()
        self.paused_ev = anyio.Event()

    def _setup_exception_handlers(self) -> None:
        self.app.exception_handler(HTTPException)(self.http_exception_handler)

    async def http_exception_handler(
        self, _: Request, exc: HTTPException
    ) -> JSONResponse:
        err = ErrorResponse(
            error=ErrorInfo(
                message=exc.detail,
                type=HTTPStatus(exc.status_code).phrase,
                code=exc.status_code,
            )
        )
        return JSONResponse(err.model_dump(), status_code=exc.status_code)

    def _setup_cors(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self) -> None:
        self.app.get("/node_id")(lambda: self.node_id)
        self.app.post("/instance")(self.create_instance)
        self.app.post("/place_instance")(self.place_instance)
        self.app.get("/instance/placement")(self.get_placement)
        self.app.get("/instance/previews")(self.get_placement_previews)
        self.app.get("/instance/{instance_id}")(self.get_instance)
        self.app.delete("/instance/{instance_id}")(self.delete_instance)
        self.app.get("/models")(self.get_models)
        self.app.get("/v1/models")(self.get_models)
        self.app.post("/models/add")(self.add_custom_model)
        self.app.delete("/models/custom/{model_id:path}")(self.delete_custom_model)
        self.app.get("/models/search")(self.search_models)
        self.app.post("/v1/chat/completions", response_model=None)(
            self.chat_completions
        )
        self.app.post("/bench/chat/completions")(self.bench_chat_completions)
        self.app.post("/v1/images/generations", response_model=None)(
            self.image_generations
        )
        self.app.post("/bench/images/generations")(self.bench_image_generations)
        self.app.post("/v1/images/edits", response_model=None)(self.image_edits)
        self.app.post("/bench/images/edits")(self.bench_image_edits)
        self.app.get("/images")(self.list_images)
        self.app.get("/images/{image_id}")(self.get_image)
        self.app.post("/v1/messages", response_model=None)(self.claude_messages)
        self.app.post("/v1/responses", response_model=None)(self.openai_responses)
        self.app.post("/v1/cancel/{command_id}")(self.cancel_command)

        # Ollama API
        self.app.head("/ollama/")(self.ollama_version)
        self.app.head("/ollama/api/version")(self.ollama_version)
        self.app.post("/ollama/api/chat", response_model=None)(self.ollama_chat)
        self.app.post("/ollama/api/api/chat", response_model=None)(self.ollama_chat)
        self.app.post("/ollama/api/v1/chat", response_model=None)(self.ollama_chat)
        self.app.post("/ollama/api/generate", response_model=None)(self.ollama_generate)
        self.app.get("/ollama/api/tags")(self.ollama_tags)
        self.app.get("/ollama/api/api/tags")(self.ollama_tags)
        self.app.get("/ollama/api/v1/tags")(self.ollama_tags)
        self.app.post("/ollama/api/show")(self.ollama_show)
        self.app.get("/ollama/api/ps")(self.ollama_ps)
        self.app.get("/ollama/api/version")(self.ollama_version)

        self.app.get("/state")(self.get_state)
        self.app.get("/state/{path:path}")(self.get_state)
        self.app.get("/dashboard/state")(self.get_dashboard_state)
        self.app.get("/dashboard/state/{path:path}")(self.get_dashboard_state)
        self.app.get("/cai/summary")(self.get_cai_summary)
        self.app.get("/v1/cai/summary")(self.get_cai_summary)
        self.app.get("/v1/cai/chain")(self.get_cai_chain)
        self.app.get("/v1/cai/validators")(self.get_cai_validator_set)
        self.app.get("/v1/cai/validator-evidence")(self.get_cai_validator_evidence)
        self.app.get("/v1/cai/node-capabilities")(self.get_cai_node_capabilities)
        self.app.get("/v1/cai/worker-capability-attestations")(
            self.get_cai_worker_capability_attestations
        )
        self.app.get("/v1/cai/route-health")(self.get_cai_route_health)
        self.app.get("/v1/cai/compute-cells")(self.get_cai_compute_cells)
        self.app.get("/v1/cai/distributed-inference/diagnostics")(
            self.get_cai_distributed_inference_diagnostics
        )
        self.app.get("/v1/cai/transport/sessions")(self.get_cai_transport_sessions)
        self.app.get("/v1/cai/transport/batch-inbox")(
            self.get_cai_transport_batch_inbox
        )
        self.app.get("/v1/cai/chunk-inventory")(self.get_cai_chunk_inventory)
        self.app.get("/v1/cai/chunks")(self.get_cai_chunk_payload)
        self.app.get("/v1/cai/history")(self.get_cai_history)
        self.app.get("/v1/cai/desktop/preferences")(self.get_cai_desktop_preferences)
        self.app.put("/v1/cai/desktop/preferences")(self.update_cai_desktop_preferences)
        self.app.get("/v1/cai/update-manifest")(self.get_cai_update_manifest)
        self.app.get("/v1/cai/update-package")(self.get_cai_update_package)
        self.app.get("/v1/cai/update-package.zip")(self.get_cai_update_package)
        self.app.get("/v1/cai/update/status")(self.get_cai_update_status)
        self.app.post("/v1/cai/update/check")(self.check_cai_update)
        self.app.post("/v1/cai/update/apply")(self.apply_cai_update)
        self.app.post("/v1/cai/update/cancel")(self.cancel_cai_update)
        self.app.post("/v1/cai/update/activity")(self.record_cai_update_activity)
        self.app.post("/v1/cai/chat/completions", response_model=None)(
            self.cai_chat_completions
        )
        self.app.post("/v1/cai/validators/sync")(self.sync_cai_validator_set)
        self.app.post("/v1/cai/chain/sync")(self.sync_cai_chain)
        self.app.post("/v1/cai/validator-evidence/sync")(self.sync_cai_validator_evidence)
        self.app.post("/v1/cai/node-capabilities/sync")(self.sync_cai_node_capabilities)
        self.app.post("/v1/cai/worker-capability-attestations/sync")(
            self.sync_cai_worker_capability_attestations
        )
        self.app.post("/v1/cai/route-health/probe")(self.probe_cai_route_health)
        self.app.post("/v1/cai/transport/sessions")(self.create_cai_transport_session)
        self.app.post("/v1/cai/transport/batch-inbox/claim-next")(
            self.claim_next_cai_transport_batch
        )
        self.app.post("/v1/cai/transport/sessions/{session_id}/offer")(
            self.accept_cai_transport_session_offer
        )
        self.app.post("/v1/cai/transport/sessions/{session_id}/batches")(
            self.record_cai_transport_batch
        )
        self.app.post("/v1/cai/transport/sessions/{session_id}/batch-envelopes")(
            self.record_cai_transport_batch_envelope
        )
        self.app.post("/v1/cai/transport/overlay/send")(
            self.send_cai_transport_overlay_message
        )
        self.app.post("/v1/cai/transport/sessions/{session_id}/batches/{batch_id}/status")(
            self.mark_cai_transport_batch_status
        )
        self.app.post("/v1/cai/transport/sessions/{session_id}/batches/{batch_id}/claim")(
            self.claim_cai_transport_batch
        )
        self.app.post("/v1/cai/transport/sessions/{session_id}/batches/{batch_id}/heartbeat")(
            self.heartbeat_cai_transport_batch
        )
        self.app.post("/v1/cai/transport/sessions/{session_id}/batches/{batch_id}/complete-work-item")(
            self.complete_cai_transport_work_item
        )
        self.app.post("/v1/cai/transport/sessions/{session_id}/batches/{batch_id}/fail-work-item")(
            self.fail_cai_transport_work_item
        )
        self.app.get("/v1/cai/transport/sessions/{session_id}/batches/{batch_id}/payload")(
            self.get_cai_transport_batch_payload
        )
        self.app.get("/v1/cai/transport/sessions/{session_id}/final-output")(
            self.get_cai_transport_final_output
        )
        self.app.post("/v1/cai/transport/sessions/{session_id}/shard-receipts")(
            self.record_cai_transport_shard_receipt
        )
        self.app.post("/v1/cai/transport/sessions/{session_id}/complete")(
            self.complete_cai_transport_session
        )
        self.app.post("/v1/cai/transport/sessions/{session_id}/completion-notice")(
            self.accept_cai_transport_completion_notice
        )
        self.app.post("/v1/cai/transport/sessions/{session_id}/await-final-result")(
            self.await_cai_transport_final_result
        )
        self.app.post("/v1/cai/transport/sessions/{session_id}/reconcile-timeouts")(
            self.reconcile_cai_transport_timeouts
        )
        self.app.post("/v1/cai/chunk-inventory/sync")(self.sync_cai_chunk_inventory)
        self.app.post("/v1/cai/settlement/attest")(self.attest_cai_settlement)
        self.app.post("/v1/cai/worker-capability/challenge")(
            self.answer_cai_worker_capability_challenge
        )
        self.app.post("/v1/cai/worker-capability/attest")(
            self.attest_cai_worker_capability
        )
        self.app.post("/v1/cai/validator-penalty/attest")(self.attest_cai_penalty_case)
        self.app.post("/v1/cai/wallet/create")(self.create_cai_wallet)
        self.app.post("/v1/cai/wallet/restore")(self.restore_cai_wallet)
        self.app.post("/v1/cai/wallet/select")(self.select_cai_wallet)
        self.app.post("/v1/cai/wallet/unlock")(self.unlock_cai_wallet)
        self.app.post("/v1/cai/wallet/lock")(self.lock_cai_wallet)
        self.app.post("/v1/cai/wallet/logout")(self.logout_cai_wallet)
        self.app.post("/v1/cai/wallet/send")(self.send_cai_wallet_transfer)
        self.app.post("/v1/cai/node/validator")(self.set_cai_validator_enabled)
        self.app.post("/v1/cai/node/validator/unbond-complete")(self.complete_cai_validator_unbond)
        self.app.post("/v1/cai/node/validator/unjail")(self.clear_cai_validator_jail)
        self.app.post("/v1/cai/node/validator/static-ip")(self.set_cai_validator_static_ip)
        self.app.post("/v1/cai/node/worker")(self.set_cai_worker_enabled)
        self.app.post("/v1/cai/node/relay")(self.set_cai_relay_enabled)
        self.app.get("/v1/cai/relay/rpc/probe", response_model=None)(
            self.cai_relay_rpc_probe
        )
        self.app.websocket("/v1/cai/relay/rpc/ws")(self.cai_relay_rpc_websocket)
        self.app.websocket("/v1/cai/relay/rpc/reverse/ws")(
            self.cai_reverse_relay_rpc_websocket
        )
        self.app.get("/events")(self.stream_events)
        self.app.post("/download/start")(self.start_download)
        self.app.delete("/download/{node_id}/{model_id:path}")(self.delete_download)
        self.app.post("/download/cancel")(self.cancel_download)
        self.app.get("/v1/traces")(self.list_traces)
        self.app.post("/v1/traces/delete")(self.delete_traces)
        self.app.get("/v1/traces/{task_id}")(self.get_trace)
        self.app.get("/v1/traces/{task_id}/stats")(self.get_trace_stats)
        self.app.get("/v1/traces/{task_id}/raw")(self.get_trace_raw)
        self.app.get("/onboarding")(self.get_onboarding)
        self.app.post("/onboarding")(self.complete_onboarding)

    @staticmethod
    def _apply_private_network_override_to_model_card(
        model_card: ModelCard,
        *,
        private_network_model: bool,
    ) -> ModelCard:
        if private_network_model and not model_card.is_custom:
            return model_card.model_copy(update={"is_custom": True})
        return model_card

    @staticmethod
    def _model_card_from_instance(instance: Instance) -> ModelCard | None:
        model_id = instance.shard_assignments.model_id
        cards: list[ModelCard] = []
        for shard in instance.shard_assignments.runner_to_shard.values():
            model_card = getattr(shard, "model_card", None)
            if isinstance(model_card, ModelCard):
                cards.append(model_card)
        if not cards:
            return None
        if any(card.model_id != model_id for card in cards):
            return None
        return cards[0]

    @staticmethod
    def _llama_cpp_layer_range_supported(model_card: ModelCard) -> bool:
        if model_card.inference_backend != InferenceBackend.LlamaCpp:
            return True
        return (
            bool(getattr(model_card, "layer_range_supported", False))
            and str(getattr(model_card, "shard_compatibility", "") or "").strip()
            == "layer_range_supported"
        )

    @staticmethod
    def _validate_llama_cpp_multi_node_sharding(
        model_card: ModelCard,
        *,
        min_nodes: int,
    ) -> None:
        if (
            model_card.inference_backend != InferenceBackend.LlamaCpp
            or int(min_nodes) <= 1
            or API._llama_cpp_layer_range_supported(model_card)
        ):
            return
        architecture = str(
            getattr(model_card, "gguf_architecture", "") or "unknown"
        ).strip()
        compatibility = str(
            getattr(model_card, "shard_compatibility", "") or "unsupported_for_sharding"
        ).strip()
        raise HTTPException(
            status_code=400,
            detail=(
                f"GGUF architecture '{architecture}' is {compatibility}; "
                "multi-node layer-range placement requires a successful "
                "CAI layer-range conformance probe. Use single-node full-model "
                "GGUF mode or add an architecture-specific probe first."
            ),
        )

    async def get_state(self, request: Request, path: str = ""):
        if self.state_local_only and not self._request_is_local(request):
            raise HTTPException(status_code=404, detail="Not found.")
        payload = self._state_payload_with_current_local_identity()
        if path == "":
            return payload
        try:
            return cast(Any, self._resolve_path(payload, path))
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"unable to find path '{path.replace('/', '.')}' in state json",
            ) from e

    def get_dashboard_state(self, path: str = ""):
        if self.dashboard_disabled:
            raise HTTPException(status_code=404, detail="Dashboard is disabled.")
        worker_node_ids = self._resolve_worker_required_nodes()
        dashboard_state = build_dashboard_state(
            self._dashboard_state_with_capability_backed_workers(worker_node_ids),
            self.node_id,
            worker_node_ids=worker_node_ids,
        )
        if path == "":
            return dashboard_state
        try:
            return cast(Any, self._resolve_path(dashboard_state, path))
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"unable to find path '{path.replace('/', '.')}' in dashboard state json",
            ) from e

    def _dashboard_state_with_capability_backed_workers(
        self,
        worker_node_ids: set[NodeId] | None,
    ) -> State:
        if not worker_node_ids:
            return self.state

        capability_records = self._load_verified_worker_capability_records()
        if not capability_records:
            return self.state

        state_node_identities = getattr(self.state, "node_identities", {}) or {}
        state_node_memory = getattr(self.state, "node_memory", {}) or {}
        state_overlay_peers = getattr(self.state, "overlay_peers", {}) or {}
        topology = copy.deepcopy(self.state.topology)
        topology_node_ids = set(topology.list_nodes())
        node_identities = dict(state_node_identities)
        node_memory = dict(state_node_memory)
        overlay_peers = {
            node_id: list(peers)
            for node_id, peers in state_overlay_peers.items()
        }
        changed = False
        relay_nodes = {
            node_id
            for node_id, identity in node_identities.items()
            if bool(getattr(identity, "relay_enabled", False))
            or any(
                str(getattr(endpoint, "route_type", "")).strip() == "relay"
                for endpoint in (getattr(identity, "transport_endpoints", None) or [])
            )
        }

        for node_id in worker_node_ids:
            record = capability_records.get(node_id)
            if record is None:
                continue

            if node_id not in topology_node_ids:
                topology.add_node(node_id)
                topology_node_ids.add(node_id)
                changed = True

            if node_id not in node_identities:
                node_identities[node_id] = _capability_record_node_identity(record)
                changed = True

            if node_id not in node_memory:
                memory = _capability_record_node_memory(record)
                if memory is not None:
                    node_memory[node_id] = memory
                    changed = True

            route_peers = _capability_record_route_peers(record)
            if not route_peers:
                continue

            current_peers = list(overlay_peers.get(node_id, []))
            for peer_node_id in sorted(route_peers, key=str):
                if peer_node_id not in topology_node_ids:
                    topology.add_node(peer_node_id)
                    topology_node_ids.add(peer_node_id)
                    changed = True
                if peer_node_id not in current_peers:
                    current_peers.append(peer_node_id)
                    changed = True
                if peer_node_id in relay_nodes:
                    reverse_peers = list(overlay_peers.get(peer_node_id, []))
                    if node_id not in reverse_peers:
                        reverse_peers.append(node_id)
                        overlay_peers[peer_node_id] = reverse_peers
                        changed = True
            if current_peers:
                overlay_peers[node_id] = current_peers

        if not changed:
            return self.state

        return self.state.model_copy(
            update={
                "topology": topology,
                "node_identities": node_identities,
                "node_memory": node_memory,
                "overlay_peers": overlay_peers,
            }
        )

    def _resolve_path(self, payload: Any, path: str) -> Any:
        current = payload
        for attr in path.split("/"):
            if attr == "":
                continue
            if isinstance(current, dict):
                current = current[attr]  # pyright: ignore[reportUnknownVariableType]
            elif isinstance(current, list):
                current = current[int(attr)]  # pyright: ignore[reportUnknownVariableType]
        return current

    @staticmethod
    def _request_is_local(request: Request) -> bool:
        client = request.client
        host = str(client.host).strip().lower() if client and client.host else ""
        return host in {"127.0.0.1", "::1", "localhost"}

    def _require_local_request(self, request: Request) -> None:
        if not self._request_is_local(request):
            logger.warning(
                "CAI audit: {}",
                safe_audit_event(
                    "local_only_denied",
                    method=request.method,
                    path=request.url.path,
                    client_host=self._rate_limit_client_key(request),
                    status="denied",
                ),
            )
            raise HTTPException(status_code=404, detail="Not found.")

    def _maybe_rate_limit_public_request(self, request: Request) -> JSONResponse | None:
        if self._request_is_local(request):
            return None
        policy = lookup_endpoint_policy(request.method, request.url.path)
        if policy is None or policy.access != EndpointAccess.PUBLIC:
            return None
        decision = self._public_rate_limiter.check(
            self._rate_limit_client_key(request)
        )
        if decision.allowed:
            return None
        logger.warning(
            "CAI audit: {}",
            safe_audit_event(
                "public_rate_limit_exceeded",
                method=request.method,
                path=request.url.path,
                client_host=self._rate_limit_client_key(request),
                status="denied",
            ),
        )
        return JSONResponse(
            {"detail": "Rate limit exceeded."},
            status_code=429,
            headers={"Retry-After": str(decision.retry_after_seconds)},
        )

    @staticmethod
    def _rate_limit_client_key(request: Request) -> str:
        client = request.client
        return str(client.host).strip().lower() if client and client.host else "unknown"

    def _require_update_server_enabled(self) -> None:
        if not bool(getattr(self, "update_server_enabled", False)):
            raise HTTPException(status_code=404, detail="Not found.")

    def _require_cai_api_bearer_token(self, request: Request) -> None:
        expected_token = str(getattr(self, "cai_api_bearer_token", "") or "").strip()
        if not expected_token:
            return

        authorization = str(request.headers.get("authorization") or "").strip()
        prefix = "bearer "
        if not authorization.lower().startswith(prefix):
            logger.warning(
                "CAI audit: {}",
                safe_audit_event(
                    "bearer_auth_failed",
                    method=request.method,
                    path=request.url.path,
                    client_host=self._rate_limit_client_key(request),
                    status="missing",
                ),
            )
            raise HTTPException(status_code=401, detail="Unauthorized.")

        provided_token = authorization[len(prefix) :].strip()
        if not provided_token or not hmac.compare_digest(provided_token, expected_token):
            logger.warning(
                "CAI audit: {}",
                safe_audit_event(
                    "bearer_auth_failed",
                    method=request.method,
                    path=request.url.path,
                    client_host=self._rate_limit_client_key(request),
                    status="invalid",
                ),
            )
            raise HTTPException(status_code=401, detail="Unauthorized.")

    def get_cai_summary(self, request: Request):
        if self.summary_local_only and not self._request_is_local(request):
            raise HTTPException(status_code=404, detail="Not found.")
        base_url = f"http://127.0.0.1:{self.port}"
        return load_cai_summary(
            state_url=f"{base_url}/state",
            cai_url=base_url,
            execution_cai_url=self._resolve_execution_cai_url(),
            state_payload_loader=self._state_payload_with_current_local_identity,
            local_node_id=str(self.node_id),
        )

    def get_cai_history(
        self,
        request: Request,
        section: Annotated[str, Query(pattern="^(journal|jobs|payouts|settlements)$")],
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=50)] = 10,
    ):
        self._require_local_request(request)
        try:
            return self._get_cai_service().history_page(
                section=section,
                offset=offset,
                limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def get_cai_desktop_preferences(self, request: Request):
        if not self._request_is_local(request):
            raise HTTPException(status_code=404, detail="Not found.")
        return {"language": resolve_language("auto")}

    def update_cai_desktop_preferences(
        self,
        request: Request,
        payload: dict[str, Any],
    ):
        if not self._request_is_local(request):
            raise HTTPException(status_code=404, detail="Not found.")
        language = str(payload.get("language") or "").strip().lower()
        if language not in SUPPORTED_DESKTOP_LANGUAGE_SET:
            raise HTTPException(status_code=400, detail="Unsupported language.")
        save_desktop_language(language)
        return {"language": resolve_language("auto")}

    def get_cai_update_manifest(self, request: Request):
        self._require_update_server_enabled()
        install_kind = str(request.query_params.get("install_kind") or "").strip().lower() or None
        try:
            return build_update_manifest(
                _resolve_cai_repo_root(),
                base_url=str(request.base_url).rstrip("/"),
                install_kind=install_kind,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    def get_cai_update_package(self, request: Request):
        self._require_update_server_enabled()
        install_kind = str(request.query_params.get("install_kind") or "").strip().lower() or None
        try:
            archive_path = build_update_package(
                _resolve_cai_repo_root(),
                install_kind=install_kind,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return _stream_cai_update_archive_response(
            archive_path,
            range_header=request.headers.get("range"),
        )

    def get_cai_update_status(self, request: Request):
        self._require_local_request(request)
        try:
            return self._get_cai_service().update_status()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    def check_cai_update(self, request: Request):
        self._require_local_request(request)
        try:
            return self._get_cai_service().check_update()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    def apply_cai_update(self, request: Request):
        self._require_local_request(request)
        try:
            return self._get_cai_service().apply_update()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    def cancel_cai_update(self, request: Request):
        self._require_local_request(request)
        try:
            return self._get_cai_service().cancel_update()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    def record_cai_update_activity(self, request: Request, payload: dict[str, Any]):
        self._require_local_request(request)
        try:
            active_count_raw = payload.get("activeRequestCount")
            active_request_count = (
                int(active_count_raw) if active_count_raw is not None else None
            )
            return record_portable_update_activity(
                source=str(payload.get("source") or "dashboard"),
                active_request_count=active_request_count,
                user_active=bool(payload.get("userActive"))
                if "userActive" in payload
                else None,
                last_user_activity_at=(
                    str(payload.get("lastUserActivityAt"))
                    if payload.get("lastUserActivityAt")
                    else None
                ),
                metadata=payload.get("metadata")
                if isinstance(payload.get("metadata"), dict)
                else None,
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    async def cai_chat_completions(
        self, request: Request, payload: ChatCompletionRequest
    ) -> ChatCompletionResponse | StreamingResponse:
        self._require_cai_api_bearer_token(request)
        try:
            result = await asyncio.to_thread(
                self._get_cai_service().chat_completion,
                payload.model_dump(mode="json", exclude_none=True),
                reserve_client_ip=self._rate_limit_client_key(request),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPError as exc:
            detail = _http_error_detail(exc) or str(exc)
            status_code = exc.code if 400 <= int(exc.code) <= 599 else 503
            raise HTTPException(status_code=status_code, detail=detail) from exc
        except (RuntimeError, TimeoutError, URLError, OSError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        response_payload = dict(result.get("response") or {})
        if not payload.stream:
            return JSONResponse(response_payload)

        return StreamingResponse(
            self._stream_cai_chat_response(
                response_payload,
                self._cai_chat_execution_event(result),
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "close",
                "X-Accel-Buffering": "no",
            },
        )

    def get_cai_validator_set(self):
        try:
            return self._get_cai_service().validator_set()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def get_cai_chain(self):
        try:
            return self._get_cai_service().chain()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def get_cai_validator_evidence(self):
        try:
            return self._get_cai_service().validator_evidence()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def get_cai_node_capabilities(self):
        try:
            return self._get_cai_service().node_capabilities()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def get_cai_worker_capability_attestations(self):
        try:
            return self._get_cai_service().worker_capability_attestations()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def get_cai_route_health(self):
        try:
            return self._get_cai_service().route_health()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def get_cai_compute_cells(self):
        try:
            return self._get_cai_service().compute_cells()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def get_cai_distributed_inference_diagnostics(
        self,
        request: Request,
        model_id: str | None = None,
    ):
        self._require_local_request(request)
        try:
            return self._get_cai_service().distributed_inference_diagnostics(
                model_id=model_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def get_cai_transport_sessions(self, request: Request):
        self._require_local_request(request)
        try:
            return self._get_cai_service().cai_owned_transport_sessions()
        except ValueError as exc:
            _raise_cai_transport_http_error(
                exc,
                operation="list_transport_sessions",
            )

    def get_cai_transport_batch_inbox(
        self,
        request: Request,
        node_id: str | None = None,
        status: str | None = "received",
    ):
        self._require_local_request(request)
        try:
            return self._get_cai_service().cai_owned_transport_batch_inbox(
                node_id=node_id,
                status=status,
            )
        except ValueError as exc:
            _raise_cai_transport_http_error(
                exc,
                operation="list_transport_batch_inbox",
            )

    def create_cai_transport_session(
        self,
        request: Request,
        payload: dict[str, Any],
    ):
        self._require_local_request(request)
        try:
            return self._get_cai_service().create_cai_owned_transport_session(payload)
        except ValueError as exc:
            _raise_cai_transport_http_error(
                exc,
                operation="create_transport_session",
            )

    def claim_next_cai_transport_batch(
        self,
        request: Request,
        payload: dict[str, Any],
    ):
        self._require_local_request(request)
        try:
            return self._get_cai_service().claim_next_cai_owned_transport_batch(
                payload,
            )
        except ValueError as exc:
            _raise_cai_transport_http_error(
                exc,
                operation="claim_next_transport_batch",
            )

    def accept_cai_transport_session_offer(
        self,
        session_id: str,
        payload: dict[str, Any],
    ):
        try:
            return self._get_cai_service().accept_cai_owned_transport_session_offer(
                session_id,
                payload,
            )
        except ValueError as exc:
            _raise_cai_transport_http_error(
                exc,
                operation="accept_transport_session_offer",
            )

    def complete_cai_transport_session(
        self,
        session_id: str,
        request: Request,
        payload: dict[str, Any],
    ):
        self._require_local_request(request)
        try:
            return self._get_cai_service().complete_cai_owned_transport_session(
                session_id,
                payload,
            )
        except ValueError as exc:
            _raise_cai_transport_http_error(
                exc,
                operation="complete_transport_session",
            )

    def accept_cai_transport_completion_notice(
        self,
        session_id: str,
        payload: dict[str, Any],
    ):
        try:
            return self._get_cai_service().accept_cai_owned_transport_completion_notice(
                session_id,
                payload,
            )
        except ValueError as exc:
            _raise_cai_transport_http_error(
                exc,
                operation="accept_transport_completion_notice",
            )

    def record_cai_transport_batch(
        self,
        session_id: str,
        request: Request,
        payload: dict[str, Any],
    ):
        self._require_local_request(request)
        try:
            return self._get_cai_service().record_cai_owned_transport_batch(
                session_id,
                payload,
            )
        except ValueError as exc:
            _raise_cai_transport_http_error(
                exc,
                operation="record_transport_batch",
            )

    def record_cai_transport_batch_envelope(
        self,
        session_id: str,
        payload: dict[str, Any],
    ):
        try:
            return self._get_cai_service().record_cai_owned_transport_batch_envelope(
                session_id,
                payload,
            )
        except ValueError as exc:
            _raise_cai_transport_http_error(
                exc,
                operation="record_transport_batch_envelope",
            )

    async def send_cai_transport_overlay_message(self, payload: dict[str, Any]):
        if self.cai_owned_transport_message_sender is None:
            _raise_cai_transport_http_error(
                RuntimeError("CAI-owned transport overlay relay is not available."),
                status_code=503,
                operation="send_transport_overlay_message",
            )
        try:
            message = CaiOwnedTransportOverlayMessage.model_validate(payload)
        except ValueError as exc:
            _raise_cai_transport_http_error(
                exc,
                operation="send_transport_overlay_message",
            )
        await self.cai_owned_transport_message_sender.send(message)
        return {
            "status": "queued",
            "messageId": message.message_id,
            "kind": message.kind,
            "sourceNodeId": message.source_node_id,
            "targetNodeId": message.target_node_id,
            "sessionId": message.session_id,
            "selectedRoute": "cai_overlay_gossipsub",
        }

    def mark_cai_transport_batch_status(
        self,
        session_id: str,
        batch_id: str,
        request: Request,
        payload: dict[str, Any],
    ):
        self._require_local_request(request)
        try:
            return self._get_cai_service().mark_cai_owned_transport_batch_status(
                session_id,
                batch_id,
                payload,
            )
        except ValueError as exc:
            _raise_cai_transport_http_error(
                exc,
                operation="mark_transport_batch_status",
            )

    def claim_cai_transport_batch(
        self,
        session_id: str,
        batch_id: str,
        request: Request,
        payload: dict[str, Any],
    ):
        self._require_local_request(request)
        try:
            return self._get_cai_service().claim_cai_owned_transport_batch(
                session_id,
                batch_id,
                payload,
            )
        except ValueError as exc:
            _raise_cai_transport_http_error(
                exc,
                operation="claim_transport_batch",
            )

    def heartbeat_cai_transport_batch(
        self,
        session_id: str,
        batch_id: str,
        request: Request,
        payload: dict[str, Any],
    ):
        self._require_local_request(request)
        try:
            return self._get_cai_service().heartbeat_cai_owned_transport_batch(
                session_id,
                batch_id,
                payload,
            )
        except ValueError as exc:
            _raise_cai_transport_http_error(
                exc,
                operation="heartbeat_transport_batch",
            )

    def complete_cai_transport_work_item(
        self,
        session_id: str,
        batch_id: str,
        request: Request,
        payload: dict[str, Any],
    ):
        self._require_local_request(request)
        try:
            return self._get_cai_service().complete_cai_owned_transport_work_item(
                session_id,
                batch_id,
                payload,
            )
        except ValueError as exc:
            _raise_cai_transport_http_error(
                exc,
                operation="complete_transport_work_item",
            )

    def fail_cai_transport_work_item(
        self,
        session_id: str,
        batch_id: str,
        request: Request,
        payload: dict[str, Any],
    ):
        self._require_local_request(request)
        try:
            return self._get_cai_service().fail_cai_owned_transport_work_item(
                session_id,
                batch_id,
                payload,
            )
        except ValueError as exc:
            _raise_cai_transport_http_error(
                exc,
                operation="fail_transport_work_item",
            )

    def get_cai_transport_batch_payload(
        self,
        session_id: str,
        batch_id: str,
        request: Request,
    ):
        self._require_local_request(request)
        try:
            path = self._get_cai_service().cai_owned_transport_batch_payload_path(
                session_id,
                batch_id,
            )
        except FileNotFoundError as exc:
            _raise_cai_transport_http_error(
                exc,
                status_code=404,
                operation="get_transport_batch_payload",
            )
        except ValueError as exc:
            _raise_cai_transport_http_error(
                exc,
                operation="get_transport_batch_payload",
            )
        return FileResponse(
            path,
            media_type="application/octet-stream",
            filename=f"{batch_id}.bin",
        )

    def get_cai_transport_final_output(
        self,
        session_id: str,
        request: Request,
        requesterNodeId: str | None = None,
    ):
        self._require_local_request(request)
        try:
            return self._get_cai_service().latest_cai_owned_transport_final_output(
                session_id,
                requester_node_id=requesterNodeId,
            )
        except FileNotFoundError as exc:
            _raise_cai_transport_http_error(
                exc,
                status_code=404,
                operation="get_transport_final_output",
            )
        except ValueError as exc:
            _raise_cai_transport_http_error(
                exc,
                operation="get_transport_final_output",
            )

    def await_cai_transport_final_result(
        self,
        session_id: str,
        request: Request,
        payload: dict[str, Any],
    ):
        self._require_local_request(request)
        try:
            return self._get_cai_service().await_cai_owned_transport_final_result(
                session_id,
                payload,
            )
        except ValueError as exc:
            _raise_cai_transport_http_error(
                exc,
                operation="await_transport_final_result",
            )

    def reconcile_cai_transport_timeouts(
        self,
        session_id: str,
        request: Request,
        payload: dict[str, Any],
    ):
        self._require_local_request(request)
        try:
            return self._get_cai_service().reconcile_cai_owned_transport_timeouts(
                session_id,
                payload,
            )
        except ValueError as exc:
            _raise_cai_transport_http_error(
                exc,
                operation="reconcile_transport_timeouts",
            )

    def record_cai_transport_shard_receipt(
        self,
        session_id: str,
        payload: dict[str, Any],
    ):
        try:
            return self._get_cai_service().record_cai_owned_transport_shard_receipt(
                session_id,
                payload,
            )
        except ValueError as exc:
            _raise_cai_transport_http_error(
                exc,
                operation="record_transport_shard_receipt",
            )

    def get_cai_chunk_inventory(
        self,
        request: Request,
        source_kind: Annotated[
            str,
            Query(pattern="^(peer_cache|storage_seed)$"),
        ] = "peer_cache",
    ):
        try:
            return self._get_cai_service().chunk_inventory(
                source_kind=source_kind,
                endpoint_base_url=str(request.base_url).rstrip("/"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def get_cai_chunk_payload(
        self,
        catalog_id: str,
        version: str,
        chunk_id: str,
    ) -> Response:
        try:
            payload = self._get_cai_service().chunk_payload(
                catalog_id=catalog_id,
                version=version,
                chunk_id=chunk_id,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return Response(content=payload, media_type="application/octet-stream")

    def sync_cai_validator_set(self, payload: dict[str, Any] | None = None):
        try:
            return self._get_cai_service().sync_validator_set(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def sync_cai_chain(self, payload: dict[str, Any]):
        try:
            return self._get_cai_service().sync_chain(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def sync_cai_validator_evidence(self):
        try:
            return self._get_cai_service().sync_validator_evidence()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def sync_cai_node_capabilities(self):
        try:
            return self._get_cai_service().sync_node_capabilities()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def sync_cai_worker_capability_attestations(self, payload: dict[str, Any]):
        try:
            return self._get_cai_service().sync_worker_capability_attestations(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def probe_cai_route_health(self, request: Request):
        self._require_local_request(request)
        try:
            return self._get_cai_service().probe_route_health()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def sync_cai_chunk_inventory(
        self,
        source_kind: Annotated[
            str,
            Query(pattern="^(peer_cache|storage_seed)$"),
        ] = "peer_cache",
    ):
        try:
            return self._get_cai_service().sync_chunk_inventory(source_kind=source_kind)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def attest_cai_settlement(self, payload: dict[str, Any]):
        try:
            return self._get_cai_service().attest_settlement(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def attest_cai_worker_capability(self, payload: dict[str, Any]):
        try:
            return self._get_cai_service().attest_worker_capability(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def answer_cai_worker_capability_challenge(self, payload: dict[str, Any]):
        try:
            return self._get_cai_service().worker_capability_challenge(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def attest_cai_penalty_case(self, payload: dict[str, Any]):
        try:
            return self._get_cai_service().attest_penalty_case(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _get_cai_service(self):
        base_url = f"http://127.0.0.1:{self.port}"
        return make_cai_service(
            state_url=f"{base_url}/state",
            cai_url=base_url,
            execution_cai_url=self._resolve_execution_cai_url(),
            state_payload_loader=self._state_payload_with_current_local_identity,
            local_node_id=str(self.node_id),
        )

    def _cai_chat_execution_event(
        self,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        job = result.get("job") if isinstance(result.get("job"), dict) else {}
        receipt = result.get("receipt") if isinstance(result.get("receipt"), dict) else {}
        settlement = (
            result.get("settlement") if isinstance(result.get("settlement"), dict) else {}
        )
        payouts = result.get("payouts") if isinstance(result.get("payouts"), list) else []
        decentralized_audit = receipt.get("decentralizedChainAudit")
        decentralized_audit = (
            decentralized_audit if isinstance(decentralized_audit, dict) else {}
        )
        proof = decentralized_audit.get("proof")
        proof = proof if isinstance(proof, dict) else {}
        reward = decentralized_audit.get("reward")
        reward = reward if isinstance(reward, dict) else {}
        network_audit = receipt.get("networkAudit")
        network_audit = network_audit if isinstance(network_audit, dict) else {}
        execution_attempts = (
            job.get("executionAttempts")
            if isinstance(job.get("executionAttempts"), list)
            else receipt.get("executionAttempts")
        )
        execution_attempts = (
            execution_attempts if isinstance(execution_attempts, list) else []
        )
        execution_attempt_status = (
            job.get("executionAttemptStatus")
            if isinstance(job.get("executionAttemptStatus"), dict)
            else receipt.get("executionAttemptStatus")
        )
        execution_attempt_status = (
            execution_attempt_status
            if isinstance(execution_attempt_status, dict)
            else None
        )
        if not job and not receipt and not settlement:
            return None
        return {
            "schemaVersion": 1,
            "source": "cai_chat_completions",
            "jobId": job.get("jobId") or receipt.get("jobId"),
            "receiptId": receipt.get("receiptId") or job.get("receiptId"),
            "settlementId": settlement.get("settlementId") or job.get("settlementId"),
            "settlementStatus": settlement.get("status"),
            "chainRecorded": settlement.get("chainRecorded"),
            "chainTransactionCount": settlement.get("chainTransactionCount"),
            "proofExecuted": proof.get("executed"),
            "proofVerified": proof.get("verified"),
            "proofError": proof.get("error"),
            "sessionId": proof.get("sessionId"),
            "finalOutputBatchCount": proof.get("finalOutputBatchCount"),
            "executionAttemptCount": len(execution_attempts),
            "executionAttempts": execution_attempts,
            "executionAttemptStatus": execution_attempt_status,
            "executorNodeIds": decentralized_audit.get("executorNodeIds") or [],
            "participantNodeIds": decentralized_audit.get("participantNodeIds") or [],
            "rewardPayoutSource": network_audit.get("rewardPayoutSource"),
            "rewardPayoutNodeIds": network_audit.get("rewardPayoutNodeIds") or [],
            "rewardSkippedNodeIdsWithoutShardReceipt": network_audit.get(
                "rewardSkippedNodeIdsWithoutShardReceipt"
            )
            or [],
            "payoutCount": reward.get("payoutCount"),
            "payoutNodes": reward.get("payoutNodes") or [],
            "workerPayoutTotalAtomic": reward.get("workerPayoutTotalAtomic"),
            "payoutStatuses": [
                item.get("status") for item in payouts if isinstance(item, dict)
            ],
        }

    async def _stream_cai_chat_response(
        self,
        response_payload: dict[str, Any],
        cai_execution_event: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        choices = response_payload.get("choices") or []
        first_choice = choices[0] if choices else {}
        message = first_choice.get("message") or {}
        delta_message = ChatCompletionMessage(
            role="assistant",
            content=message.get("content"),
            reasoning_content=message.get("reasoning_content"),
            tool_calls=message.get("tool_calls"),
        )
        usage_payload = self._normalize_chat_completion_usage(
            response_payload.get("usage")
        )
        stream_chunk = ChatCompletionResponse(
            id=str(response_payload.get("id") or uuid4()),
            created=int(response_payload.get("created") or time.time()),
            model=str(response_payload.get("model") or ""),
            choices=[
                StreamingChoiceResponse(
                    index=int(first_choice.get("index") or 0),
                    delta=delta_message,
                    logprobs=first_choice.get("logprobs"),
                    finish_reason=first_choice.get("finish_reason"),
                    usage=usage_payload,
                )
            ],
            usage=usage_payload,
            service_tier=response_payload.get("service_tier"),
        )
        yield f"data: {stream_chunk.model_dump_json()}\n\n"
        if cai_execution_event:
            yield (
                ": cai_execution "
                + json.dumps(cai_execution_event, separators=(",", ":"))
                + "\n\n"
            )
        yield "data: [DONE]\n\n"

    @staticmethod
    def _normalize_chat_completion_usage(
        usage: Any,
    ) -> dict[str, Any] | None:
        if not isinstance(usage, dict):
            return None
        try:
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            total_tokens = int(
                usage.get("total_tokens") or prompt_tokens + completion_tokens
            )
        except (TypeError, ValueError):
            return None
        payload = dict(usage)
        payload["prompt_tokens"] = prompt_tokens
        payload["completion_tokens"] = completion_tokens
        payload["total_tokens"] = total_tokens
        payload.setdefault("prompt_tokens_details", {})
        payload.setdefault("completion_tokens_details", {})
        return payload

    def create_cai_wallet(self, request: Request, payload: dict[str, Any]):
        self._require_local_request(request)
        try:
            return self._get_cai_service().create_wallet(
                name=str(payload["name"]),
                password=str(payload["password"]),
            )
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"Missing field: {exc.args[0]}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def restore_cai_wallet(self, request: Request, payload: dict[str, Any]):
        self._require_local_request(request)
        try:
            return self._get_cai_service().restore_wallet(
                name=str(payload["name"]),
                password=str(payload["password"]),
                seed_phrase=str(payload["seed_phrase"]),
            )
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"Missing field: {exc.args[0]}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def select_cai_wallet(self, request: Request, payload: dict[str, Any]):
        self._require_local_request(request)
        try:
            return self._get_cai_service().select_wallet(
                selector=str(payload["selector"]),
            )
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"Missing field: {exc.args[0]}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def unlock_cai_wallet(self, request: Request, payload: dict[str, Any]):
        self._require_local_request(request)
        try:
            wallet = payload.get("wallet")
            return self._get_cai_service().unlock_wallet(
                password=str(payload["password"]),
                wallet=str(wallet) if wallet is not None and str(wallet).strip() else None,
            )
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"Missing field: {exc.args[0]}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def lock_cai_wallet(self, request: Request):
        self._require_local_request(request)
        try:
            return self._get_cai_service().lock_wallet()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def logout_cai_wallet(self, request: Request):
        self._require_local_request(request)
        try:
            return self._get_cai_service().logout_wallet()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def send_cai_wallet_transfer(self, request: Request, payload: dict[str, Any]):
        self._require_local_request(request)
        try:
            return self._get_cai_service().send_wallet_transfer(
                to=str(payload["to"]),
                amount=str(payload["amount"]),
            )
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"Missing field: {exc.args[0]}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def set_cai_validator_enabled(self, request: Request, payload: dict[str, Any]):
        self._require_local_request(request)
        try:
            return self._get_cai_service().set_validator_enabled(
                enabled=bool(payload["enabled"])
            )
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"Missing field: {exc.args[0]}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def set_cai_validator_static_ip(self, request: Request, payload: dict[str, Any]):
        self._require_local_request(request)
        try:
            return self._get_cai_service().set_validator_static_ip_confirmed(
                confirmed=bool(payload["confirmed"])
            )
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"Missing field: {exc.args[0]}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def complete_cai_validator_unbond(self, request: Request):
        self._require_local_request(request)
        try:
            return self._get_cai_service().complete_validator_unbond()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def clear_cai_validator_jail(self, request: Request):
        self._require_local_request(request)
        try:
            return self._get_cai_service().clear_validator_jail()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def set_cai_worker_enabled(self, request: Request, payload: dict[str, Any]):
        self._require_local_request(request)
        try:
            return self._get_cai_service().set_worker_enabled(
                enabled=bool(payload["enabled"])
            )
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"Missing field: {exc.args[0]}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def set_cai_relay_enabled(self, request: Request, payload: dict[str, Any]):
        self._require_local_request(request)
        try:
            return self._get_cai_service().set_relay_enabled(
                enabled=bool(payload["enabled"])
            )
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"Missing field: {exc.args[0]}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _local_node_relay_enabled(self) -> bool:
        identity = self.state.node_identities.get(self.node_id)
        return bool(getattr(identity, "relay_enabled", False))

    def _relay_target_allowed(
        self,
        sink_node_id: str,
        target_host: str,
        target_port: int,
    ) -> bool:
        normalized_sink_node_id = str(sink_node_id or "").strip()
        normalized_target_host = str(target_host or "").strip()
        if (
            not normalized_sink_node_id
            or not normalized_target_host
            or target_port <= 0
        ):
            return False

        identity = self.state.node_identities.get(NodeId(normalized_sink_node_id))
        if identity is None:
            return False

        sink = NodeId(normalized_sink_node_id)

        def endpoint_matches(host: str | None, port: int | None) -> bool:
            return (
                str(host or "").strip() == normalized_target_host
                and int(port or 0) == int(target_port)
            )

        for endpoint in identity.transport_endpoints_for(
            purpose="data",
            require_port=True,
        ):
            if endpoint_matches(endpoint.host, endpoint.port):
                return True

        if endpoint_matches(identity.data_host, identity.data_port):
            return True

        known_hosts = {
            str(host or "").strip()
            for host in (identity.data_host, identity.api_host)
            if str(host or "").strip()
        }
        for purpose in ("data", "api"):
            for endpoint in identity.transport_endpoints_for(purpose=purpose):
                endpoint_host = str(endpoint.host or "").strip()
                if endpoint_host:
                    known_hosts.add(endpoint_host)

        network_info = self.state.node_network.get(sink)
        for interface in getattr(network_info, "interfaces", []) or []:
            interface_host = str(getattr(interface, "ip_address", "") or "").strip()
            if interface_host:
                known_hosts.add(interface_host)

        if normalized_target_host not in known_hosts:
            return False

        for instance in self.state.instances.values():
            if not isinstance(instance, MlxRingInstance):
                continue
            if int(instance.ephemeral_port) != int(target_port):
                continue
            if sink in instance.shard_assignments.node_to_runner:
                return True

        return False

    async def _relay_websocket_to_stream(
        self,
        websocket: WebSocket,
        stream,
    ) -> None:
        while True:
            message = await websocket.receive()
            message_type = str(message.get("type") or "")
            if message_type == "websocket.disconnect":
                return
            if message_type != "websocket.receive":
                continue
            payload = message.get("bytes")
            if payload is None:
                text_payload = message.get("text")
                if text_payload is None:
                    continue
                if str(text_payload).strip() == _RELAY_EOF_MESSAGE:
                    await self._relay_stream_send_eof(stream)
                    return
                payload = str(text_payload).encode("utf-8")
            await stream.send(cast(bytes, payload))

    async def _relay_stream_to_websocket(
        self,
        stream,
        websocket: WebSocket,
    ) -> None:
        while True:
            chunk = await stream.receive(_RELAY_STREAM_CHUNK_SIZE)
            if not chunk:
                await websocket.send_text(_RELAY_EOF_MESSAGE)
                return
            await websocket.send_bytes(chunk)

    async def _relay_stream_send_eof(self, stream) -> None:
        send_eof = getattr(stream, "send_eof", None)
        if callable(send_eof):
            await send_eof()
            return

    @staticmethod
    def _reverse_relay_key(
        sink_node_id: str,
        target_host: str,
        target_port: int,
    ) -> tuple[str, str, int]:
        return (
            str(sink_node_id or "").strip(),
            str(target_host or "").strip(),
            int(target_port),
        )

    async def _reverse_relay_queue(
        self,
        key: tuple[str, str, int],
    ) -> asyncio.Queue[_ReverseRelaySession]:
        if not hasattr(self, "_reverse_relay_queues"):
            self._reverse_relay_queues = {}
        if not hasattr(self, "_reverse_relay_queues_lock"):
            self._reverse_relay_queues_lock = asyncio.Lock()
        async with self._reverse_relay_queues_lock:
            queue = self._reverse_relay_queues.get(key)
            if queue is None:
                queue = asyncio.Queue()
                self._reverse_relay_queues[key] = queue
            return queue

    async def _wait_for_reverse_relay_session(
        self,
        key: tuple[str, str, int],
    ) -> _ReverseRelaySession | None:
        queue = await self._reverse_relay_queue(key)
        try:
            with anyio.fail_after(_REVERSE_RELAY_WAIT_TIMEOUT_SECONDS):
                return await queue.get()
        except TimeoutError:
            return None

    async def _reverse_relay_queue_size(
        self,
        key: tuple[str, str, int],
    ) -> int:
        if not hasattr(self, "_reverse_relay_queues"):
            return 0
        if not hasattr(self, "_reverse_relay_queues_lock"):
            return 0
        async with self._reverse_relay_queues_lock:
            queue = self._reverse_relay_queues.get(key)
            return 0 if queue is None else int(queue.qsize())

    def _record_relay_probe_route_health(
        self,
        *,
        source_node_id: str | None,
        sink_node_id: str,
        transit_node_id: str | None,
        target_host: str,
        target_port: int,
        ready: bool,
        mode: str,
        reverse_channels: int = 0,
        error: str | None = None,
    ) -> None:
        try:
            cai_service = make_cai_service(
                state_url=f"http://127.0.0.1:{self.port}/state",
                cai_url=f"http://127.0.0.1:{self.port}",
                local_node_id=str(self.node_id),
            )
            cai_service.modules.route_health.record_relay_probe_result(
                source_node_id=source_node_id,
                sink_node_id=sink_node_id,
                transit_node_id=transit_node_id or str(self.node_id),
                target_host=target_host,
                target_port=target_port,
                ready=ready,
                mode=mode,
                reverse_channels=reverse_channels,
                error=error,
                policy=cai_service.wallet_policy,
            )
        except Exception as exc:
            logger.debug("Unable to record relay probe route health: {}", exc)

    async def _relay_websocket_to_websocket(
        self,
        source: WebSocket,
        target: WebSocket,
    ) -> None:
        while True:
            message = await source.receive()
            message_type = str(message.get("type") or "")
            if message_type == "websocket.disconnect":
                return
            if message_type != "websocket.receive":
                continue
            payload = message.get("bytes")
            if payload is not None:
                await target.send_bytes(cast(bytes, payload))
                continue
            text_payload = message.get("text")
            if text_payload is not None:
                await target.send_text(str(text_payload))

    async def _bridge_relay_websockets(
        self,
        source_websocket: WebSocket,
        reverse_session: _ReverseRelaySession,
    ) -> None:
        reverse_websocket = reverse_session.websocket
        try:
            await reverse_websocket.send_text("connected")
            await source_websocket.send_text("connected")
            logger.debug("Relay websocket paired with reverse relay channel")

            async def relay_direction(source: WebSocket, target: WebSocket) -> None:
                try:
                    await self._relay_websocket_to_websocket(source, target)
                finally:
                    tg.cancel_scope.cancel()

            async with anyio.create_task_group() as tg:
                tg.start_soon(
                    relay_direction,
                    source_websocket,
                    reverse_websocket,
                )
                tg.start_soon(
                    relay_direction,
                    reverse_websocket,
                    source_websocket,
                )
        finally:
            reverse_session.done.set()
            with contextlib.suppress(Exception):
                await reverse_websocket.close()
            with contextlib.suppress(Exception):
                await source_websocket.close()

    async def _wait_for_reverse_relay_target_ready(
        self,
        reverse_websocket: WebSocket,
    ) -> None:
        try:
            with anyio.fail_after(_REVERSE_RELAY_TARGET_READY_TIMEOUT_SECONDS):
                while True:
                    message = await reverse_websocket.receive()
                    message_type = str(message.get("type") or "")
                    if message_type == "websocket.disconnect":
                        raise RuntimeError(
                            "reverse relay disconnected before target became ready"
                        )
                    if message_type != "websocket.receive":
                        continue
                    if message.get("bytes") is not None:
                        raise RuntimeError(
                            "reverse relay sent data before target readiness"
                        )
                    text_payload = message.get("text")
                    if text_payload is None:
                        continue
                    payload = str(text_payload or "").strip()
                    if payload == _RELAY_TARGET_CONNECTED_MESSAGE:
                        return
                    if payload.lower() in {"connected", "registered"}:
                        continue
                    raise RuntimeError(
                        "reverse relay returned a non-ready target status: "
                        f"{payload}"
                    )
        except TimeoutError as exc:
            raise RuntimeError(
                "reverse relay target did not become ready in time"
            ) from exc

    async def _probe_llama_cpp_rpc_hello_stream(self, stream) -> str:
        await stream.send(
            bytes([_LLAMA_CPP_RPC_CMD_HELLO])
            + struct.pack("<Q", _LLAMA_CPP_RPC_CONN_CAPS_SIZE)
            + bytes(_LLAMA_CPP_RPC_CONN_CAPS_SIZE)
        )
        response_size = struct.unpack(
            "<Q",
            await self._receive_stream_exact(stream, 8),
        )[0]
        if response_size != _LLAMA_CPP_RPC_HELLO_RESPONSE_SIZE:
            raise RuntimeError(f"unexpected HELLO response size {response_size}")
        response = await self._receive_stream_exact(stream, response_size)
        return self._parse_llama_cpp_rpc_hello_response(response)

    async def _receive_stream_exact(self, stream, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = int(size)
        while remaining > 0:
            chunk = await stream.receive(remaining)
            if not chunk:
                raise RuntimeError("connection closed during HELLO")
            chunks.append(bytes(chunk))
            remaining -= len(chunk)
        return b"".join(chunks)

    async def _probe_llama_cpp_rpc_hello_websocket(
        self,
        websocket: WebSocket,
    ) -> str:
        await websocket.send_text("connected")
        await websocket.send_bytes(
            bytes([_LLAMA_CPP_RPC_CMD_HELLO])
            + struct.pack("<Q", _LLAMA_CPP_RPC_CONN_CAPS_SIZE)
            + bytes(_LLAMA_CPP_RPC_CONN_CAPS_SIZE)
        )
        buffer = bytearray()
        while len(buffer) < 8:
            buffer.extend(await self._receive_websocket_chunk(websocket))
        response_size = struct.unpack("<Q", bytes(buffer[:8]))[0]
        if response_size != _LLAMA_CPP_RPC_HELLO_RESPONSE_SIZE:
            raise RuntimeError(f"unexpected HELLO response size {response_size}")
        while len(buffer) < 8 + response_size:
            buffer.extend(await self._receive_websocket_chunk(websocket))
        response = bytes(buffer[8 : 8 + response_size])
        return self._parse_llama_cpp_rpc_hello_response(response)

    async def _receive_websocket_chunk(
        self,
        websocket: WebSocket,
    ) -> bytes:
        while True:
            message = await websocket.receive()
            message_type = str(message.get("type") or "")
            if message_type == "websocket.disconnect":
                raise RuntimeError("reverse relay disconnected during HELLO")
            if message_type != "websocket.receive":
                continue
            payload = message.get("bytes")
            if payload is None:
                text = str(message.get("text") or "").strip()
                if text == _RELAY_EOF_MESSAGE:
                    raise RuntimeError("reverse relay closed during HELLO")
                continue
            return bytes(cast(bytes, payload))

    @staticmethod
    def _parse_llama_cpp_rpc_hello_response(response: bytes) -> str:
        if len(response) != _LLAMA_CPP_RPC_HELLO_RESPONSE_SIZE:
            raise RuntimeError(f"invalid HELLO response length {len(response)}")
        major, minor, patch = response[0], response[1], response[2]
        if major <= 0:
            raise RuntimeError("invalid HELLO protocol version")
        return f"{major}.{minor}.{patch}"

    async def cai_reverse_relay_rpc_websocket(
        self,
        websocket: WebSocket,
        target_host: str,
        target_port: int,
        sink_node_id: str,
        source_node_id: str | None = None,
        transit_node_id: str | None = None,
    ) -> None:
        if not self._local_node_relay_enabled():
            await websocket.close(code=4403, reason="relay disabled")
            return
        if not self._relay_target_allowed(sink_node_id, target_host, target_port):
            await websocket.close(code=4403, reason="relay target not allowed")
            return

        key = self._reverse_relay_key(sink_node_id, target_host, target_port)
        reverse_session = _ReverseRelaySession(websocket=websocket)
        await websocket.accept()
        logger.debug(
            "Reverse relay websocket registered source_node_id={} transit_node_id={} sink_node_id={} target={}:{}",
            source_node_id,
            transit_node_id or self.node_id,
            sink_node_id,
            target_host,
            target_port,
        )
        await websocket.send_text("registered")
        try:
            await self._wait_for_reverse_relay_target_ready(websocket)
            queue = await self._reverse_relay_queue(key)
            await queue.put(reverse_session)
            await reverse_session.done.wait()
        finally:
            reverse_session.done.set()
            with contextlib.suppress(Exception):
                await websocket.close()

    async def cai_relay_rpc_probe(
        self,
        target_host: str,
        target_port: int,
        sink_node_id: str,
        source_node_id: str | None = None,
        transit_node_id: str | None = None,
        protocol: str | None = None,
    ) -> Any:
        if not self._local_node_relay_enabled():
            raise HTTPException(status_code=403, detail="relay disabled")
        if not self._relay_target_allowed(sink_node_id, target_host, target_port):
            raise HTTPException(status_code=403, detail="relay target not allowed")

        protocol_probe = str(protocol or "").strip().lower().replace("-", "_")
        wants_llama_cpp_rpc = protocol_probe in {"llama_cpp_rpc", "ggml_rpc"}
        key = self._reverse_relay_key(sink_node_id, target_host, target_port)
        reverse_channels = await self._reverse_relay_queue_size(key)
        if reverse_channels > 0:
            if wants_llama_cpp_rpc:
                reverse_session = await self._wait_for_reverse_relay_session(key)
                if reverse_session is not None:
                    try:
                        version = await self._probe_llama_cpp_rpc_hello_websocket(
                            reverse_session.websocket
                        )
                        self._record_relay_probe_route_health(
                            source_node_id=source_node_id,
                            sink_node_id=sink_node_id,
                            transit_node_id=transit_node_id,
                            target_host=target_host,
                            target_port=target_port,
                            ready=True,
                            mode="reverse",
                            reverse_channels=reverse_channels,
                        )
                        return {
                            "ready": True,
                            "mode": "reverse",
                            "reverseChannels": reverse_channels,
                            "protocol": "llama_cpp_rpc",
                            "protocolReady": True,
                            "protocolVersion": version,
                        }
                    except Exception as exc:
                        self._record_relay_probe_route_health(
                            source_node_id=source_node_id,
                            sink_node_id=sink_node_id,
                            transit_node_id=transit_node_id,
                            target_host=target_host,
                            target_port=target_port,
                            ready=False,
                            mode="reverse_protocol_failed",
                            reverse_channels=reverse_channels,
                            error=str(exc),
                        )
                        return JSONResponse(
                            status_code=503,
                            content={
                                "ready": False,
                                "mode": "reverse_protocol_failed",
                                "reverseChannels": reverse_channels,
                                "protocol": "llama_cpp_rpc",
                                "protocolReady": False,
                                "message": str(exc),
                            },
                        )
                    finally:
                        reverse_session.done.set()
                        with contextlib.suppress(Exception):
                            await reverse_session.websocket.close()
            else:
                self._record_relay_probe_route_health(
                    source_node_id=source_node_id,
                    sink_node_id=sink_node_id,
                    transit_node_id=transit_node_id,
                    target_host=target_host,
                    target_port=target_port,
                    ready=True,
                    mode="reverse",
                    reverse_channels=reverse_channels,
                )
                return {
                    "ready": True,
                    "mode": "reverse",
                    "reverseChannels": reverse_channels,
                }

        try:
            with anyio.fail_after(_RELAY_TARGET_CONNECT_TIMEOUT_SECONDS):
                stream = await anyio.connect_tcp(target_host, target_port)
            async with stream:
                protocol_version = None
                if wants_llama_cpp_rpc:
                    protocol_version = await self._probe_llama_cpp_rpc_hello_stream(
                        stream
                    )
                self._record_relay_probe_route_health(
                    source_node_id=source_node_id,
                    sink_node_id=sink_node_id,
                    transit_node_id=transit_node_id,
                    target_host=target_host,
                    target_port=target_port,
                    ready=True,
                    mode="direct",
                    reverse_channels=reverse_channels,
                )
                return {
                    "ready": True,
                    "mode": "direct",
                    "reverseChannels": reverse_channels,
                    "protocol": "llama_cpp_rpc" if wants_llama_cpp_rpc else None,
                    "protocolReady": True if wants_llama_cpp_rpc else None,
                    "protocolVersion": protocol_version,
                }
        except Exception as exc:
            logger.debug(
                "Relay probe unavailable source_node_id={} transit_node_id={} sink_node_id={} target={}:{}: {}",
                source_node_id,
                transit_node_id or self.node_id,
                sink_node_id,
                target_host,
                target_port,
                exc,
            )
            self._record_relay_probe_route_health(
                source_node_id=source_node_id,
                sink_node_id=sink_node_id,
                transit_node_id=transit_node_id,
                target_host=target_host,
                target_port=target_port,
                ready=False,
                mode="unavailable",
                reverse_channels=reverse_channels,
                error=str(exc),
            )
            return JSONResponse(
                status_code=503,
                content={
                    "ready": False,
                    "mode": "unavailable",
                    "reverseChannels": reverse_channels,
                    "protocol": "llama_cpp_rpc" if wants_llama_cpp_rpc else None,
                    "protocolReady": False if wants_llama_cpp_rpc else None,
                    "message": str(exc),
                },
            )

    async def cai_relay_rpc_websocket(
        self,
        websocket: WebSocket,
        target_host: str,
        target_port: int,
        sink_node_id: str,
        source_node_id: str | None = None,
        transit_node_id: str | None = None,
    ) -> None:
        if not self._local_node_relay_enabled():
            await websocket.close(code=4403, reason="relay disabled")
            return
        if not self._relay_target_allowed(sink_node_id, target_host, target_port):
            await websocket.close(code=4403, reason="relay target not allowed")
            return

        await websocket.accept()
        logger.debug(
            "Relay websocket opening source_node_id={} transit_node_id={} sink_node_id={} target={}:{}",
            source_node_id,
            transit_node_id or self.node_id,
            sink_node_id,
            target_host,
            target_port,
        )
        try:
            with anyio.fail_after(_RELAY_TARGET_CONNECT_TIMEOUT_SECONDS):
                stream = await anyio.connect_tcp(target_host, target_port)
        except Exception as direct_exc:
            logger.debug(
                "Relay direct target {}:{} unavailable, waiting for reverse relay channel: {}",
                target_host,
                target_port,
                direct_exc,
            )
            key = self._reverse_relay_key(sink_node_id, target_host, target_port)
            reverse_session = await self._wait_for_reverse_relay_session(key)
            if reverse_session is None:
                logger.debug(
                    "Relay reverse channel unavailable sink_node_id={} target={}:{}",
                    sink_node_id,
                    target_host,
                    target_port,
                )
                await websocket.send_text(
                    "relay target is not reachable and no reverse relay channel is available: "
                    f"{direct_exc}"
                )
                await websocket.close(code=1011)
                return
            try:
                await self._bridge_relay_websockets(websocket, reverse_session)
            except Exception as reverse_exc:
                with contextlib.suppress(Exception):
                    await websocket.send_text(f"reverse relay channel failed: {reverse_exc}")
                with contextlib.suppress(Exception):
                    await websocket.close(code=1011)
            return

        async with stream:
            await websocket.send_text("connected")
            try:
                async with anyio.create_task_group() as tg:
                    tg.start_soon(self._relay_websocket_to_stream, websocket, stream)
                    tg.start_soon(self._relay_stream_to_websocket, stream, websocket)
            except WebSocketDisconnect:
                return
            except (BrokenResourceError, ClosedResourceError):
                return
            finally:
                with contextlib.suppress(Exception):
                    await websocket.close()

    async def place_instance(self, payload: PlaceInstanceParams):
        model_card = self._apply_private_network_override_to_model_card(
            await ModelCard.load(payload.model_id),
            private_network_model=payload.private_network_model,
        )
        if (
            model_card.inference_backend == InferenceBackend.LlamaCpp
            and payload.instance_meta == InstanceMeta.MlxRing
        ):
            payload = payload.model_copy(update={"instance_meta": InstanceMeta.LlamaCpp})
        node_scope = self._resolve_execution_node_scope()
        (
            execution_topology,
            execution_memory,
            execution_network,
            execution_identities,
            execution_overlay_peers,
            execution_instances,
            execution_downloads,
        ) = self._build_execution_view(node_scope)
        try:
            sharding, min_nodes = enforce_private_network_model_request(
                model_card.model_id,
                payload.sharding,
                payload.min_nodes,
                model_card=model_card,
                available_nodes=len(execution_memory),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        self._validate_llama_cpp_multi_node_sharding(
            model_card,
            min_nodes=min_nodes,
        )
        command = PlaceInstance(
            model_card=model_card,
            sharding=sharding,
            instance_meta=payload.instance_meta,
            min_nodes=min_nodes,
        )
        self._probe_llama_cpp_rpc_route_health_before_placement(
            model_card=model_card,
            min_nodes=min_nodes,
        )
        route_health_records = self._load_route_health_records()
        try:
            placements = get_instance_placements(
                command,
                node_memory=execution_memory,
                node_network=execution_network,
                topology=execution_topology,
                current_instances=execution_instances,
                node_identities=execution_identities,
                overlay_peers=execution_overlay_peers,
                required_nodes=None,
                download_status=execution_downloads,
                route_health_records=route_health_records,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        current_ids = set(execution_instances.keys())
        new_instances = [
            instance
            for instance_id, instance in placements.items()
            if instance_id not in current_ids
        ]
        if len(new_instances) != 1:
            raise HTTPException(
                status_code=500,
                detail="Expected exactly one new instance from placement",
            )

        create_command = CreateInstance(instance=new_instances[0])
        self._validate_worker_only_instance(create_command.instance)
        await self._send(create_command)

        return CreateInstanceResponse(
            message="Command received.",
            command_id=create_command.command_id,
            model_card=model_card,
        )

    async def create_instance(
        self, payload: CreateInstanceParams
    ) -> CreateInstanceResponse:
        instance = payload.instance
        logger.debug(
            "Create instance request received instance_id={} model_id={} nodes={}",
            instance.instance_id,
            instance.shard_assignments.model_id,
            sorted(str(node_id) for node_id in instance.shard_assignments.node_to_runner),
        )
        try:
            validate_private_network_instance(instance)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        self._validate_worker_only_instance(instance)
        model_card = self._model_card_from_instance(instance)
        if model_card is None:
            model_card = await ModelCard.load(instance.shard_assignments.model_id)
        self._validate_llama_cpp_multi_node_sharding(
            model_card,
            min_nodes=len(instance.shard_assignments.node_to_runner),
        )
        required_memory = model_card.storage_size
        available_memory = self._calculate_total_available_memory()

        if required_memory > available_memory:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient memory to create instance. Required: {required_memory.in_gb:.1f}GB, Available: {available_memory.in_gb:.1f}GB",
            )

        command = CreateInstance(
            instance=instance,
        )
        logger.debug(
            "Dispatching CreateInstance command_id={} instance_id={}",
            command.command_id,
            instance.instance_id,
        )
        await self._send(command)

        return CreateInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            model_card=model_card,
        )

    async def get_placement(
        self,
        model_id: ModelId,
        sharding: Sharding = Sharding.Pipeline,
        instance_meta: InstanceMeta = InstanceMeta.MlxRing,
        min_nodes: int = 1,
        private_network_model: bool = False,
    ) -> Instance:
        model_card = self._apply_private_network_override_to_model_card(
            await ModelCard.load(model_id),
            private_network_model=private_network_model,
        )
        if (
            model_card.inference_backend == InferenceBackend.LlamaCpp
            and instance_meta == InstanceMeta.MlxRing
        ):
            instance_meta = InstanceMeta.LlamaCpp
        node_scope = self._resolve_execution_node_scope()
        (
            execution_topology,
            execution_memory,
            execution_network,
            execution_identities,
            execution_overlay_peers,
            execution_instances,
            execution_downloads,
        ) = self._build_execution_view(node_scope)
        sharding, min_nodes = enforce_private_network_model_request(
            model_card.model_id,
            sharding,
            min_nodes,
            model_card=model_card,
            available_nodes=len(execution_memory),
        )
        self._validate_llama_cpp_multi_node_sharding(
            model_card,
            min_nodes=min_nodes,
        )

        try:
            self._probe_llama_cpp_rpc_route_health_before_placement(
                model_card=model_card,
                min_nodes=min_nodes,
            )
            route_health_records = self._load_route_health_records()
            placements = get_instance_placements(
                PlaceInstance(
                    model_card=model_card,
                    sharding=sharding,
                    instance_meta=instance_meta,
                    min_nodes=min_nodes,
                ),
                node_memory=execution_memory,
                node_network=execution_network,
                topology=execution_topology,
                current_instances=execution_instances,
                node_identities=execution_identities,
                overlay_peers=execution_overlay_peers,
                required_nodes=None,
                download_status=execution_downloads,
                route_health_records=route_health_records,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        current_ids = set(execution_instances.keys())
        new_ids = [
            instance_id for instance_id in placements if instance_id not in current_ids
        ]
        if len(new_ids) != 1:
            raise HTTPException(
                status_code=500,
                detail="Expected exactly one new instance from placement",
            )

        return placements[new_ids[0]]

    async def get_placement_previews(
        self,
        model_id: ModelId,
        node_ids: Annotated[list[NodeId] | None, Query()] = None,
        private_network_model: bool = False,
    ) -> PlacementPreviewResponse:
        seen: set[tuple[ModelId, Sharding, InstanceMeta, int]] = set()
        previews: list[PlacementPreview] = []
        requested_nodes = set(node_ids) if node_ids else None
        node_scope = self._resolve_execution_node_scope(requested_nodes)
        (
            execution_topology,
            execution_memory,
            execution_network,
            execution_identities,
            execution_overlay_peers,
            execution_instances,
            execution_downloads,
        ) = self._build_execution_view(node_scope)

        if len(list(execution_topology.list_nodes())) == 0:
            return PlacementPreviewResponse(previews=[])

        try:
            model_card = self._apply_private_network_override_to_model_card(
                await ModelCard.load(model_id),
                private_network_model=private_network_model,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Failed to load model card: {exc}"
            ) from exc
        policy = get_private_network_model_policy()
        is_private_model = is_private_network_model(
            model_card.model_id,
            model_card=model_card,
        )
        instance_combinations: list[tuple[Sharding, InstanceMeta, int]] = []
        if model_card.inference_backend == InferenceBackend.LlamaCpp:
            min_nodes_start = (
                private_network_model_effective_min_nodes(
                    model_card.model_id,
                    model_card=model_card,
                    available_nodes=len(execution_memory),
                )
                if is_private_model
                else 1
            )
            max_nodes = max(1, len(execution_memory))
            if not self._llama_cpp_layer_range_supported(model_card):
                max_nodes = 1
            instance_combinations = [
                (Sharding.Pipeline, InstanceMeta.LlamaCpp, i)
                for i in range(min_nodes_start, max_nodes + 1)
            ]
        else:
            sharding_options = (
                (Sharding.Pipeline,)
                if is_private_model and policy.require_pipeline_sharding
                else (Sharding.Pipeline, Sharding.Tensor)
            )
            min_nodes_start = (
                private_network_model_effective_min_nodes(
                    model_card.model_id,
                    model_card=model_card,
                    available_nodes=len(execution_memory),
                )
                if is_private_model
                else 1
            )
            max_nodes = max(1, len(execution_memory))
            for sharding in sharding_options:
                for instance_meta in (InstanceMeta.MlxRing, InstanceMeta.MlxJaccl):
                    instance_combinations.extend(
                        [
                            (sharding, instance_meta, i)
                            for i in range(
                                min_nodes_start,
                                max_nodes + 1,
                            )
                        ]
                    )
        # TODO: PDD
        # instance_combinations.append((Sharding.PrefillDecodeDisaggregation, InstanceMeta.MlxRing, 1))

        self._probe_llama_cpp_rpc_route_health_before_placement(
            model_card=model_card,
            min_nodes=max(
                (
                    candidate_min_nodes
                    for _sharding, _meta, candidate_min_nodes in instance_combinations
                ),
                default=1,
            ),
        )
        route_health_records = self._load_route_health_records()

        for sharding, instance_meta, min_nodes in instance_combinations:
            try:
                placements = get_instance_placements(
                    PlaceInstance(
                        model_card=model_card,
                        sharding=sharding,
                        instance_meta=instance_meta,
                        min_nodes=min_nodes,
                    ),
                    node_memory=execution_memory,
                    node_network=execution_network,
                    topology=execution_topology,
                    current_instances=execution_instances,
                    node_identities=execution_identities,
                    overlay_peers=execution_overlay_peers,
                    required_nodes=None,
                    download_status=execution_downloads,
                    route_health_records=route_health_records,
                )
            except ValueError as exc:
                if (model_card.model_id, sharding, instance_meta, 0) not in seen:
                    previews.append(
                        PlacementPreview(
                            model_id=model_card.model_id,
                            sharding=sharding,
                            instance_meta=instance_meta,
                            instance=None,
                            error=str(exc),
                        )
                    )
                seen.add((model_card.model_id, sharding, instance_meta, 0))
                continue

            current_ids = set(execution_instances.keys())
            new_instances = [
                instance
                for instance_id, instance in placements.items()
                if instance_id not in current_ids
            ]

            if len(new_instances) != 1:
                if (model_card.model_id, sharding, instance_meta, 0) not in seen:
                    previews.append(
                        PlacementPreview(
                            model_id=model_card.model_id,
                            sharding=sharding,
                            instance_meta=instance_meta,
                            instance=None,
                            error="Expected exactly one new instance from placement",
                        )
                    )
                seen.add((model_card.model_id, sharding, instance_meta, 0))
                continue

            instance = new_instances[0]
            shard_assignments = instance.shard_assignments
            placement_node_ids = list(shard_assignments.node_to_runner.keys())

            memory_delta_by_node: dict[str, int] = {}
            if placement_node_ids:
                total_bytes = model_card.storage_size.in_bytes
                per_node = total_bytes // len(placement_node_ids)
                remainder = total_bytes % len(placement_node_ids)
                for index, node_id in enumerate(sorted(placement_node_ids, key=str)):
                    extra = 1 if index < remainder else 0
                    memory_delta_by_node[str(node_id)] = per_node + extra

            if (
                model_card.model_id,
                sharding,
                instance_meta,
                len(placement_node_ids),
            ) not in seen:
                previews.append(
                    PlacementPreview(
                        model_id=model_card.model_id,
                        sharding=sharding,
                        instance_meta=instance_meta,
                        instance=instance,
                        memory_delta_by_node=memory_delta_by_node or None,
                        error=None,
                    )
                )
            seen.add(
                (
                    model_card.model_id,
                    sharding,
                    instance_meta,
                    len(placement_node_ids),
                )
            )

        return PlacementPreviewResponse(previews=previews)

    def get_instance(self, instance_id: InstanceId) -> Instance:
        if instance_id not in self.state.instances:
            raise HTTPException(status_code=404, detail="Instance not found")
        return self.state.instances[instance_id]

    async def delete_instance(self, instance_id: InstanceId) -> DeleteInstanceResponse:
        if instance_id not in self.state.instances:
            raise HTTPException(status_code=404, detail="Instance not found")

        command = DeleteInstance(
            instance_id=instance_id,
        )
        await self._send(command)
        return DeleteInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            instance_id=instance_id,
        )

    async def cancel_command(self, command_id: CommandId) -> CancelCommandResponse:
        """Cancel an active command by closing its stream and notifying workers."""
        sender = self._text_generation_queues.get(
            command_id
        ) or self._image_generation_queues.get(command_id)
        if sender is None and not self._state_has_active_command(command_id):
            raise HTTPException(
                status_code=404,
                detail="Command not found or already completed",
            )

        await self._send(TaskCancelled(cancelled_command_id=command_id))
        if sender is not None:
            sender.close()

        return CancelCommandResponse(
            message="Command cancelled.",
            command_id=command_id,
        )

    def _state_has_active_command(self, command_id: CommandId) -> bool:
        for task in self.state.tasks.values():
            task_command_id = getattr(task, "command_id", None)
            if task_command_id != command_id:
                continue

            task_status = getattr(task, "task_status", None)
            if task_status in {"Pending", "Running"}:
                return True
            if getattr(task_status, "value", None) in {"Pending", "Running"}:
                return True

        return False

    async def _token_chunk_stream(
        self, command_id: CommandId
    ) -> AsyncGenerator[
        TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk, None
    ]:
        """Yield chunks for a given command until completion.

        This is the internal low-level stream used by all API adapters.
        """
        try:
            logger.debug("opening token chunk stream for command {}", command_id)
            self._text_generation_queues[command_id], recv = channel[
                TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk
            ]()

            with recv as token_chunks:
                async for chunk in token_chunks:
                    logger.debug(
                        "token chunk stream command_id={} received chunk_type={} finish_reason={}",
                        command_id,
                        type(chunk).__name__,
                        getattr(chunk, "finish_reason", None),
                    )
                    yield chunk
                    if isinstance(chunk, PrefillProgressChunk):
                        continue
                    if chunk.finish_reason is not None:
                        break

        except anyio.get_cancelled_exc_class():
            command = TaskCancelled(cancelled_command_id=command_id)
            with anyio.CancelScope(shield=True):
                await self.command_sender.send(
                    ForwarderCommand(origin=self._system_id, command=command)
                )
            raise
        finally:
            logger.debug("closing token chunk stream for command {}", command_id)
            with anyio.move_on_after(1, shield=True) as scope:
                await self._send(TaskFinished(finished_command_id=command_id))
            if scope.cancel_called:
                logger.warning(
                    "timed out sending TaskFinished for command {}, closing response stream anyway",
                    command_id,
                )
            if command_id in self._text_generation_queues:
                del self._text_generation_queues[command_id]

    async def _collect_text_generation_with_stats(
        self, command_id: CommandId
    ) -> BenchChatCompletionResponse:
        sampler = PowerSampler(get_node_system=lambda: self.state.node_system)
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        model: ModelId | None = None
        finish_reason: FinishReason | None = None

        stats: GenerationStats | None = None

        async with anyio.create_task_group() as tg:
            tg.start_soon(sampler.run)

            async for chunk in self._token_chunk_stream(command_id):
                if isinstance(chunk, PrefillProgressChunk):
                    continue

                if chunk.finish_reason == "error":
                    raise HTTPException(
                        status_code=500,
                        detail=chunk.error_message or "Internal server error",
                    )

                if model is None:
                    model = chunk.model

                if isinstance(chunk, TokenChunk):
                    text_parts.append(chunk.text)

                if isinstance(chunk, ToolCallChunk):
                    tool_calls.extend(
                        ToolCall(
                            id=str(uuid4()),
                            index=i,
                            function=tool,
                        )
                        for i, tool in enumerate(chunk.tool_calls)
                    )

                stats = chunk.stats or stats

                if chunk.finish_reason is not None:
                    finish_reason = chunk.finish_reason

            tg.cancel_scope.cancel()

        combined_text = "".join(text_parts)
        assert model is not None

        return BenchChatCompletionResponse(
            id=command_id,
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=combined_text,
                        tool_calls=tool_calls if tool_calls else None,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            generation_stats=stats,
            power_usage=sampler.result(),
        )

    async def _trigger_notify_user_to_download_model(self, model_id: ModelId) -> None:
        logger.warning(
            "TODO: we should send a notification to the user to download the model"
        )

    _sent_image_hashes: set[str] = set()

    async def _send_text_generation_with_images(
        self, task_params: TextGenerationTaskParams
    ) -> TextGeneration:
        images = task_params.images
        if not images:
            command = TextGeneration(task_params=task_params)
            await self._send(command)
            return command

        hashes = [hashlib.sha256(img.encode("ascii")).hexdigest() for img in images]

        cached_hashes: dict[int, str] = {}
        new_images: list[tuple[int, str]] = []
        for idx, (img, h) in enumerate(zip(images, hashes, strict=True)):
            if h in self._sent_image_hashes:
                cached_hashes[idx] = h
            else:
                self._sent_image_hashes.add(h)
                new_images.append((idx, img))

        wrapped_hashes = {idx: Base64Image(h) for idx, h in cached_hashes.items()}

        if not new_images:
            task_params = task_params.model_copy(
                update={"images": [], "image_hashes": wrapped_hashes}
            )
            command = TextGeneration(task_params=task_params)
            await self._send(command)
            return command

        all_chunks: list[tuple[int, str]] = []
        for img_idx, img_data in new_images:
            for i in range(0, len(img_data), CAI_MAX_CHUNK_SIZE):
                all_chunks.append((img_idx, img_data[i : i + CAI_MAX_CHUNK_SIZE]))

        task_params = task_params.model_copy(
            update={
                "images": [],
                "image_hashes": wrapped_hashes,
                "total_input_chunks": len(all_chunks),
                "image_count": len(new_images),
            }
        )
        command = TextGeneration(task_params=task_params)

        for global_idx, (img_idx, chunk_data) in enumerate(all_chunks):
            await self._send(
                SendInputChunk(
                    chunk=InputImageChunk(
                        model=task_params.model,
                        command_id=command.command_id,
                        data=chunk_data,
                        chunk_index=global_idx,
                        total_chunks=len(all_chunks),
                        image_index=img_idx,
                    )
                )
            )

        await self._send(command)
        return command

    async def chat_completions(
        self, payload: ChatCompletionRequest
    ) -> ChatCompletionResponse | StreamingResponse:
        """OpenAI Chat Completions API - adapter."""
        task_params = await chat_request_to_text_generation(payload)
        resolved_model = await self._resolve_and_validate_text_model(
            ModelId(task_params.model)
        )
        task_params = task_params.model_copy(update={"model": resolved_model})

        command = await self._send_text_generation_with_images(task_params)

        if payload.stream:
            return StreamingResponse(
                with_sse_keepalive(
                    generate_chat_stream(
                        command.command_id,
                        self._token_chunk_stream(command.command_id),
                    ),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            try:
                response_payload = await collect_chat_response_payload(
                    command.command_id,
                    self._token_chunk_stream(command.command_id),
                )
            except ValueError as exc:
                detail = text_generation_failure_detail(
                    getattr(self, "state", None),
                    command_id=command.command_id,
                    model_id=resolved_model,
                    fallback=str(exc),
                )
                raise HTTPException(status_code=503, detail=detail) from exc
            return JSONResponse(response_payload.model_dump(mode="json"))

    async def bench_chat_completions(
        self, payload: BenchChatCompletionRequest
    ) -> BenchChatCompletionResponse:
        task_params = await chat_request_to_text_generation(payload)
        resolved_model = await self._resolve_and_validate_text_model(
            ModelId(task_params.model)
        )
        task_params = task_params.model_copy(update={"model": resolved_model})

        task_params = task_params.model_copy(
            update={
                "stream": False,
                "bench": True,
                "use_prefix_cache": payload.use_prefix_cache,
            }
        )

        command = await self._send_text_generation_with_images(task_params)

        return await self._collect_text_generation_with_stats(command.command_id)

    async def _resolve_and_validate_text_model(self, model_id: ModelId) -> ModelId:
        """Validate a text model exists and return the resolved model ID.

        Raises HTTPException 404 if no instance is found for the model.
        """
        if not any(
            instance.shard_assignments.model_id == model_id
            for instance in self.state.instances.values()
        ):
            await self._trigger_notify_user_to_download_model(model_id)
            raise HTTPException(
                status_code=404,
                detail=f"No instance found for model {model_id}",
            )
        runner_failure = runner_failure_message_for_model(self.state, model_id)
        if runner_failure:
            raise HTTPException(status_code=503, detail=runner_failure)
        return model_id

    async def _validate_image_model(self, model: ModelId) -> ModelId:
        """Validate model exists and return resolved model ID.

        Raises HTTPException 404 if no instance is found for the model.
        """
        model_card = await ModelCard.load(model)
        resolved_model = model_card.model_id
        if not any(
            instance.shard_assignments.model_id == resolved_model
            for instance in self.state.instances.values()
        ):
            await self._trigger_notify_user_to_download_model(resolved_model)
            raise HTTPException(
                status_code=404, detail=f"No instance found for model {resolved_model}"
            )
        return resolved_model

    def stream_events(self) -> StreamingResponse:
        def _generate_json_array(events: Iterable[Event]) -> Iterable[str]:
            yield "["
            first = True
            for event in events:
                if not first:
                    yield ","
                first = False
                yield event.model_dump_json()
            yield "]"

        return StreamingResponse(
            _generate_json_array(self._event_log.read_all()),
            media_type="application/json",
        )

    async def get_image(self, image_id: str) -> FileResponse:
        stored = self._image_store.get(Id(image_id))
        if stored is None:
            raise HTTPException(status_code=404, detail="Image not found or expired")
        return FileResponse(path=stored.file_path, media_type=stored.content_type)

    async def list_images(self, request: Request) -> ImageListResponse:
        """List all stored images."""
        stored_images = self._image_store.list_images()
        return ImageListResponse(
            data=[
                ImageListItem(
                    image_id=img.image_id,
                    url=self._build_image_url(request, img.image_id),
                    content_type=img.content_type,
                    expires_at=img.expires_at,
                )
                for img in stored_images
            ]
        )

    def _build_image_url(self, request: Request, image_id: Id) -> str:
        host = request.headers.get("host", f"localhost:{self.port}")
        scheme = "https" if request.url.scheme == "https" else "http"
        return f"{scheme}://{host}/v1/images/{image_id}"

    async def image_generations(
        self, request: Request, payload: ImageGenerationTaskParams
    ) -> ImageGenerationResponse | StreamingResponse:
        """Handle image generation requests.

        When stream=True and partial_images > 0, returns a StreamingResponse
        with SSE-formatted events for partial and final images.
        """
        payload = payload.model_copy(
            update={
                "model": await self._validate_image_model(ModelId(payload.model)),
                "advanced_params": _ensure_seed(payload.advanced_params),
            }
        )

        command = ImageGeneration(
            task_params=payload,
        )
        await self._send(command)

        # Check if streaming is requested
        if payload.stream and payload.partial_images and payload.partial_images > 0:
            return StreamingResponse(
                self._generate_image_stream(
                    request=request,
                    command_id=command.command_id,
                    num_images=payload.n or 1,
                    response_format=payload.response_format or "b64_json",
                ),
                media_type="text/event-stream",
            )

        # Non-streaming: collect all image chunks
        return await self._collect_image_generation(
            request=request,
            command_id=command.command_id,
            num_images=payload.n or 1,
            response_format=payload.response_format or "b64_json",
        )

    async def _generate_image_stream(
        self,
        request: Request,
        command_id: CommandId,
        num_images: int,
        response_format: str,
    ) -> AsyncGenerator[str, None]:
        """Generate SSE stream of partial and final images."""
        # Track chunks: {(image_index, is_partial): {chunk_index: data}}
        image_chunks: dict[tuple[int, bool], dict[int, str]] = {}
        image_total_chunks: dict[tuple[int, bool], int] = {}
        image_metadata: dict[tuple[int, bool], tuple[int | None, int | None]] = {}
        images_complete = 0

        try:
            self._image_generation_queues[command_id], recv = channel[
                ImageChunk | ErrorChunk
            ]()

            with recv as chunks:
                async for chunk in chunks:
                    if chunk.finish_reason == "error":
                        error_response = ErrorResponse(
                            error=ErrorInfo(
                                message=chunk.error_message or "Internal server error",
                                type="InternalServerError",
                                code=500,
                            )
                        )
                        yield f"data: {error_response.model_dump_json()}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    key = (chunk.image_index, chunk.is_partial)

                    if key not in image_chunks:
                        image_chunks[key] = {}
                        image_total_chunks[key] = chunk.total_chunks
                        image_metadata[key] = (
                            chunk.partial_index,
                            chunk.total_partials,
                        )

                    image_chunks[key][chunk.chunk_index] = chunk.data

                    # Check if this image is complete
                    if len(image_chunks[key]) == image_total_chunks[key]:
                        full_data = "".join(
                            image_chunks[key][i] for i in range(len(image_chunks[key]))
                        )

                        partial_idx, total_partials = image_metadata[key]

                        if chunk.is_partial:
                            # Yield partial image event (always use b64_json for partials)
                            event_data = {
                                "type": "partial",
                                "image_index": chunk.image_index,
                                "partial_index": partial_idx,
                                "total_partials": total_partials,
                                "format": str(chunk.format),
                                "data": {
                                    "b64_json": full_data
                                    if response_format == "b64_json"
                                    else None,
                                },
                            }
                            yield f"data: {json.dumps(event_data)}\n\n"
                        else:
                            # Final image
                            if response_format == "url":
                                image_bytes = base64.b64decode(full_data)
                                content_type = _format_to_content_type(chunk.format)
                                stored = self._image_store.store(
                                    image_bytes, content_type
                                )
                                url = self._build_image_url(request, stored.image_id)
                                event_data = {
                                    "type": "final",
                                    "image_index": chunk.image_index,
                                    "format": str(chunk.format),
                                    "data": {"url": url},
                                }
                            else:
                                event_data = {
                                    "type": "final",
                                    "image_index": chunk.image_index,
                                    "format": str(chunk.format),
                                    "data": {"b64_json": full_data},
                                }
                            yield f"data: {json.dumps(event_data)}\n\n"
                            images_complete += 1

                            if images_complete >= num_images:
                                yield "data: [DONE]\n\n"
                                break

                        # Clean up completed image chunks
                        del image_chunks[key]
                        del image_total_chunks[key]
                        del image_metadata[key]

        except anyio.get_cancelled_exc_class():
            command = TaskCancelled(cancelled_command_id=command_id)
            with anyio.CancelScope(shield=True):
                await self.command_sender.send(
                    ForwarderCommand(origin=self._system_id, command=command)
                )
            raise
        finally:
            await self._send(TaskFinished(finished_command_id=command_id))
            if command_id in self._image_generation_queues:
                del self._image_generation_queues[command_id]

    async def _collect_image_chunks(
        self,
        request: Request | None,
        command_id: CommandId,
        num_images: int,
        response_format: str,
        capture_stats: bool = False,
    ) -> tuple[list[ImageData], ImageGenerationStats | None]:
        """Collect image chunks and optionally capture stats."""
        # Track chunks per image: {image_index: {chunk_index: data}}
        # Only track non-partial (final) images
        image_chunks: dict[int, dict[int, str]] = {}
        image_total_chunks: dict[int, int] = {}
        image_formats: dict[int, Literal["png", "jpeg", "webp"] | None] = {}
        images_complete = 0
        stats: ImageGenerationStats | None = None

        try:
            self._image_generation_queues[command_id], recv = channel[
                ImageChunk | ErrorChunk
            ]()

            while images_complete < num_images:
                with recv as chunks:
                    async for chunk in chunks:
                        if chunk.finish_reason == "error":
                            raise HTTPException(
                                status_code=500,
                                detail=chunk.error_message or "Internal server error",
                            )

                        if chunk.is_partial:
                            continue

                        if chunk.image_index not in image_chunks:
                            image_chunks[chunk.image_index] = {}
                            image_total_chunks[chunk.image_index] = chunk.total_chunks
                            image_formats[chunk.image_index] = chunk.format

                        image_chunks[chunk.image_index][chunk.chunk_index] = chunk.data

                        if capture_stats and chunk.stats is not None:
                            stats = chunk.stats

                        if (
                            len(image_chunks[chunk.image_index])
                            == image_total_chunks[chunk.image_index]
                        ):
                            images_complete += 1

                        if images_complete >= num_images:
                            break

            images: list[ImageData] = []
            for image_idx in range(num_images):
                chunks_dict = image_chunks[image_idx]
                full_data = "".join(chunks_dict[i] for i in range(len(chunks_dict)))
                if response_format == "url" and request is not None:
                    image_bytes = base64.b64decode(full_data)
                    content_type = _format_to_content_type(image_formats.get(image_idx))
                    stored = self._image_store.store(image_bytes, content_type)
                    url = self._build_image_url(request, stored.image_id)
                    images.append(ImageData(b64_json=None, url=url))
                else:
                    images.append(
                        ImageData(
                            b64_json=full_data
                            if response_format == "b64_json"
                            else None,
                            url=None,
                        )
                    )

            return (images, stats if capture_stats else None)
        except anyio.get_cancelled_exc_class():
            command = TaskCancelled(cancelled_command_id=command_id)
            with anyio.CancelScope(shield=True):
                await self.command_sender.send(
                    ForwarderCommand(origin=self._system_id, command=command)
                )
            raise
        finally:
            await self._send(TaskFinished(finished_command_id=command_id))
            if command_id in self._image_generation_queues:
                del self._image_generation_queues[command_id]

    async def _collect_image_generation(
        self,
        request: Request,
        command_id: CommandId,
        num_images: int,
        response_format: str,
    ) -> ImageGenerationResponse:
        """Collect all image chunks (non-streaming) and return a single response."""
        images, _ = await self._collect_image_chunks(
            request, command_id, num_images, response_format, capture_stats=False
        )
        return ImageGenerationResponse(data=images)

    async def _collect_image_generation_with_stats(
        self,
        request: Request | None,
        command_id: CommandId,
        num_images: int,
        response_format: str,
    ) -> BenchImageGenerationResponse:
        sampler = PowerSampler(get_node_system=lambda: self.state.node_system)
        images: list[ImageData] = []
        stats: ImageGenerationStats | None = None
        async with anyio.create_task_group() as tg:
            tg.start_soon(sampler.run)
            images, stats = await self._collect_image_chunks(
                request, command_id, num_images, response_format, capture_stats=True
            )
            tg.cancel_scope.cancel()
        return BenchImageGenerationResponse(
            data=images, generation_stats=stats, power_usage=sampler.result()
        )

    async def bench_image_generations(
        self, request: Request, payload: BenchImageGenerationTaskParams
    ) -> BenchImageGenerationResponse:
        payload = payload.model_copy(
            update={
                "model": await self._validate_image_model(ModelId(payload.model)),
                "stream": False,
                "partial_images": 0,
                "advanced_params": _ensure_seed(payload.advanced_params),
            }
        )

        command = ImageGeneration(
            task_params=payload,
        )
        await self._send(command)

        return await self._collect_image_generation_with_stats(
            request=request,
            command_id=command.command_id,
            num_images=payload.n or 1,
            response_format=payload.response_format or "b64_json",
        )

    async def _send_image_edits_command(
        self,
        image: UploadFile,
        prompt: str,
        model: ModelId,
        n: int,
        size: ImageSize,
        response_format: Literal["url", "b64_json"],
        input_fidelity: Literal["low", "high"],
        stream: bool,
        partial_images: int,
        bench: bool,
        quality: Literal["high", "medium", "low"],
        output_format: Literal["png", "jpeg", "webp"],
        advanced_params: AdvancedImageParams | None,
    ) -> ImageEdits:
        """Prepare and send an image edits command with chunked image upload."""
        resolved_model = await self._validate_image_model(model)
        advanced_params = _ensure_seed(advanced_params)

        image_content = await image.read()
        image_data = base64.b64encode(image_content).decode("utf-8")

        image_strength = 0.7 if input_fidelity == "high" else 0.3

        data_chunks = [
            image_data[i : i + CAI_MAX_CHUNK_SIZE]
            for i in range(0, len(image_data), CAI_MAX_CHUNK_SIZE)
        ]
        total_chunks = len(data_chunks)

        command = ImageEdits(
            task_params=ImageEditsTaskParams(
                image_data="",
                total_input_chunks=total_chunks,
                prompt=prompt,
                model=resolved_model,
                n=n,
                size=size,
                response_format=response_format,
                image_strength=image_strength,
                stream=stream,
                partial_images=partial_images,
                bench=bench,
                quality=quality,
                output_format=output_format,
                advanced_params=advanced_params,
            ),
        )

        logger.info(
            f"Sending input image: {len(image_data)} bytes in {total_chunks} chunks"
        )
        for chunk_index, chunk_data in enumerate(data_chunks):
            await self._send(
                SendInputChunk(
                    chunk=InputImageChunk(
                        model=resolved_model,
                        command_id=command.command_id,
                        data=chunk_data,
                        chunk_index=chunk_index,
                        total_chunks=total_chunks,
                    )
                )
            )

        await self._send(command)
        return command

    async def image_edits(
        self,
        request: Request,
        image: UploadFile = File(...),  # noqa: B008
        prompt: str = Form(...),
        model: str = Form(...),
        n: int = Form(1),
        size: str | None = Form(None),
        response_format: Literal["url", "b64_json"] = Form("b64_json"),
        input_fidelity: Literal["low", "high"] = Form("low"),
        stream: str = Form("false"),
        partial_images: str = Form("0"),
        quality: Literal["high", "medium", "low"] = Form("medium"),
        output_format: Literal["png", "jpeg", "webp"] = Form("png"),
        advanced_params: str | None = Form(None),
    ) -> ImageGenerationResponse | StreamingResponse:
        """Handle image editing requests (img2img)."""
        # Parse string form values to proper types
        stream_bool = stream.lower() in ("true", "1", "yes")
        partial_images_int = int(partial_images) if partial_images.isdigit() else 0

        parsed_advanced_params: AdvancedImageParams | None = None
        if advanced_params:
            with contextlib.suppress(Exception):
                parsed_advanced_params = AdvancedImageParams.model_validate_json(
                    advanced_params
                )

        command = await self._send_image_edits_command(
            image=image,
            prompt=prompt,
            model=ModelId(model),
            n=n,
            size=normalize_image_size(size),
            response_format=response_format,
            input_fidelity=input_fidelity,
            stream=stream_bool,
            partial_images=partial_images_int,
            bench=False,
            quality=quality,
            output_format=output_format,
            advanced_params=parsed_advanced_params,
        )

        if stream_bool and partial_images_int > 0:
            return StreamingResponse(
                self._generate_image_stream(
                    request=request,
                    command_id=command.command_id,
                    num_images=n,
                    response_format=response_format,
                ),
                media_type="text/event-stream",
            )

        return await self._collect_image_generation(
            request=request,
            command_id=command.command_id,
            num_images=n,
            response_format=response_format,
        )

    async def bench_image_edits(
        self,
        request: Request,
        image: UploadFile = File(...),  # noqa: B008
        prompt: str = Form(...),
        model: str = Form(...),
        n: int = Form(1),
        size: str | None = Form(None),
        response_format: Literal["url", "b64_json"] = Form("b64_json"),
        input_fidelity: Literal["low", "high"] = Form("low"),
        quality: Literal["high", "medium", "low"] = Form("medium"),
        output_format: Literal["png", "jpeg", "webp"] = Form("png"),
        advanced_params: str | None = Form(None),
    ) -> BenchImageGenerationResponse:
        """Handle benchmark image editing requests with generation stats."""
        parsed_advanced_params: AdvancedImageParams | None = None
        if advanced_params:
            with contextlib.suppress(Exception):
                parsed_advanced_params = AdvancedImageParams.model_validate_json(
                    advanced_params
                )

        command = await self._send_image_edits_command(
            image=image,
            prompt=prompt,
            model=ModelId(model),
            n=n,
            size=normalize_image_size(size),
            response_format=response_format,
            input_fidelity=input_fidelity,
            stream=False,
            partial_images=0,
            bench=True,
            quality=quality,
            output_format=output_format,
            advanced_params=parsed_advanced_params,
        )

        return await self._collect_image_generation_with_stats(
            request=request,
            command_id=command.command_id,
            num_images=n,
            response_format=response_format,
        )

    async def claude_messages(
        self, payload: ClaudeMessagesRequest
    ) -> ClaudeMessagesResponse | StreamingResponse:
        """Claude Messages API - adapter."""
        task_params = await claude_request_to_text_generation(payload)
        resolved_model = await self._resolve_and_validate_text_model(
            ModelId(task_params.model)
        )
        task_params = task_params.model_copy(update={"model": resolved_model})

        command = await self._send_text_generation_with_images(task_params)

        if payload.stream:
            return StreamingResponse(
                with_sse_keepalive(
                    generate_claude_stream(
                        command.command_id,
                        payload.model,
                        self._token_chunk_stream(command.command_id),
                    ),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return StreamingResponse(
                collect_claude_response(
                    command.command_id,
                    payload.model,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/json",
            )

    async def openai_responses(
        self, payload: ResponsesRequest
    ) -> ResponsesResponse | StreamingResponse:
        """OpenAI Responses API."""
        task_params = await responses_request_to_text_generation(payload)
        resolved_model = await self._resolve_and_validate_text_model(task_params.model)
        task_params = task_params.model_copy(update={"model": resolved_model})

        command = await self._send_text_generation_with_images(task_params)

        if payload.stream:
            return StreamingResponse(
                with_sse_keepalive(
                    generate_responses_stream(
                        command.command_id,
                        payload.model,
                        self._token_chunk_stream(command.command_id),
                    ),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )

        else:
            return StreamingResponse(
                collect_responses_response(
                    command.command_id,
                    payload.model,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/json",
            )

    async def _ollama_root(self) -> JSONResponse:
        """Respond to HEAD / from Ollama CLI connectivity checks."""
        return JSONResponse(content="Ollama is running")

    async def ollama_chat(
        self, request: Request
    ) -> OllamaChatResponse | StreamingResponse:
        """Ollama Chat API — accepts JSON regardless of Content-Type."""
        body = await request.body()
        payload = OllamaChatRequest.model_validate_json(body)
        task_params = ollama_request_to_text_generation(payload)
        resolved_model = await self._resolve_and_validate_text_model(
            ModelId(task_params.model)
        )
        task_params = task_params.model_copy(update={"model": resolved_model})

        command = await self._send_text_generation_with_images(task_params)

        if payload.stream:
            return StreamingResponse(
                generate_ollama_chat_stream(
                    command.command_id,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return StreamingResponse(
                collect_ollama_chat_response(
                    command.command_id,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/json",
            )

    async def ollama_generate(
        self, request: Request
    ) -> OllamaGenerateResponse | StreamingResponse:
        """Ollama Generate API — accepts JSON regardless of Content-Type."""
        body = await request.body()
        payload = OllamaGenerateRequest.model_validate_json(body)
        task_params = ollama_generate_request_to_text_generation(payload)
        resolved_model = await self._resolve_and_validate_text_model(
            ModelId(task_params.model)
        )
        task_params = task_params.model_copy(update={"model": resolved_model})

        command = await self._send_text_generation_with_images(task_params)

        if payload.stream:
            return StreamingResponse(
                generate_ollama_generate_stream(
                    command.command_id,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return StreamingResponse(
                collect_ollama_generate_response(
                    command.command_id,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/json",
            )

    async def ollama_tags(self) -> OllamaTagsResponse:
        """Returns list of models in Ollama tags format. We return the downloaded ones only."""

        def none_if_empty(value: str) -> str | None:
            return value or None

        downloaded_model_ids: set[str] = set()
        for node_downloads in self.state.downloads.values():
            for dl in node_downloads:
                if isinstance(dl, DownloadCompleted):
                    downloaded_model_ids.add(dl.shard_metadata.model_card.model_id)

        cards = [
            c for c in await get_model_cards() if c.model_id in downloaded_model_ids
        ]

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return OllamaTagsResponse(
            models=[
                OllamaModelTag(
                    name=str(card.model_id),
                    model=str(card.model_id),
                    modified_at=now,
                    size=card.storage_size.in_bytes,
                    digest="sha256:000000000000",
                    details=OllamaModelDetails(
                        family=none_if_empty(card.family),
                        quantization_level=none_if_empty(card.quantization),
                    ),
                )
                for card in cards
            ]
        )

    async def ollama_show(self, request: Request) -> OllamaShowResponse:
        """Returns model information in Ollama show format."""
        body = await request.body()
        payload = OllamaShowRequest.model_validate_json(body)
        model_name = payload.name or payload.model
        if not model_name:
            raise HTTPException(status_code=400, detail="name or model is required")
        try:
            card = await ModelCard.load(ModelId(model_name))
        except Exception as exc:
            raise HTTPException(
                status_code=404, detail=f"Model not found: {model_name}"
            ) from exc

        return OllamaShowResponse(
            modelfile=f"FROM {card.model_id}",
            template="{{ .Prompt }}",
            details=OllamaModelDetails(
                family=card.family or None,
                quantization_level=card.quantization or None,
            ),
        )

    async def ollama_ps(self) -> OllamaPsResponse:
        """Returns list of running models (active instances)."""
        models: list[OllamaPsModel] = []
        seen: set[str] = set()
        for instance in self.state.instances.values():
            model_id = str(instance.shard_assignments.model_id)
            if model_id in seen:
                continue
            seen.add(model_id)
            models.append(
                OllamaPsModel(
                    name=model_id,
                    model=model_id,
                    size=0,
                )
            )
        return OllamaPsResponse(models=models)

    async def ollama_version(self) -> dict[str, str]:
        """Returns version information for Ollama API compatibility."""
        return {"version": "cai v1.0"}

    def _calculate_total_available_memory(self) -> Memory:
        """Calculate total available memory across all nodes in bytes."""
        total_available = Memory()

        for memory in self.state.node_memory.values():
            total_available += memory.ram_available

        return total_available

    async def get_models(self, status: str | None = Query(default=None)) -> ModelList:
        """Returns list of available models, optionally filtered by being downloaded."""
        all_cards = await get_model_cards()
        llama_cards = [
            card for card in all_cards if card.inference_backend == InferenceBackend.LlamaCpp
        ]
        cards_by_id = {str(card.model_id): card for card in llama_cards}
        curated_models = tuple(curated_model_registry())
        private_execution_ids = {
            model.execution_model_id for model in curated_models if model.private_network
        }
        visible_curated_models = [
            model
            for model in curated_models
            if model.private_network or model.model_id not in private_execution_ids
        ]
        selected_cards: list[tuple[ModelCard, str, str | None]] = []
        selected_ids: set[str] = set()

        for model in visible_curated_models:
            card = cards_by_id.get(model.model_id) or cards_by_id.get(model.execution_model_id)
            if card is None:
                continue
            selected_cards.append((card, model.model_id, model.display_name))
            selected_ids.add(model.model_id)

        for card in llama_cards:
            card_id = str(card.model_id)
            if card.is_custom and card_id not in selected_ids:
                selected_cards.append((card, card_id, None))
                selected_ids.add(card_id)

        if status == "downloaded":
            downloaded_model_ids: set[str] = set()
            for node_downloads in self.state.downloads.values():
                for dl in node_downloads:
                    if isinstance(dl, DownloadCompleted):
                        downloaded_model_ids.add(str(dl.shard_metadata.model_card.model_id))

            aliases_by_model_id = {
                model.model_id: {
                    model.model_id,
                    model.execution_model_id,
                    *model.runtime_model_ids,
                }
                for model in curated_models
            }
            selected_cards = [
                item
                for item in selected_cards
                if aliases_by_model_id.get(item[1], {item[1]}) & downloaded_model_ids
            ]

        return ModelList(
            data=[
                ModelListModel(
                    id=model_id,
                    hugging_face_id=model_id,
                    name=display_name or ModelId(model_id).short(),
                    description="",
                    tags=[],
                    storage_size_megabytes=card.storage_size.in_mb,
                    supports_tensor=card.supports_tensor,
                    tasks=[task.value for task in card.tasks],
                    is_custom=card.is_custom,
                    inference_backend=card.inference_backend.value,
                    family=card.family,
                    quantization=card.quantization,
                    base_model=card.base_model,
                    capabilities=card.capabilities,
                    context_length=card.context_length,
                    gguf_architecture=card.gguf_architecture,
                    shard_compatibility=card.shard_compatibility,
                    layer_range_supported=card.layer_range_supported,
                    model_package_manifest_url=card.model_package_manifest_url,
                    model_package_catalog_id=card.model_package_catalog_id,
                    model_package_version=card.model_package_version,
                    layer_range_probe_abi=card.layer_range_probe_abi,
                    layer_range_probe_report=card.layer_range_probe_report,
                    layer_range_equivalence_probe_report=(
                        card.layer_range_equivalence_probe_report
                    ),
                    state_format=card.state_format,
                    activation_state_format=card.activation_state_format,
                    decode_state_format=card.decode_state_format,
                    shard_compatibility_reason=card.shard_compatibility_reason,
                )
                for card, model_id, display_name in selected_cards
            ]
        )

    async def add_custom_model(self, payload: AddCustomModelParams) -> ModelListModel:
        """Add a custom model from HuggingFace or from a local model directory."""
        if payload.local_path:
            resolved_model_id = payload.model_id or derive_custom_model_id_from_local_path(
                payload.local_path
            )
            existing = get_card(resolved_model_id)
            if existing is not None and not existing.is_custom:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Model ID '{resolved_model_id}' already exists in the public catalog. "
                        "Choose a different custom model ID."
                    ),
                )
            try:
                card = await ModelCard.load_from_local_directory(
                    resolved_model_id,
                    payload.local_path,
                )
                if not _model_card_supported_for_cai_gguf_compute(card):
                    raise HTTPException(
                        status_code=400,
                        detail=_unsupported_gguf_model_detail(card),
                    )
                set_custom_model_local_path(card.model_id, payload.local_path)
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to add local model: {exc}",
                ) from exc
        else:
            try:
                card = await ModelCard.fetch_from_hf(payload.model_id)
            except Exception as exc:
                raise HTTPException(
                    status_code=400, detail=f"Failed to fetch model: {exc}"
                ) from exc
            if not _model_card_supported_for_cai_gguf_compute(card):
                raise HTTPException(
                    status_code=400,
                    detail=_unsupported_gguf_model_detail(card),
                )

        await self.command_sender.send(
            ForwarderCommand(
                origin=self._system_id,
                command=AddCustomModelCard(model_card=card),
            )
        )

        # Immediately update the local cache so the subsequent GET /models
        # returns the new model without waiting for the event round-trip.
        add_to_card_cache(card)

        return ModelListModel(
            id=card.model_id,
            hugging_face_id="" if payload.local_path else card.model_id,
            name=card.model_id.short(),
            description="",
            tags=[],
            storage_size_megabytes=int(card.storage_size.in_mb),
            supports_tensor=card.supports_tensor,
            tasks=[task.value for task in card.tasks],
            is_custom=card.is_custom,
            inference_backend=card.inference_backend.value,
            family=card.family,
            quantization=card.quantization,
            base_model=card.base_model,
            capabilities=card.capabilities,
            context_length=card.context_length,
            gguf_architecture=card.gguf_architecture,
            shard_compatibility=card.shard_compatibility,
            layer_range_supported=card.layer_range_supported,
            layer_range_probe_abi=card.layer_range_probe_abi,
            layer_range_probe_report=card.layer_range_probe_report,
            layer_range_equivalence_probe_report=(
                card.layer_range_equivalence_probe_report
            ),
            state_format=card.state_format,
            activation_state_format=card.activation_state_format,
            decode_state_format=card.decode_state_format,
            shard_compatibility_reason=card.shard_compatibility_reason,
        )

    async def delete_custom_model(self, model_id: ModelId) -> JSONResponse:
        """Delete a user-added custom model card and sync deletion across the cluster."""
        card = get_card(model_id)
        if card is None or not card.is_custom:
            raise HTTPException(status_code=404, detail="Custom model card not found")

        await self.command_sender.send(
            ForwarderCommand(
                origin=self._system_id,
                command=DeleteCustomModelCard(model_id=model_id),
            )
        )

        return JSONResponse(
            {"message": "Model card deleted", "model_id": str(model_id)}
        )

    async def search_models(
        self, query: str = "", limit: int = 20
    ) -> list[HuggingFaceSearchResult]:
        """Search HuggingFace Hub and prefer GGUF repositories for llama.cpp."""
        from huggingface_hub import ModelInfo, list_models

        def _to_results(models: Iterable[ModelInfo]) -> list[HuggingFaceSearchResult]:
            return [
                HuggingFaceSearchResult(
                    id=m.id,
                    author=m.author or "",
                    downloads=m.downloads or 0,
                    likes=m.likes or 0,
                    last_modified=str(m.last_modified or ""),
                    tags=list(m.tags or []),
                )
                for m in models
            ]

        search_terms = f"{query} GGUF".strip() if query else "GGUF"
        gguf_candidates = list(
            list_models(
                search=search_terms or None,
                sort="downloads",
                limit=max(limit * 5, 20),
            )
        )
        gguf_results = [
            model
            for model in gguf_candidates
            if _model_info_supported_for_cai_gguf_compute(model)
        ]

        if len(gguf_results) < limit and query:
            fallback_candidates = list(
                list_models(
                    search=query or None,
                    sort="downloads",
                    limit=max(limit * 8, 40),
                )
            )
            seen_ids = {str(getattr(model, "id", "")) for model in gguf_results}
            for model in fallback_candidates:
                model_id = str(getattr(model, "id", ""))
                if (
                    model_id in seen_ids
                    or not _model_info_supported_for_cai_gguf_compute(model)
                ):
                    continue
                gguf_results.append(model)
                seen_ids.add(model_id)
                if len(gguf_results) >= limit:
                    break

        return _to_results(gguf_results[:limit])

    async def run(self):
        shutdown_ev = anyio.Event()

        try:
            async with self._tg as tg:
                logger.info("Starting API")
                tg.start_soon(self._apply_state)
                tg.start_soon(self._pause_on_new_election)
                tg.start_soon(self._cleanup_expired_images)
                if self.cai_owned_transport_message_receiver is not None:
                    tg.start_soon(self._apply_cai_owned_transport_overlay_messages)
                print_startup_banner(self.port)
                tg.start_soon(self.run_api, shutdown_ev)
                try:
                    await anyio.sleep_forever()
                finally:
                    with anyio.CancelScope(shield=True):
                        shutdown_ev.set()
        finally:
            self._event_log.close()
            self.command_sender.close()
            self.event_receiver.close()
            if self.cai_owned_transport_message_receiver is not None:
                self.cai_owned_transport_message_receiver.close()

    async def _apply_cai_owned_transport_overlay_messages(self):
        receiver = self.cai_owned_transport_message_receiver
        if receiver is None:
            return
        with receiver as messages:
            async for message in messages:
                if str(message.target_node_id).strip() != str(self.node_id):
                    continue
                try:
                    result = self._handle_cai_owned_transport_overlay_message(message)
                    logger.info(
                        "CAI-owned transport overlay message {} applied on {}: {}",
                        message.message_id,
                        self.node_id,
                        result.get("status") if isinstance(result, dict) else "ok",
                    )
                except Exception as exc:
                    logger.opt(exception=exc).warning(
                        "Failed to apply CAI-owned transport overlay message {} "
                        "({}) on {}",
                        message.message_id,
                        message.kind,
                        self.node_id,
                    )

    def _handle_cai_owned_transport_overlay_message(
        self,
        message: CaiOwnedTransportOverlayMessage,
    ) -> dict[str, Any]:
        service = self._get_cai_service()
        if message.kind == "session_offer":
            return service.accept_cai_owned_transport_session_offer(
                message.session_id,
                message.payload,
            )
        if message.kind == "batch_envelope":
            return service.record_cai_owned_transport_batch_envelope(
                message.session_id,
                message.payload,
            )
        if message.kind == "shard_receipt":
            return service.record_cai_owned_transport_shard_receipt(
                message.session_id,
                message.payload,
            )
        if message.kind == "completion_notice":
            return service.accept_cai_owned_transport_completion_notice(
                message.session_id,
                message.payload,
            )
        raise ValueError(f"Unsupported CAI-owned transport overlay kind: {message.kind}")

    async def run_api(self, ev: anyio.Event):
        cfg = Config()
        cfg.bind = [f"0.0.0.0:{self.port}"]
        # nb: shared.logging needs updating if any of this changes
        cfg.accesslog = None
        cfg.errorlog = "-"
        cfg.logger_class = InterceptLogger
        with anyio.CancelScope(shield=True):
            await serve(
                cast(ASGIFramework, self.app),
                cfg,
                shutdown_trigger=ev.wait,
            )

    async def _apply_state(self):
        with self.event_receiver as events:
            async for i_event in events:
                self._event_log.append(i_event.event)
                self.state = apply(self.state, i_event)
                self._cache_local_node_state_from(self.state)
                self.state = self._overlay_local_node_state(self.state)
                event = i_event.event

                if isinstance(event, ChunkGenerated):
                    if queue := self._image_generation_queues.get(
                        event.command_id, None
                    ):
                        assert isinstance(event.chunk, ImageChunk)
                        try:
                            await queue.send(event.chunk)
                        except (BrokenResourceError, ClosedResourceError):
                            self._image_generation_queues.pop(event.command_id, None)
                    if queue := self._text_generation_queues.get(
                        event.command_id, None
                    ):
                        assert not isinstance(event.chunk, ImageChunk)
                        logger.debug(
                            "api apply_state forwarding chunk command_id={} chunk_type={} finish_reason={}",
                            event.command_id,
                            type(event.chunk).__name__,
                            getattr(event.chunk, "finish_reason", None),
                        )
                        try:
                            await queue.send(event.chunk)
                        except (BrokenResourceError, ClosedResourceError):
                            self._text_generation_queues.pop(event.command_id, None)
                if isinstance(event, InstanceDeleted):
                    self._close_streams_for_instance(event.instance_id)
                if isinstance(event, TracesMerged):
                    self._save_merged_trace(event)

    def _close_streams_for_instance(self, instance_id: InstanceId) -> None:
        """Close any active generation streams for commands running on the given instance."""
        for task in self.state.tasks.values():
            if task.instance_id != instance_id:
                continue
            if not isinstance(
                task, (TextGenerationTask, ImageGenerationTask, ImageEditsTask)
            ):
                continue
            if sender := self._text_generation_queues.pop(task.command_id, None):
                sender.close()
            if sender := self._image_generation_queues.pop(task.command_id, None):
                sender.close()

    def _save_merged_trace(self, event: TracesMerged) -> None:
        traces = [
            TraceEvent(
                name=t.name,
                start_us=t.start_us,
                duration_us=t.duration_us,
                rank=t.rank,
                category=t.category,
            )
            for t in event.traces
        ]
        output_path = CAI_TRACING_CACHE_DIR / f"trace_{event.task_id}.json"
        export_trace(traces, output_path)
        logger.debug(f"Saved merged trace to {output_path}")

    async def _pause_on_new_election(self):
        with self.election_receiver as ems:
            async for message in ems:
                if message.clock > self.last_completed_election:
                    self.paused = True

    async def _cleanup_expired_images(self):
        """Periodically clean up expired images from the store."""
        cleanup_interval_seconds = 300  # 5 minutes
        while True:
            await anyio.sleep(cleanup_interval_seconds)
            removed = self._image_store.cleanup_expired()
            if removed > 0:
                logger.debug(f"Cleaned up {removed} expired images")

    async def _send(self, command: Command):
        timeout_seconds = _api_command_send_timeout_seconds()
        try:
            with anyio.fail_after(timeout_seconds):
                while self.paused:
                    await self.paused_ev.wait()
                await self.command_sender.send(
                    ForwarderCommand(origin=self._system_id, command=command)
                )
        except TimeoutError as exc:
            command_name = type(command).__name__
            logger.warning(
                "Timed out dispatching API command command={} timeout_seconds={} paused={}",
                command_name,
                timeout_seconds,
                self.paused,
            )
            pause_note = (
                " while API is paused for master election"
                if self.paused
                else ""
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "Timed out dispatching command to CAI control plane"
                    f"{pause_note}. Retry after the node finishes synchronizing."
                ),
            ) from exc

    async def _send_download(self, command: DownloadCommand):
        await self.download_command_sender.send(
            ForwarderDownloadCommand(origin=self._system_id, command=command)
        )

    async def start_download(
        self, payload: StartDownloadParams
    ) -> StartDownloadResponse:
        command = StartDownload(
            target_node_id=payload.target_node_id,
            shard_metadata=payload.shard_metadata,
        )
        await self._send_download(command)
        return StartDownloadResponse(command_id=command.command_id)

    async def delete_download(
        self, node_id: NodeId, model_id: ModelId
    ) -> DeleteDownloadResponse:
        command = DeleteDownload(
            target_node_id=node_id,
            model_id=ModelId(model_id),
        )
        await self._send_download(command)
        return DeleteDownloadResponse(command_id=command.command_id)

    async def cancel_download(
        self,
        payload: CancelDownloadParams,
    ) -> CancelDownloadResponse:
        command = CancelDownload(
            target_node_id=payload.target_node_id,
            model_id=payload.model_id,
        )
        await self._send_download(command)
        return CancelDownloadResponse(command_id=command.command_id)

    @staticmethod
    def _get_trace_path(task_id: str) -> Path:
        trace_path = CAI_TRACING_CACHE_DIR / f"trace_{task_id}.json"
        if not trace_path.resolve().is_relative_to(CAI_TRACING_CACHE_DIR.resolve()):
            raise HTTPException(status_code=400, detail=f"Invalid task ID: {task_id}")
        return trace_path

    async def list_traces(self) -> TraceListResponse:
        traces: list[TraceListItem] = []

        for trace_file in sorted(
            CAI_TRACING_CACHE_DIR.glob("trace_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            # Extract task_id from filename (trace_{task_id}.json)
            task_id = trace_file.stem.removeprefix("trace_")
            stat = trace_file.stat()
            created_at = datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat()
            traces.append(
                TraceListItem(
                    task_id=task_id,
                    created_at=created_at,
                    file_size=stat.st_size,
                )
            )

        return TraceListResponse(traces=traces)

    async def get_trace(self, task_id: str) -> TraceResponse:
        trace_path = self._get_trace_path(task_id)

        if not trace_path.exists():
            raise HTTPException(status_code=404, detail=f"Trace not found: {task_id}")

        trace_events = load_trace_file(trace_path)

        return TraceResponse(
            task_id=task_id,
            traces=[
                TraceEventResponse(
                    name=event.name,
                    start_us=event.start_us,
                    duration_us=event.duration_us,
                    rank=event.rank,
                    category=event.category,
                )
                for event in trace_events
            ],
        )

    async def get_trace_stats(self, task_id: str) -> TraceStatsResponse:
        trace_path = self._get_trace_path(task_id)

        if not trace_path.exists():
            raise HTTPException(status_code=404, detail=f"Trace not found: {task_id}")

        trace_events = load_trace_file(trace_path)
        stats = compute_stats(trace_events)

        return TraceStatsResponse(
            task_id=task_id,
            total_wall_time_us=stats.total_wall_time_us,
            by_category={
                category: TraceCategoryStats(
                    total_us=cat_stats.total_us,
                    count=cat_stats.count,
                    min_us=cat_stats.min_us,
                    max_us=cat_stats.max_us,
                    avg_us=cat_stats.avg_us,
                )
                for category, cat_stats in stats.by_category.items()
            },
            by_rank={
                rank: TraceRankStats(
                    by_category={
                        category: TraceCategoryStats(
                            total_us=cat_stats.total_us,
                            count=cat_stats.count,
                            min_us=cat_stats.min_us,
                            max_us=cat_stats.max_us,
                            avg_us=cat_stats.avg_us,
                        )
                        for category, cat_stats in rank_stats.items()
                    }
                )
                for rank, rank_stats in stats.by_rank.items()
            },
        )

    async def get_trace_raw(self, task_id: str) -> FileResponse:
        trace_path = self._get_trace_path(task_id)

        if not trace_path.exists():
            raise HTTPException(status_code=404, detail=f"Trace not found: {task_id}")

        return FileResponse(
            path=trace_path,
            media_type="application/json",
            filename=f"trace_{task_id}.json",
        )

    async def delete_traces(self, request: DeleteTracesRequest) -> DeleteTracesResponse:
        deleted: list[str] = []
        not_found: list[str] = []
        for task_id in request.task_ids:
            trace_path = self._get_trace_path(task_id)
            if trace_path.exists():
                trace_path.unlink()
                deleted.append(task_id)
            else:
                not_found.append(task_id)
        return DeleteTracesResponse(deleted=deleted, not_found=not_found)

    async def get_onboarding(self) -> JSONResponse:
        return JSONResponse({"completed": ONBOARDING_COMPLETE_FILE.exists()})

    async def complete_onboarding(self) -> JSONResponse:
        ONBOARDING_COMPLETE_FILE.parent.mkdir(parents=True, exist_ok=True)
        ONBOARDING_COMPLETE_FILE.write_text("true")
        return JSONResponse({"completed": True})
