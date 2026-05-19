# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import hashlib
import shutil
import socket
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

import httpx

from cai.api.types import (
    CompletionTokensDetails,
    PromptTokensDetails,
    ToolCallItem,
    Usage,
)
from cai.download.download_utils import resolve_existing_model
from cai.shared.constants import CAI_CACHE_HOME
from cai.shared.types.chunks import ErrorChunk, TokenChunk, ToolCallChunk
from cai.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from cai.shared.types.tasks import (
    ConnectToGroup,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from cai.shared.types.text_generation import TextGenerationTaskParams
from cai.shared.types.common import Host, NodeId
from cai.shared.types.worker.instances import BoundInstance, MlxRingInstance
from cai.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShuttingDown,
    RunnerShutdown,
    RunnerStatus,
    RunnerWarmingUp,
)
from cai.utils.channels import MpReceiver, MpSender
from cai.worker.runner.bootstrap import logger
from cai.worker.runner.llama_cpp.relay_tunnel import (
    LlamaCppRelayTunnelManager,
    LlamaCppReverseRelayManager,
    _LLAMA_CPP_RPC_HELLO_RESPONSE_SIZE,
    _llama_cpp_rpc_hello_payload,
    _parse_llama_cpp_rpc_hello_response,
)


def _env_positive_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, "") or default)
    except ValueError:
        return default
    return max(1, value)


DEFAULT_READY_TIMEOUT_SECONDS = _env_positive_int(
    "CAI_LLAMA_CPP_READY_TIMEOUT_SECONDS",
    60,
)
DEFAULT_DISTRIBUTED_READY_TIMEOUT_SECONDS = _env_positive_int(
    "CAI_LLAMA_CPP_DISTRIBUTED_READY_TIMEOUT_SECONDS",
    600,
)
DEFAULT_REQUEST_TIMEOUT_SECONDS = 600
DEFAULT_CONTEXT_SIZE = 4096
DEFAULT_GPU_LAYERS = 999
_LLAMA_CPP_RPC_PROTOCOL_FAILURE_MARKERS = (
    "ggml-rpc",
    "remote rpc server crashed",
    "malformed response",
)


def _shared_backend_runtime_module():
    try:
        from cai_compute_chain import cai_llama_cpp_backend_runtime as shared_runtime
    except Exception:
        return None
    return shared_runtime


def _windows_subprocess_flags() -> int:
    shared_runtime = _shared_backend_runtime_module()
    if shared_runtime is not None:
        return shared_runtime.windows_subprocess_creation_flags(os_name=os.name)
    if os.name != "nt":
        return 0
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
        subprocess, "DETACHED_PROCESS", 0x00000008
    )


def _windows_startupinfo() -> subprocess.STARTUPINFO | None:
    shared_runtime = _shared_backend_runtime_module()
    if shared_runtime is not None:
        return shared_runtime.windows_subprocess_startupinfo(os_name=os.name)
    if os.name != "nt":
        return None
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return startupinfo


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _choose_free_port() -> int:
    shared_runtime = _shared_backend_runtime_module()
    if shared_runtime is not None:
        return shared_runtime.choose_loopback_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(size)
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise RuntimeError("connection closed during HELLO")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _probe_llama_cpp_rpc_hello_socket(
    host: str,
    port: int,
    *,
    timeout: float,
) -> str:
    with socket.create_connection((host, int(port)), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(_llama_cpp_rpc_hello_payload())
        response_size = int.from_bytes(_recv_exact(sock, 8), "little")
        if response_size != _LLAMA_CPP_RPC_HELLO_RESPONSE_SIZE:
            raise RuntimeError(f"unexpected HELLO response size {response_size}")
        response = _recv_exact(sock, response_size)
    return _parse_llama_cpp_rpc_hello_response(response)


def _fallback_find_llama_cpp_binary(*, binary_stem: str, env_var: str) -> Path | None:
    env_path = os.environ.get(env_var)
    candidates: list[Path] = []
    suffix = ".exe" if os.name == "nt" else ""
    binary_name = f"{binary_stem}{suffix}"
    if env_path:
        candidates.append(Path(env_path))

    repo_root = _repo_root()
    if os.name == "nt":
        candidates.extend(
            [
                repo_root / ".runtime" / "llama.cpp" / "windows" / "build" / binary_name,
                repo_root / ".runtime" / "llama.cpp" / "windows" / "build" / "bin" / binary_name,
                repo_root / ".runtime" / "llama.cpp" / "build" / "bin" / binary_name,
            ]
        )
    else:
        candidates.extend(
            [
                repo_root / ".runtime" / "llama.cpp" / "wsl" / "build" / "bin" / binary_stem,
                repo_root / ".runtime" / "llama.cpp" / "windows" / "build" / f"{binary_stem}.exe",
                repo_root / ".runtime" / "llama.cpp" / "windows" / "build" / "bin" / f"{binary_stem}.exe",
                repo_root / ".runtime" / "llama.cpp" / "build" / "bin" / binary_stem,
            ]
        )

    which_path = shutil.which(binary_name if os.name == "nt" else binary_stem)
    if which_path:
        candidates.append(Path(which_path))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _find_llama_server_binary() -> Path:
    shared_runtime = _shared_backend_runtime_module()
    binary = None
    if shared_runtime is not None:
        binary = shared_runtime.resolve_llama_cpp_binary_set(
            repo_root=_repo_root(),
            env=dict(os.environ),
            os_name=os.name,
        ).llama_server
    if binary is None:
        binary = _fallback_find_llama_cpp_binary(
            binary_stem="llama-server",
            env_var="CAI_LLAMA_CPP_SERVER",
        )
    if binary is not None:
        return binary

    raise FileNotFoundError(
        "llama-server binary not found. Set CAI_LLAMA_CPP_SERVER or install/build "
        "llama.cpp under CAI/.runtime/llama.cpp."
    )


def _find_llama_rpc_server_binary() -> Path:
    shared_runtime = _shared_backend_runtime_module()
    binary = None
    if shared_runtime is not None:
        binary = shared_runtime.resolve_llama_cpp_binary_set(
            repo_root=_repo_root(),
            env=dict(os.environ),
            os_name=os.name,
        ).rpc_server
    if binary is None:
        binary = _fallback_find_llama_cpp_binary(
            binary_stem="rpc-server",
            env_var="CAI_LLAMA_CPP_RPC_SERVER",
        )
    if binary is not None:
        return binary

    raise FileNotFoundError(
        "rpc-server binary not found. Set CAI_LLAMA_CPP_RPC_SERVER or install/build "
        "llama.cpp under CAI/.runtime/llama.cpp."
    )


def _find_local_gguf_model_path(bound_instance: BoundInstance) -> Path:
    model_path = resolve_existing_model(bound_instance.bound_shard.model_card.model_id)
    if model_path is None:
        raise FileNotFoundError(
            f"Unable to resolve local model path for {bound_instance.bound_shard.model_card.model_id}"
        )
    if model_path.is_file() and model_path.suffix.lower() == ".gguf":
        return model_path

    if model_path.is_dir():
        preferred_filename = bound_instance.bound_shard.model_card.preferred_filename
        if preferred_filename:
            preferred_path = model_path / preferred_filename
            if preferred_path.exists() and preferred_path.is_file():
                return preferred_path
        direct = sorted(model_path.glob("*.gguf"))
        if direct:
            return direct[0]
        recursive = sorted(model_path.glob("**/*.gguf"))
        if recursive:
            return recursive[0]

    raise FileNotFoundError(f"No GGUF file found for model at {model_path}")


def _distributed_prompt_cache_enabled() -> bool:
    return (
        os.environ.get("CAI_LLAMA_CPP_DISTRIBUTED_PROMPT_CACHE", "").strip().lower()
        in {"1", "true", "yes", "on"}
    )


def _cai_owned_transport_journal_enabled() -> bool:
    return (
        os.environ.get("CAI_OWNED_TRANSPORT_JOURNAL_ENABLED", "").strip().lower()
        in {"1", "true", "yes", "on"}
    )


def _cai_owned_transport_offer_submit_enabled() -> bool:
    return (
        os.environ.get("CAI_OWNED_TRANSPORT_OFFER_SUBMIT_ENABLED", "").strip().lower()
        in {"1", "true", "yes", "on"}
    )


def _cai_owned_transport_generation_enabled() -> bool:
    return (
        os.environ.get("CAI_OWNED_TRANSPORT_GENERATION_ENABLED", "").strip().lower()
        in {"1", "true", "yes", "on"}
    )


def _cai_owned_transport_generation_required() -> bool:
    return (
        os.environ.get("CAI_OWNED_TRANSPORT_GENERATION_REQUIRED", "").strip().lower()
        in {"1", "true", "yes", "on"}
    )


def _cai_owned_transport_skip_local_llama_server() -> bool:
    configured = os.environ.get("CAI_OWNED_TRANSPORT_SKIP_LOCAL_LLAMA_SERVER")
    if configured is not None:
        return configured.strip().lower() in {"1", "true", "yes", "on"}
    return (
        _cai_owned_transport_generation_enabled()
        and _cai_owned_transport_generation_required()
    )


def _cai_owned_transport_generation_require_executor_readiness() -> bool:
    return (
        os.environ.get(
            "CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_EXECUTOR_READINESS",
            "1",
        )
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )


def _cai_owned_transport_generation_require_runtime_ready() -> bool:
    return (
        os.environ.get(
            "CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_RUNTIME_READY",
            "",
        )
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )


def _cai_owned_transport_generation_require_shard_readiness() -> bool:
    return (
        os.environ.get(
            "CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_SHARD_READINESS",
            "1",
        )
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )


def _cai_owned_transport_generation_require_data_plane_route() -> bool:
    return (
        os.environ.get(
            "CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_DATA_PLANE_ROUTE",
            "1",
        )
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )


def _cai_owned_transport_generation_require_proven_data_plane_route() -> bool:
    configured = os.environ.get(
        "CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_PROVEN_DATA_PLANE_ROUTE"
    )
    if configured is None:
        return _cai_owned_transport_generation_require_data_plane_route()
    return configured.strip().lower() in {"1", "true", "yes", "on"}


def _cai_owned_transport_generation_minimum_relay_quorum() -> int:
    try:
        return max(
            0,
            int(
                os.environ.get(
                    "CAI_OWNED_TRANSPORT_GENERATION_MIN_RELAY_QUORUM",
                    "0",
                )
            ),
        )
    except ValueError:
        return 0


def _cai_owned_transport_generation_timeout_seconds() -> float:
    try:
        return max(
            1.0,
            float(os.environ.get("CAI_OWNED_TRANSPORT_GENERATION_TIMEOUT_SECONDS", "120")),
        )
    except ValueError:
        return 120.0


def _cai_owned_transport_generation_poll_seconds() -> float:
    try:
        return max(
            0.01,
            float(os.environ.get("CAI_OWNED_TRANSPORT_GENERATION_POLL_SECONDS", "0.25")),
        )
    except ValueError:
        return 0.25


def _cai_owned_transport_payload_compression() -> str | None:
    value = str(os.environ.get("CAI_OWNED_TRANSPORT_PAYLOAD_COMPRESSION") or "").strip()
    if not value or value.lower() in {"0", "false", "no", "off", "none"}:
        return None
    return value


def _cai_owned_transport_payload_chunk_size_bytes() -> int | None:
    raw = str(
        os.environ.get("CAI_OWNED_TRANSPORT_PAYLOAD_CHUNK_SIZE_BYTES") or ""
    ).strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _coerce_usage(payload: object) -> Usage | None:
    if not isinstance(payload, dict):
        return None
    prompt_tokens = int(payload.get("prompt_tokens", 0) or 0)
    completion_tokens = int(payload.get("completion_tokens", 0) or 0)
    total_tokens = int(
        payload.get("total_tokens", prompt_tokens + completion_tokens)
        or (prompt_tokens + completion_tokens)
    )
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_tokens_details=PromptTokensDetails(),
        completion_tokens_details=CompletionTokensDetails(),
    )


def _coerce_finish_reason(raw_finish_reason: object) -> str | None:
    if raw_finish_reason in ("stop", "length", "content_filter"):
        return str(raw_finish_reason)
    return None


def _build_messages(task_params: TextGenerationTaskParams) -> list[dict[str, object]]:
    if task_params.images or task_params.image_hashes or task_params.total_input_chunks > 0:
        raise ValueError("llama.cpp backend does not support multimodal requests yet")

    if task_params.chat_template_messages:
        return [dict(message) for message in task_params.chat_template_messages]

    messages: list[dict[str, object]] = []
    if task_params.instructions:
        messages.append({"role": "system", "content": str(task_params.instructions)})
    for message in task_params.input:
        messages.append({"role": message.role, "content": str(message.content)})
    if not messages:
        messages.append({"role": "user", "content": ""})
    return messages


def _resolve_enable_thinking(task_params: TextGenerationTaskParams) -> bool | None:
    if task_params.enable_thinking is not None:
        return task_params.enable_thinking
    if task_params.reasoning_effort is not None:
        return task_params.reasoning_effort != "none"
    return None


def _is_qwen3_model(model_id: object) -> bool:
    lowered = str(model_id).lower()
    return "qwen3" in lowered and "qwen2.5" not in lowered


def _has_qwen_thinking_directive(text: str) -> bool:
    lowered = text.lower()
    return "/no_think" in lowered or "/think" in lowered


def _with_prefixed_directive(text: str, directive: str) -> str:
    if _has_qwen_thinking_directive(text):
        return text
    return f"{directive}\n{text}" if text else directive


def _apply_qwen3_message_directives(
    model_id: object,
    task_params: TextGenerationTaskParams,
    messages: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not _is_qwen3_model(model_id):
        return messages

    enable_thinking = _resolve_enable_thinking(task_params)
    if enable_thinking is None:
        return messages

    directive = "/think" if enable_thinking else "/no_think"
    updated_messages = [dict(message) for message in messages]

    for preferred_role in ("user", "developer", "system"):
        for index in range(len(updated_messages) - 1, -1, -1):
            message = updated_messages[index]
            if str(message.get("role", "")) != preferred_role:
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            updated_messages[index]["content"] = _with_prefixed_directive(
                content, directive
            )
            return updated_messages

    updated_messages.append({"role": "user", "content": directive})
    return updated_messages


def _apply_model_sampling_defaults(
    model_id: object,
    task_params: TextGenerationTaskParams,
    payload: dict[str, object],
) -> None:
    if not _is_qwen3_model(model_id):
        return

    enable_thinking = _resolve_enable_thinking(task_params)
    if enable_thinking is None:
        return

    # Qwen3 model cards recommend these defaults:
    # thinking: temp=0.6, top_p=0.95, top_k=20
    # non-thinking: temp=0.7, top_p=0.8, top_k=20
    payload.setdefault("temperature", 0.6 if enable_thinking else 0.7)
    payload.setdefault("top_p", 0.95 if enable_thinking else 0.8)
    payload.setdefault("top_k", 20)


class Runner:
    def __init__(
        self,
        bound_instance: BoundInstance,
        event_sender: MpSender[Event],
        task_receiver: MpReceiver[Task],
        cancel_receiver: MpReceiver[TaskId],
    ):
        self.event_sender = event_sender
        self.task_receiver = task_receiver
        self.cancel_receiver = cancel_receiver
        self.bound_instance = bound_instance

        self.instance = self.bound_instance.instance
        self.runner_id = self.bound_instance.bound_runner_id
        self.shard_metadata = self.bound_instance.bound_shard
        self.model_id = self.shard_metadata.model_card.model_id

        self.server_process: subprocess.Popen[str] | None = None
        self.server_port: int | None = None
        self.server_stdout_handle = None
        self.server_stderr_handle = None
        self.rpc_process: subprocess.Popen[str] | None = None
        self.rpc_stdout_handle = None
        self.rpc_stderr_handle = None
        self.http_client: httpx.Client | None = None
        self.relay_tunnel_manager: LlamaCppRelayTunnelManager | None = None
        self.reverse_relay_manager: LlamaCppReverseRelayManager | None = None
        self.seen: set[TaskId] = set()

        logger.info("hello from the llama.cpp runner")
        self.update_status(RunnerIdle())

    def update_status(self, status: RunnerStatus):
        self.current_status = status
        self.event_sender.send(
            RunnerStatusUpdated(
                runner_id=self.runner_id,
                runner_status=self.current_status,
            )
        )

    def send_task_status(self, task_id: TaskId, task_status: TaskStatus):
        self.event_sender.send(
            TaskStatusUpdated(task_id=task_id, task_status=task_status)
        )

    def acknowledge_task(self, task: Task):
        self.event_sender.send(TaskAcknowledged(task_id=task.task_id))

    def main(self):
        with self.task_receiver:
            for task in self.task_receiver:
                if task.task_id in self.seen:
                    logger.warning("repeat task - potential error")
                    continue
                self.seen.add(task.task_id)
                self.handle_task(task)
                if isinstance(self.current_status, RunnerShutdown):
                    break

    def handle_task(self, task: Task):
        self.send_task_status(task.task_id, TaskStatus.Running)

        match task:
            case ConnectToGroup() if isinstance(self.current_status, RunnerIdle):
                self.update_status(RunnerConnecting())
                self.acknowledge_task(task)
                if self._is_rpc_worker():
                    self._start_rpc_server()
                    self._start_reverse_relay_tunnels()
                elif self._is_distributed():
                    self._start_relay_tunnels()
                    self._wait_for_remote_rpc_servers()
                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerConnected())

            case LoadModel() if isinstance(
                self.current_status, (RunnerIdle, RunnerConnected)
            ):
                self.update_status(RunnerLoading(layers_loaded=0, total_layers=1))
                self.acknowledge_task(task)
                if (
                    not self._is_rpc_worker()
                    and not self._uses_cai_owned_transport_without_local_server()
                ):
                    self._start_server()
                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerLoaded())

            case StartWarmup() if isinstance(self.current_status, RunnerLoaded):
                self.update_status(RunnerWarmingUp())
                self.acknowledge_task(task)
                if (
                    not self._is_rpc_worker()
                    and not self._uses_cai_owned_transport_without_local_server()
                ):
                    self._warmup()
                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerReady())

            case TextGeneration() if isinstance(self.current_status, RunnerReady):
                self.update_status(RunnerRunning())
                self.acknowledge_task(task)
                self._maybe_create_cai_owned_transport_session(str(task.task_id))
                if not self._is_rpc_worker():
                    if not self._run_cai_owned_transport_generation_if_enabled(task):
                        self._run_text_generation(task)
                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerReady())

            case Shutdown():
                self._shutdown_server()
                self.update_status(RunnerShuttingDown())
                self.acknowledge_task(task)
                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerShutdown())

            case _:
                raise ValueError(
                    f"Received {task.__class__.__name__} outside of state machine in {self.current_status=}"
                )

    def _start_server(self):
        if self.server_process is not None:
            return

        model_path = _find_local_gguf_model_path(self.bound_instance)
        server_binary = _find_llama_server_binary()
        self.server_port = _choose_free_port()

        logs_dir = CAI_CACHE_HOME / "llama_cpp_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = logs_dir / f"{self.runner_id}.stdout.log"
        stderr_log = logs_dir / f"{self.runner_id}.stderr.log"
        self.server_stdout_handle = open(stdout_log, "a", encoding="utf-8")
        self.server_stderr_handle = open(stderr_log, "a", encoding="utf-8")

        args = self._build_server_args(server_binary, model_path, self.server_port)

        logger.info("starting llama.cpp server: {}", " ".join(args))
        self.server_process = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=self.server_stdout_handle,
            stderr=self.server_stderr_handle,
            text=True,
            creationflags=_windows_subprocess_flags(),
            startupinfo=_windows_startupinfo(),
        )
        self.http_client = httpx.Client(
            base_url=f"http://127.0.0.1:{self.server_port}",
            timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        )

        self._wait_until_server_ready(
            stderr_log,
            timeout_seconds=self._server_ready_timeout_seconds(),
        )

    def _start_rpc_server(self):
        if self.rpc_process is not None:
            return

        rpc_server_binary = _find_llama_rpc_server_binary()
        rpc_port = self._rpc_bind_port()

        logs_dir = CAI_CACHE_HOME / "llama_cpp_rpc_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = logs_dir / f"{self.runner_id}.stdout.log"
        stderr_log = logs_dir / f"{self.runner_id}.stderr.log"
        self.rpc_stdout_handle = open(stdout_log, "a", encoding="utf-8")
        self.rpc_stderr_handle = open(stderr_log, "a", encoding="utf-8")

        args = self._build_rpc_server_args(rpc_server_binary, rpc_port)

        logger.info("starting llama.cpp rpc-server: {}", " ".join(args))
        self.rpc_process = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=self.rpc_stdout_handle,
            stderr=self.rpc_stderr_handle,
            text=True,
            creationflags=_windows_subprocess_flags(),
            startupinfo=_windows_startupinfo(),
        )
        self._wait_until_socket_ready(
            host="127.0.0.1",
            port=rpc_port,
            process=self.rpc_process,
            stderr_log=stderr_log,
            timeout_seconds=DEFAULT_READY_TIMEOUT_SECONDS,
        )

    def _build_rpc_server_args(
        self, rpc_server_binary: Path, rpc_port: int
    ) -> list[str]:
        args = [
            str(rpc_server_binary),
            "--host",
            "0.0.0.0",
            "--port",
            str(rpc_port),
        ]
        if os.environ.get("CAI_LLAMA_CPP_RPC_CACHE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            args.append("--cache")

        device_override = os.environ.get("CAI_LLAMA_CPP_RPC_DEVICE")
        if device_override:
            args.extend(["--device", device_override])

        return args

    def _wait_until_server_ready(
        self,
        stderr_log: Path,
        *,
        timeout_seconds: int = DEFAULT_READY_TIMEOUT_SECONDS,
    ):
        assert self.server_process is not None
        assert self.http_client is not None

        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if self.server_process.poll() is not None:
                log_tail = ""
                if stderr_log.exists():
                    try:
                        log_tail = "\n".join(stderr_log.read_text(encoding="utf-8").splitlines()[-20:])
                    except Exception:
                        log_tail = ""
                error_message = (
                    "llama.cpp server exited before becoming ready"
                    + (f"\n{log_tail}" if log_tail else "")
                )
                self._record_remote_rpc_protocol_failure(error_message)
                raise RuntimeError(error_message)

            try:
                response = self.http_client.get("/health")
                if response.status_code == 200:
                    self._record_remote_rpc_protocol_success()
                    return
            except Exception:
                time.sleep(1)

        raise TimeoutError("Timed out waiting for llama.cpp server readiness")

    def _record_remote_rpc_protocol_success(self) -> None:
        self._record_remote_rpc_route_health(reachable=True)

    def _record_remote_rpc_protocol_failure(self, error_message: str) -> None:
        if getattr(self, "shard_metadata", None) is None:
            return
        if not self._is_distributed() or self._is_rpc_worker():
            return
        normalized_error = str(error_message or "").lower()
        if not any(
            marker in normalized_error
            for marker in _LLAMA_CPP_RPC_PROTOCOL_FAILURE_MARKERS
        ):
            return

        self._record_remote_rpc_route_health(
            reachable=False,
            error_message=str(error_message or "")[:1000],
        )

    def _record_remote_rpc_route_health(
        self,
        *,
        reachable: bool,
        error_message: str | None = None,
    ) -> None:
        if not self._is_distributed() or self._is_rpc_worker():
            return

        try:
            from cai_compute_chain.route_health import record_llama_cpp_rpc_result
        except Exception as exc:
            logger.debug("Unable to import route health for llama.cpp RPC result: {}", exc)
            return

        relay_routes = self._instance_relay_routes()
        for sink_node_id, endpoint in self._remote_rpc_server_specs():
            relay_route = next(iter(relay_routes.get(sink_node_id, [])), None)
            selected_mode: str | None = None
            selected_route = None
            relay_tunnel = self._relay_tunnel_for_sink(sink_node_id)
            if relay_tunnel is not None:
                try:
                    selected = relay_tunnel.selected_route_for_sink(sink_node_id)
                except Exception:
                    selected = None
                if selected is not None:
                    selected_mode, selected_route = selected
                    relay_route = selected_route
            transit_node_id = (
                str(getattr(relay_route, "transit_node_id", "") or "").strip()
                or None
            )
            endpoint_url = f"llama-cpp-rpc://{endpoint}"
            if relay_route is not None:
                target_host = str(getattr(relay_route, "target_host", "") or "").strip()
                target_port = int(getattr(relay_route, "target_port", 0) or 0)
                if target_host and target_port > 0:
                    if selected_mode == "direct":
                        endpoint_url = f"llama-cpp-rpc://{target_host}:{target_port}"
                        transit_node_id = None
                    else:
                        endpoint_url = f"relay://{transit_node_id}/{target_host}:{target_port}"
            try:
                record_llama_cpp_rpc_result(
                    source_node_id=str(self.bound_instance.bound_node_id),
                    sink_node_id=str(sink_node_id),
                    transit_node_id=transit_node_id,
                    endpoint_url=endpoint_url,
                    reachable=reachable,
                    error=error_message,
                )
            except Exception as exc:
                logger.debug(
                    "Unable to record llama.cpp RPC result source={} sink={}: {}",
                    self.bound_instance.bound_node_id,
                    sink_node_id,
                    exc,
                )

    def _wait_until_socket_ready(
        self,
        *,
        host: str,
        port: int,
        process: subprocess.Popen[str],
        stderr_log: Path,
        timeout_seconds: int,
    ) -> None:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if process.poll() is not None:
                log_tail = ""
                if stderr_log.exists():
                    try:
                        log_tail = "\n".join(
                            stderr_log.read_text(encoding="utf-8").splitlines()[-20:]
                        )
                    except Exception:
                        log_tail = ""
                raise RuntimeError(
                    "llama.cpp rpc-server exited before becoming ready"
                    + (f"\n{log_tail}" if log_tail else "")
                )

            try:
                _probe_llama_cpp_rpc_hello_socket(host, port, timeout=1)
                return
            except Exception:
                time.sleep(1)

        raise TimeoutError("Timed out waiting for llama.cpp rpc-server readiness")

    def _wait_for_remote_rpc_servers(self) -> None:
        pending_specs = self._remote_rpc_server_specs()
        if not pending_specs:
            return

        deadline = time.time() + DEFAULT_READY_TIMEOUT_SECONDS
        pending = {(sink_node_id, endpoint) for sink_node_id, endpoint in pending_specs}
        last_errors: dict[tuple[NodeId, str], str] = {}
        while pending and time.time() < deadline:
            ready: set[tuple[NodeId, str]] = set()
            for sink_node_id, endpoint in list(pending):
                relay_tunnel = self._relay_tunnel_for_sink(sink_node_id)
                if relay_tunnel is not None:
                    try:
                        relay_tunnel.probe_llama_cpp_rpc_route(
                            sink_node_id,
                            timeout=min(max(deadline - time.time(), 1.0), 6.0),
                        )
                        ready.add((sink_node_id, endpoint))
                        last_errors.pop((sink_node_id, endpoint), None)
                    except Exception as exc:
                        last_errors[(sink_node_id, endpoint)] = str(exc)[:500]
                        continue
                    continue

                host, port_text = endpoint.rsplit(":", 1)
                try:
                    _probe_llama_cpp_rpc_hello_socket(
                        host,
                        int(port_text),
                        timeout=1,
                    )
                    ready.add((sink_node_id, endpoint))
                    last_errors.pop((sink_node_id, endpoint), None)
                except Exception as exc:
                    last_errors[(sink_node_id, endpoint)] = str(exc)[:500]
                    continue
            pending -= ready
            if pending and time.time() < deadline:
                time.sleep(1)

        if pending:
            endpoints = sorted(endpoint for _sink_node_id, endpoint in pending)
            details = "; ".join(
                f"{endpoint}: {last_errors.get((sink_node_id, endpoint), 'not ready')}"
                for sink_node_id, endpoint in sorted(
                    pending,
                    key=lambda item: item[1],
                )
            )
            message = (
                "Timed out waiting for remote llama.cpp rpc-server peers: "
                + ", ".join(endpoints)
            )
            if details:
                message += f" ({details})"
            self._record_remote_rpc_route_health(
                reachable=False,
                error_message=message,
            )
            raise TimeoutError(message)

    def _warmup(self):
        if self._is_rpc_worker():
            return
        warmup_tokens = int(os.environ.get("CAI_WARMUP_MAX_OUTPUT_TOKENS", "1") or "1")
        if warmup_tokens <= 0:
            return
        try:
            self._request_chat_completion(
                payload={
                    "model": str(self.model_id),
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": warmup_tokens,
                    "stream": False,
                }
            )
        except Exception as exc:
            self._record_remote_rpc_protocol_failure(str(exc))
            raise

    def _request_chat_completion(self, payload: dict[str, object]) -> dict[str, object]:
        if self.http_client is None:
            raise RuntimeError("llama.cpp HTTP client is not initialized")
        response = self.http_client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Unexpected llama.cpp response payload")
        return data

    def _run_text_generation(self, task: TextGeneration):
        if self._is_rpc_worker():
            return
        try:
            messages = _apply_qwen3_message_directives(
                self.model_id,
                task.task_params,
                _build_messages(task.task_params),
            )
            payload: dict[str, object] = {
                "model": str(self.model_id),
                "messages": messages,
                "stream": False,
            }
            if task.task_params.max_output_tokens is not None:
                payload["max_tokens"] = task.task_params.max_output_tokens
            if task.task_params.temperature is not None:
                payload["temperature"] = task.task_params.temperature
            if task.task_params.top_p is not None:
                payload["top_p"] = task.task_params.top_p
            if task.task_params.top_k is not None:
                payload["top_k"] = task.task_params.top_k
            if task.task_params.min_p is not None:
                payload["min_p"] = task.task_params.min_p
            if task.task_params.repetition_penalty is not None:
                payload["repeat_penalty"] = task.task_params.repetition_penalty
            if task.task_params.repetition_context_size is not None:
                payload["repeat_last_n"] = task.task_params.repetition_context_size
            if task.task_params.stop is not None:
                payload["stop"] = task.task_params.stop
            if task.task_params.seed is not None:
                payload["seed"] = task.task_params.seed
            _apply_model_sampling_defaults(self.model_id, task.task_params, payload)

            data = self._request_chat_completion(payload)
            choices = data.get("choices")
            if not isinstance(choices, list) or len(choices) == 0:
                raise ValueError("llama.cpp response did not contain choices")
            first_choice = choices[0]
            if not isinstance(first_choice, dict):
                raise ValueError("Invalid llama.cpp choice payload")
            message = first_choice.get("message", {})
            if not isinstance(message, dict):
                message = {}

            usage = _coerce_usage(data.get("usage"))
            finish_reason = _coerce_finish_reason(first_choice.get("finish_reason"))

            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                normalized_tool_calls: list[ToolCallItem] = []
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    function_payload = tool_call.get("function", {})
                    if not isinstance(function_payload, dict):
                        continue
                    normalized_tool_calls.append(
                        ToolCallItem(
                            id=str(tool_call.get("id", "")),
                            name=str(function_payload.get("name", "")),
                            arguments=str(function_payload.get("arguments", "")),
                        )
                    )
                if normalized_tool_calls:
                    self.event_sender.send(
                        ChunkGenerated(
                            command_id=task.command_id,
                            chunk=ToolCallChunk(
                                model=self.model_id,
                                tool_calls=normalized_tool_calls,
                                usage=usage,
                            ),
                        )
                    )
                    return

            content = str(message.get("content", "") or "")
            reasoning_content = str(message.get("reasoning_content", "") or "")
            terminal_finish_reason = finish_reason or "stop"
            emitted_any = False

            if reasoning_content:
                emitted_any = True
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=task.command_id,
                        chunk=TokenChunk(
                            model=self.model_id,
                            text=reasoning_content,
                            token_id=0,
                            usage=usage if not content else None,
                            finish_reason=terminal_finish_reason if not content else None,
                            is_thinking=True,
                        ),
                    )
                )

            if content or not emitted_any:
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=task.command_id,
                        chunk=TokenChunk(
                            model=self.model_id,
                            text=content,
                            token_id=1 if emitted_any else 0,
                            usage=usage,
                            finish_reason=terminal_finish_reason,
                        ),
                    )
                )
        except Exception as exc:
            self._record_remote_rpc_protocol_failure(str(exc))
            logger.opt(exception=exc).warning("llama.cpp generation failed: {}", exc)
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        model=self.model_id,
                        error_message=str(exc),
                    ),
                )
            )

    def _run_cai_owned_transport_generation_if_enabled(
        self,
        task: TextGeneration,
    ) -> bool:
        if not _cai_owned_transport_generation_enabled():
            return False
        if not self._is_distributed() or not self._is_coordinator():
            return False

        try:
            result = self._dispatch_cai_owned_transport_generation(task)
            output = result.get("finalOutput")
            payload = output.get("payload") if isinstance(output, dict) else None
            text = self._decode_cai_owned_transport_generation_payload(payload)
            if not text:
                raise ValueError("CAI-owned transport final output is empty.")
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=TokenChunk(
                        model=self.model_id,
                        text=text,
                        token_id=0,
                        usage=None,
                        finish_reason="stop",
                    ),
                )
            )
            return True
        except Exception as exc:
            logger.opt(exception=exc).warning(
                "CAI-owned transport generation path failed: {}",
                exc,
            )
            if not _cai_owned_transport_generation_required():
                return False
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        model=self.model_id,
                        error_message=str(exc),
                    ),
                )
            )
            return True

    def _dispatch_cai_owned_transport_generation(
        self,
        task: TextGeneration,
    ) -> dict[str, object]:
        from cai_compute_chain.decentralized_compute import (
            await_cai_owned_transport_session_final_result,
            dispatch_cai_owned_transport_execution_dag,
        )

        requester_node_id = str(self.bound_instance.bound_node_id)
        executor_node_ids = self._cai_owned_transport_participant_node_ids()
        if not executor_node_ids:
            raise ValueError("CAI-owned transport generation has no executor nodes.")
        peer_cai_urls_by_node = self._cai_owned_transport_peer_cai_urls_by_node(
            [requester_node_id, *executor_node_ids]
        )
        missing_urls = [
            node_id
            for node_id in [requester_node_id, *executor_node_ids]
            if not peer_cai_urls_by_node.get(node_id)
        ]
        if missing_urls:
            raise ValueError(
                "CAI-owned transport generation is missing CAI API URLs for: "
                + ", ".join(missing_urls)
            )

        timeout_sec = _cai_owned_transport_generation_timeout_seconds()
        require_proven_data_plane_route = (
            _cai_owned_transport_generation_require_proven_data_plane_route()
        )
        minimum_relay_quorum = _cai_owned_transport_generation_minimum_relay_quorum()
        dispatch = dispatch_cai_owned_transport_execution_dag(
            instance_id=str(self.instance.instance_id),
            requester_node_id=requester_node_id,
            executor_node_ids=executor_node_ids,
            peer_cai_urls_by_node=peer_cai_urls_by_node,
            initial_payload=self._cai_owned_transport_generation_initial_payload(task),
            total_layer_count=self._cai_owned_transport_total_layer_count(),
            model_id=str(self.model_id),
            task_id=str(task.task_id),
            tokenizer_config_hash=self._cai_owned_transport_tokenizer_config_hash(),
            llm_runtime_metadata=self._cai_owned_transport_llm_runtime_metadata(),
            payload_compression=_cai_owned_transport_payload_compression(),
            payload_chunk_size_bytes=_cai_owned_transport_payload_chunk_size_bytes(),
            require_executor_readiness=(
                _cai_owned_transport_generation_require_executor_readiness()
            ),
            require_cai_owned_runtime_ready=(
                _cai_owned_transport_generation_require_runtime_ready()
            ),
            require_executor_shard_readiness=(
                _cai_owned_transport_generation_require_shard_readiness()
            ),
            require_data_plane_route=(
                _cai_owned_transport_generation_require_data_plane_route()
            ),
            require_proven_data_plane_route=require_proven_data_plane_route,
            timeout_sec=min(30.0, timeout_sec),
            route_policy={
                "runtime": "llama.cpp",
                "journalOnly": False,
                "dataPlane": "cai_owned_transport_execution_dag",
                "coordinator": "llama_cpp_runner",
                "avoidSingleTransitBottleneck": True,
                "requireProvenDataPlaneRoute": require_proven_data_plane_route,
                "minimumRelayQuorum": minimum_relay_quorum,
            },
        )
        result = await_cai_owned_transport_session_final_result(
            str(dispatch["sessionId"]),
            requester_node_id=requester_node_id,
            timeout_sec=timeout_sec,
            poll_interval_sec=_cai_owned_transport_generation_poll_seconds(),
        )
        if not bool(result.get("proofVerified")):
            raise ValueError(
                str(result.get("error") or "")
                or "CAI-owned transport execution proof was not verified."
            )
        return result

    def _cai_owned_transport_generation_initial_payload(
        self,
        task: TextGeneration,
    ) -> bytes:
        messages = _apply_qwen3_message_directives(
            self.model_id,
            task.task_params,
            _build_messages(task.task_params),
        )
        payload: dict[str, object] = {
            "schemaVersion": 1,
            "kind": "llama_cpp_text_generation_request",
            "model": str(self.model_id),
            "messages": messages,
            "stream": False,
            "taskId": str(task.task_id),
            "commandId": str(task.command_id),
        }
        if task.task_params.max_output_tokens is not None:
            payload["max_tokens"] = task.task_params.max_output_tokens
        if task.task_params.temperature is not None:
            payload["temperature"] = task.task_params.temperature
        if task.task_params.top_p is not None:
            payload["top_p"] = task.task_params.top_p
        if task.task_params.top_k is not None:
            payload["top_k"] = task.task_params.top_k
        if task.task_params.min_p is not None:
            payload["min_p"] = task.task_params.min_p
        if task.task_params.repetition_penalty is not None:
            payload["repeat_penalty"] = task.task_params.repetition_penalty
        if task.task_params.repetition_context_size is not None:
            payload["repeat_last_n"] = task.task_params.repetition_context_size
        if task.task_params.stop is not None:
            payload["stop"] = task.task_params.stop
        if task.task_params.seed is not None:
            payload["seed"] = task.task_params.seed
        _apply_model_sampling_defaults(self.model_id, task.task_params, payload)
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )

    def _cai_owned_transport_peer_cai_urls_by_node(
        self,
        node_ids: Sequence[str],
    ) -> dict[str, list[str]]:
        urls_by_node: dict[str, list[str]] = {}
        local_node_id = str(self.bound_instance.bound_node_id)
        local_fallback = str(
            os.environ.get("CAI_OWNED_TRANSPORT_LOCAL_CAI_URL") or ""
        ).strip()
        for node_id in node_ids:
            clean_node_id = str(node_id or "").strip()
            if not clean_node_id:
                continue
            urls = list(self._cai_api_urls_for_node(clean_node_id))
            if clean_node_id == local_node_id and local_fallback:
                urls.append(local_fallback.rstrip("/"))
            cleaned: list[str] = []
            seen: set[str] = set()
            for url in urls:
                clean_url = str(url or "").strip().rstrip("/")
                if not clean_url or clean_url in seen:
                    continue
                seen.add(clean_url)
                cleaned.append(clean_url)
            urls_by_node[clean_node_id] = cleaned
        return urls_by_node

    def _cai_owned_transport_total_layer_count(self) -> int:
        layer_counts = [
            int(shard.n_layers)
            for shard in self.instance.shard_assignments.runner_to_shard.values()
            if int(shard.n_layers) > 0
        ]
        if layer_counts:
            return max(layer_counts)
        end_layers = [
            int(shard.end_layer)
            for shard in self.instance.shard_assignments.runner_to_shard.values()
            if int(shard.end_layer) > 0
        ]
        if end_layers:
            return max(end_layers)
        model_layers = int(getattr(self.shard_metadata.model_card, "n_layers", 0) or 0)
        return max(1, model_layers)

    def _cai_owned_transport_tokenizer_config_hash(self) -> str:
        payload = {
            "modelId": str(self.model_id),
            "backend": "llama.cpp",
            "tokenizerConfig": "runtime-resolved",
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    def _cai_owned_transport_llm_runtime_metadata(self) -> dict[str, object]:
        model_card = self.shard_metadata.model_card
        total_layers = self._cai_owned_transport_total_layer_count()
        hidden_size = max(1, int(getattr(model_card, "hidden_size", 0) or 1))
        metadata: dict[str, object] = {
            "modelId": str(self.model_id),
            "totalLayerCount": total_layers,
            "hiddenSize": hidden_size,
            "activationDtype": (
                os.environ.get("CAI_LLM_SHARD_ACTIVATION_DTYPE") or "f16"
            ),
            "tensorEncoding": (
                os.environ.get("CAI_LLM_SHARD_TENSOR_ENCODING")
                or "ggml-tensor-v1"
            ),
            "tokenizerConfigHash": self._cai_owned_transport_tokenizer_config_hash(),
            "backend": "llama.cpp-patched",
            "backendVersion": (
                os.environ.get("CAI_LLM_SHARD_BACKEND_VERSION")
                or "llama.cpp/cai-shard-0.1"
            ),
            "metadataSource": "cai.llama_cpp.runner",
        }
        if int(getattr(model_card, "context_length", 0) or 0) > 0:
            metadata["contextLength"] = int(model_card.context_length)
        if str(getattr(model_card, "family", "") or "").strip():
            metadata["family"] = str(model_card.family)
        if str(getattr(model_card, "quantization", "") or "").strip():
            metadata["quantization"] = str(model_card.quantization)
        if str(getattr(model_card, "preferred_filename", "") or "").strip():
            metadata["preferredFilename"] = str(model_card.preferred_filename)
        if str(getattr(model_card, "gguf_architecture", "") or "").strip():
            metadata["ggufArchitecture"] = str(model_card.gguf_architecture)
        if str(getattr(model_card, "shard_compatibility", "") or "").strip():
            metadata["shardCompatibility"] = str(model_card.shard_compatibility)
        metadata["layerRangeSupported"] = bool(
            getattr(model_card, "layer_range_supported", False)
        )
        if str(getattr(model_card, "layer_range_probe_abi", "") or "").strip():
            metadata["layerRangeProbeAbi"] = str(model_card.layer_range_probe_abi)
        if str(getattr(model_card, "layer_range_probe_report", "") or "").strip():
            metadata["layerRangeProbeReport"] = str(
                model_card.layer_range_probe_report
            )
        if (
            str(
                getattr(model_card, "layer_range_equivalence_probe_report", "")
                or ""
            ).strip()
        ):
            metadata["layerRangeEquivalenceProbeReport"] = str(
                model_card.layer_range_equivalence_probe_report
            )
        if str(getattr(model_card, "state_format", "") or "").strip():
            metadata["stateFormat"] = str(model_card.state_format)
        if str(getattr(model_card, "activation_state_format", "") or "").strip():
            metadata["activationStateFormat"] = str(
                model_card.activation_state_format
            )
        if str(getattr(model_card, "decode_state_format", "") or "").strip():
            metadata["decodeStateFormat"] = str(model_card.decode_state_format)
        return metadata

    @staticmethod
    def _decode_cai_owned_transport_generation_payload(payload: object) -> str:
        if isinstance(payload, bytes):
            return payload.decode("utf-8", errors="replace")
        if isinstance(payload, bytearray):
            return bytes(payload).decode("utf-8", errors="replace")
        if isinstance(payload, str):
            return payload
        return ""

    def _shutdown_server(self):
        if self.http_client is not None:
            self.http_client.close()
            self.http_client = None

        if self.relay_tunnel_manager is not None:
            self.relay_tunnel_manager.stop()
            self.relay_tunnel_manager = None

        if self.reverse_relay_manager is not None:
            self.reverse_relay_manager.stop()
            self.reverse_relay_manager = None

        if self.server_process is not None:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait(timeout=10)
            self.server_process = None

        if self.server_stdout_handle is not None:
            self.server_stdout_handle.close()
            self.server_stdout_handle = None
        if self.server_stderr_handle is not None:
            self.server_stderr_handle.close()
            self.server_stderr_handle = None

        if self.rpc_process is not None:
            self.rpc_process.terminate()
            try:
                self.rpc_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.rpc_process.kill()
                self.rpc_process.wait(timeout=10)
            self.rpc_process = None

        if self.rpc_stdout_handle is not None:
            self.rpc_stdout_handle.close()
            self.rpc_stdout_handle = None
        if self.rpc_stderr_handle is not None:
            self.rpc_stderr_handle.close()
            self.rpc_stderr_handle = None

    def _is_distributed(self) -> bool:
        return self.shard_metadata.world_size > 1

    def _is_coordinator(self) -> bool:
        return self.shard_metadata.device_rank == 0

    def _is_rpc_worker(self) -> bool:
        return self._is_distributed() and not self._is_coordinator()

    def _uses_cai_owned_transport_without_local_server(self) -> bool:
        return (
            self._is_distributed()
            and self._is_coordinator()
            and _cai_owned_transport_skip_local_llama_server()
        )

    def _server_ready_timeout_seconds(self) -> int:
        if self._is_distributed() and not self._is_rpc_worker():
            return max(
                DEFAULT_READY_TIMEOUT_SECONDS,
                DEFAULT_DISTRIBUTED_READY_TIMEOUT_SECONDS,
            )
        return DEFAULT_READY_TIMEOUT_SECONDS

    def _build_server_args(
        self,
        server_binary: Path,
        model_path: Path,
        server_port: int,
    ) -> list[str]:
        default_context_size = (
            self.shard_metadata.model_card.context_length or DEFAULT_CONTEXT_SIZE
        )
        default_context_size = min(default_context_size, DEFAULT_CONTEXT_SIZE)
        context_size = int(
            os.environ.get(
                "CAI_LLAMA_CPP_CTX_SIZE",
                str(default_context_size),
            )
        )
        gpu_layers = int(
            os.environ.get("CAI_LLAMA_CPP_GPU_LAYERS", str(DEFAULT_GPU_LAYERS))
        )

        args = [
            str(server_binary),
            "-m",
            str(model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(server_port),
            "-c",
            str(context_size),
            "-ngl",
            str(gpu_layers),
        ]
        remote_rpc_servers = self._remote_rpc_servers()
        if remote_rpc_servers:
            args.extend(["--rpc", ",".join(remote_rpc_servers)])
            args.extend(["--fit", "off", "--split-mode", "layer"])
            if not _distributed_prompt_cache_enabled():
                args.extend(["--no-cache-prompt", "--cache-ram", "0"])

            device_names = self._distributed_device_names(len(remote_rpc_servers))
            if device_names:
                args.extend(["--device", ",".join(device_names)])

            tensor_split = self._distributed_tensor_split()
            if len(tensor_split) == 1 + len(remote_rpc_servers):
                args.extend(
                    ["--tensor-split", ",".join(str(weight) for weight in tensor_split)]
                )

        return args

    def _distributed_device_names(self, remote_count: int) -> list[str]:
        if remote_count <= 0:
            return []

        local_device = os.environ.get("CAI_LLAMA_CPP_COORDINATOR_DEVICE")
        if not local_device:
            local_device = "CUDA0" if os.name == "nt" else "CPU"
        return [local_device, *[f"RPC{i}" for i in range(remote_count)]]

    def _distributed_tensor_split(self) -> list[int]:
        if not self._is_distributed():
            return []

        shards = sorted(
            self.instance.shard_assignments.runner_to_shard.values(),
            key=lambda shard: shard.device_rank,
        )
        return [max(1, shard.end_layer - shard.start_layer) for shard in shards]

    def _instance_hosts(self) -> list[Host]:
        if isinstance(self.instance, MlxRingInstance):
            return list(self.instance.hosts_by_node.get(self.bound_instance.bound_node_id, []))
        return []

    def _instance_node_ids_by_rank(self) -> list[NodeId]:
        node_ids_by_rank: list[NodeId | None] = [None] * self.shard_metadata.world_size
        for node_id, runner_id in self.instance.shard_assignments.node_to_runner.items():
            shard = self.instance.shard_assignments.runner_to_shard.get(runner_id)
            if shard is None:
                continue
            node_ids_by_rank[shard.device_rank] = node_id
        return [
            node_id
            for node_id in node_ids_by_rank
            if node_id is not None
        ]

    def _cai_owned_transport_participant_node_ids(self) -> list[str]:
        if not self._is_distributed():
            return []
        node_ids = [str(node_id) for node_id in self._instance_node_ids_by_rank()]
        if len(node_ids) != int(self.shard_metadata.world_size):
            return []
        return node_ids

    def _cai_owned_transport_session_id(self, task_id: str | None = None) -> str | None:
        participants = self._cai_owned_transport_participant_node_ids()
        if not participants:
            return None
        try:
            from cai_compute_chain.decentralized_compute import (
                deterministic_cai_owned_transport_session_id,
            )
        except Exception as exc:
            logger.debug("Unable to import CAI transport session helper: {}", exc)
            return None
        try:
            return deterministic_cai_owned_transport_session_id(
                str(self.instance.instance_id),
                participants,
                task_id=task_id,
            )
        except ValueError as exc:
            logger.debug("Unable to derive CAI transport session id: {}", exc)
            return None

    def _cai_owned_transport_session_payload(
        self,
        task_id: str | None = None,
    ) -> dict[str, object] | None:
        participants = self._cai_owned_transport_participant_node_ids()
        if not participants:
            return None
        try:
            from cai_compute_chain.decentralized_compute import (
                build_cai_owned_transport_session_offer,
            )
        except Exception as exc:
            logger.debug("Unable to import CAI transport session offer helper: {}", exc)
            return None
        try:
            return build_cai_owned_transport_session_offer(
                instance_id=str(self.instance.instance_id),
                participant_node_ids=participants,
                model_id=str(self.model_id),
                task_id=task_id,
                source_node_id=participants[0],
                route_policy={
                    "runtime": "llama.cpp",
                    "journalOnly": True,
                    "dataPlane": "standard_llama_cpp_rpc",
                },
            )
        except ValueError as exc:
            logger.debug("Unable to build CAI transport session offer: {}", exc)
            return None

    def _cai_api_urls_for_node(self, node_id: NodeId | str) -> list[str]:
        if not isinstance(self.instance, MlxRingInstance):
            return []
        node_key = NodeId(str(node_id))
        raw_urls = self.instance.cai_api_urls_by_node.get(node_key)
        if not raw_urls:
            return []
        urls: list[str] = []
        seen: set[str] = set()
        for raw_url in raw_urls:
            url = str(raw_url or "").strip().rstrip("/")
            if not url or url in seen:
                continue
            seen.add(url)
            urls.append(url)
        return urls

    def _submit_cai_owned_transport_session_offer(
        self,
        payload: dict[str, object],
    ) -> None:
        if not _cai_owned_transport_offer_submit_enabled():
            return
        try:
            from cai_compute_chain.decentralized_compute import (
                submit_cai_owned_transport_session_offer,
            )
        except Exception as exc:
            logger.debug("Unable to import CAI transport session offer client: {}", exc)
            return
        local_node_id = str(self.bound_instance.bound_node_id)
        for node_id in payload.get("participantNodeIds", []):
            clean_node_id = str(node_id or "").strip()
            if not clean_node_id or clean_node_id == local_node_id:
                continue
            peer_urls = self._cai_api_urls_for_node(clean_node_id)
            if not peer_urls:
                logger.debug(
                    "No CAI API URL available for transport offer target {}",
                    clean_node_id,
                )
                continue
            last_error: Exception | None = None
            for peer_url in peer_urls:
                try:
                    submit_cai_owned_transport_session_offer(peer_url, payload)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
            if last_error is not None:
                logger.debug(
                    "Unable to submit CAI transport session offer to {} via {} URLs: {}",
                    clean_node_id,
                    len(peer_urls),
                    last_error,
                )

    def _maybe_create_cai_owned_transport_session(
        self,
        task_id: str | None = None,
    ) -> None:
        if not _cai_owned_transport_journal_enabled():
            return
        if not self._is_distributed() or not self._is_coordinator():
            return
        payload = self._cai_owned_transport_session_payload(task_id=task_id)
        if payload is None:
            return
        try:
            from cai_compute_chain.decentralized_compute import (
                create_cai_owned_transport_session,
            )
        except Exception as exc:
            logger.debug("Unable to import CAI transport session journal: {}", exc)
            return
        try:
            create_cai_owned_transport_session(
                session_id=str(payload["sessionId"]),
                instance_id=str(payload["instanceId"]),
                participant_node_ids=[
                    str(node_id)
                    for node_id in payload.get("participantNodeIds", [])
                ],
                model_id=str(payload["modelId"]),
                task_id=str(payload["taskId"] or "").strip() or None,
                source_node_id=str(payload["sourceNodeId"]),
                execution_mode=str(payload["executionMode"]),
                route_policy=payload.get("routePolicy")
                if isinstance(payload.get("routePolicy"), dict)
                else None,
            )
            self._submit_cai_owned_transport_session_offer(payload)
        except Exception as exc:
            logger.debug("Unable to create CAI transport session journal: {}", exc)

    def _instance_relay_routes(self) -> dict[NodeId, list[object]]:
        if isinstance(self.instance, MlxRingInstance):
            routes_by_sink: dict[NodeId, list[object]] = {}
            for route in self.instance.relay_routes_by_node.get(
                self.bound_instance.bound_node_id,
                [],
            ):
                routes_by_sink.setdefault(route.sink_node_id, []).append(route)
            return routes_by_sink
        return {}

    def _instance_incoming_relay_routes(self) -> list[object]:
        if not isinstance(self.instance, MlxRingInstance):
            return []

        incoming_routes = []
        for routes in self.instance.relay_routes_by_node.values():
            for route in routes:
                if route.sink_node_id == self.bound_instance.bound_node_id:
                    incoming_routes.append(route)
        return incoming_routes

    def _start_relay_tunnels(self) -> None:
        if getattr(self, "relay_tunnel_manager", None) is not None:
            return
        relay_routes = [
            route
            for routes in self._instance_relay_routes().values()
            for route in routes
        ]
        if not relay_routes:
            return
        self.relay_tunnel_manager = LlamaCppRelayTunnelManager(relay_routes)
        self.relay_tunnel_manager.start()

    def _start_reverse_relay_tunnels(self) -> None:
        if getattr(self, "reverse_relay_manager", None) is not None:
            return
        incoming_routes = self._instance_incoming_relay_routes()
        if not incoming_routes:
            return
        self.reverse_relay_manager = LlamaCppReverseRelayManager(incoming_routes)
        self.reverse_relay_manager.start()

    def _relay_tunnel_for_sink(
        self,
        sink_node_id: NodeId,
    ) -> LlamaCppRelayTunnelManager | None:
        relay_tunnel_manager = getattr(self, "relay_tunnel_manager", None)
        if relay_tunnel_manager is None:
            return None
        if relay_tunnel_manager.local_endpoint_for_sink(sink_node_id) is None:
            return None
        return relay_tunnel_manager

    def _rpc_bind_port(self) -> int:
        if isinstance(self.instance, MlxRingInstance):
            instance_hosts = self._instance_hosts()
            if len(instance_hosts) > self.shard_metadata.device_rank:
                host = instance_hosts[self.shard_metadata.device_rank]
                if host.port > 0:
                    return host.port
            return self.instance.ephemeral_port
        return _choose_free_port()

    def _remote_rpc_server_specs(self) -> list[tuple[NodeId, str]]:
        remote_hosts: list[tuple[NodeId, str]] = []
        node_ids_by_rank = self._instance_node_ids_by_rank()
        relay_routes = self._instance_relay_routes()
        for idx, host in enumerate(self._instance_hosts()):
            if idx == self.shard_metadata.device_rank or idx >= len(node_ids_by_rank):
                continue
            sink_node_id = node_ids_by_rank[idx]
            relay_tunnel = self._relay_tunnel_for_sink(sink_node_id)
            if relay_tunnel is not None:
                local_endpoint = relay_tunnel.local_endpoint_for_sink(sink_node_id)
                if local_endpoint is None:
                    continue
                endpoint = f"{local_endpoint.ip}:{local_endpoint.port}"
                if (sink_node_id, endpoint) not in remote_hosts:
                    remote_hosts.append((sink_node_id, endpoint))
                continue
            if sink_node_id in relay_routes:
                continue
            if host.port <= 0:
                continue
            if host.ip in {"0.0.0.0", "198.51.100.1"}:
                continue
            endpoint = f"{host.ip}:{host.port}"
            if (sink_node_id, endpoint) not in remote_hosts:
                remote_hosts.append((sink_node_id, endpoint))
        return remote_hosts

    def _remote_rpc_servers(self) -> list[str]:
        return [endpoint for _sink_node_id, endpoint in self._remote_rpc_server_specs()]

