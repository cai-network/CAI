# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import argparse
import ipaddress
import multiprocessing as mp
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Self

try:
    import resource
except ImportError:
    resource = None

import anyio
from anyio import EndOfStream
from anyio import BrokenResourceError, ClosedResourceError
from loguru import logger
from pydantic import PositiveInt

import cai.routing.topics as topics
from cai.api.main import API
from cai.download.coordinator import DownloadCoordinator
from cai.download.impl_shard_downloader import cai_shard_downloader
from cai.master.main import Master
from cai.routing.event_router import EventRouter
from cai.routing.router import Router, get_node_id_keypair
from cai.shared.constants import CAI_LOG
from cai.shared.election import Election, ElectionResult
from cai.shared.logging import logger_cleanup, logger_setup
from cai.shared.single_instance import (
    NodeSingleInstanceGuard,
    node_instance_error_message,
    should_enforce_node_single_instance,
)
from cai.shared.types.commands import ForwarderCommand, RequestEventLog
from cai.shared.types.common import NodeId, SessionId, SystemId
from cai.shared.types.events import (
    Event as CAIEvent,
    NodeGatheredInfo,
    OverlayBootstrapPeersAdvertised,
    OverlayPeerConnected,
    OverlayPeerDisconnected,
)
from cai.shared.types.multiaddr import Multiaddr
from cai.utils.channels import Receiver, Sender, channel
from cai.utils.info_gatherer.info_gatherer import (
    GatheredInfo,
    InfoGatherer,
    resolve_advertised_host,
)
from cai.utils.pydantic_ext import CamelCaseModel
from cai.utils.task_group import TaskGroup
from cai.worker.main import Worker


DEFAULT_SESSION_TRANSITION_DEBOUNCE = float(
    os.getenv("CAI_SESSION_TRANSITION_DEBOUNCE_SECONDS")
    or "0.0"
)
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_FALSE_ENV_VALUES = {"0", "false", "no", "off", "disabled"}


def _overlay_advertise_retry_delays() -> tuple[float, ...]:
    raw = os.getenv("CAI_OVERLAY_ADVERTISE_RETRY_SECONDS", "0,3,10,30")
    delays: list[float] = []
    for item in raw.split(","):
        normalized = item.strip()
        if not normalized:
            continue
        try:
            delay = float(normalized)
        except ValueError:
            continue
        delays.append(max(0.0, delay))
    return tuple(delays or (0.0,))


def _load_local_node_runtime_config() -> object | None:
    try:
        from cai_compute_chain.model import WalletPolicy
        from cai_compute_chain.node_config import load_or_create_node_config

        return load_or_create_node_config(WalletPolicy())
    except Exception:
        return None


def _env_flag_override(*names: str) -> bool | None:
    for name in names:
        raw = str(os.getenv(name) or "").strip()
        if not raw:
            continue
        return raw.lower() in _TRUE_ENV_VALUES
    return None


def _resolve_master_candidate_enabled(
    *,
    bootstrap_peers: list[str],
    worker_runtime_enabled: bool,
    force_master: bool,
    config: object | None = None,
) -> tuple[bool, str]:
    if force_master:
        return True, "enabled by --force-master"

    override = _env_flag_override("CAI_MASTER_CANDIDATE", "CAI_MASTER_CANDIDATE")
    if override is not None:
        state = "enabled" if override else "disabled"
        return override, f"{state} via environment override"

    active_config = config if config is not None else _load_local_node_runtime_config()
    validator_enabled = bool(getattr(active_config, "validator_enabled", False))
    validator_state = str(getattr(active_config, "validator_state", "") or "").strip().lower()
    worker_enabled = bool(getattr(active_config, "worker_enabled", False))

    if validator_enabled or validator_state in {"bonded", "unbonding"}:
        return True, "enabled for validator-capable node"

    if (
        worker_runtime_enabled
        and worker_enabled
        and any(str(peer).strip() for peer in bootstrap_peers)
    ):
        return False, "disabled for bootstrap worker node"

    return True, "enabled by default"


def _new_boot_session_clock() -> int:
    raw = str(os.getenv("CAI_BOOT_SESSION_CLOCK") or "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            logger.warning("Ignoring invalid CAI_BOOT_SESSION_CLOCK={}", raw)
    return time.time_ns()


def _patch_windows_spawn_no_console() -> None:
    if os.name != "nt":
        return

    try:
        import subprocess
        import multiprocessing.popen_spawn_win32 as spawn_win32
    except ImportError:
        return

    if getattr(spawn_win32.Popen, "_cai_no_console_patch", False):
        return

    class _NoConsolePopen(spawn_win32.Popen):
        _cai_no_console_patch = True

        def __init__(self, process_obj):
            prep_data = spawn_win32.spawn.get_preparation_data(process_obj._name)

            rhandle, whandle = spawn_win32._winapi.CreatePipe(None, 0)
            wfd = spawn_win32.msvcrt.open_osfhandle(whandle, 0)
            cmd = spawn_win32.spawn.get_command_line(
                parent_pid=os.getpid(),
                pipe_handle=rhandle,
            )

            python_exe = spawn_win32.spawn.get_executable()

            if spawn_win32.WINENV and spawn_win32._path_eq(
                python_exe, sys.executable
            ):
                cmd[0] = python_exe = sys._base_executable
                env = os.environ.copy()
                env["__PYVENV_LAUNCHER__"] = sys.executable
            else:
                env = None

            cmd = " ".join(f'"{x}"' for x in cmd)

            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

            with open(wfd, "wb", closefd=True) as to_child:
                try:
                    hp, ht, pid, tid = spawn_win32._winapi.CreateProcess(
                        python_exe,
                        cmd,
                        None,
                        None,
                        False,
                        creationflags,
                        env,
                        None,
                        startupinfo,
                    )
                    spawn_win32._winapi.CloseHandle(ht)
                except Exception:
                    spawn_win32._winapi.CloseHandle(rhandle)
                    raise

                self.pid = pid
                self.returncode = None
                self._handle = hp
                self.sentinel = int(hp)
                self.finalizer = spawn_win32.util.Finalize(
                    self,
                    spawn_win32._close_handles,
                    (self.sentinel, int(rhandle)),
                )

                spawn_win32.set_spawning_popen(self)
                try:
                    spawn_win32.reduction.dump(prep_data, to_child)
                    spawn_win32.reduction.dump(process_obj, to_child)
                finally:
                    spawn_win32.set_spawning_popen(None)

    spawn_win32.Popen = _NoConsolePopen


async def _stabilize_election_result(
    results: Receiver[ElectionResult],
    first: ElectionResult,
    *,
    current_session_id: SessionId,
) -> ElectionResult:
    debounce_seconds = DEFAULT_SESSION_TRANSITION_DEBOUNCE
    if debounce_seconds <= 0 or first.session_id == current_session_id:
        return first

    stabilized = first
    deadline = anyio.current_time() + debounce_seconds
    logger.info(
        "Debouncing session transition for {:.1f}s: {} -> {}",
        debounce_seconds,
        current_session_id,
        first.session_id,
    )

    while True:
        remaining = deadline - anyio.current_time()
        if remaining <= 0:
            return stabilized

        try:
            with anyio.move_on_after(remaining) as scope:
                candidate = await results.receive()
        except EndOfStream:
            return stabilized

        if scope.cancelled_caught:
            return stabilized

        stabilized = candidate


def _auto_advertise_overlay_peer_enabled() -> bool:
    raw = str(
        os.getenv("CAI_AUTO_ADVERTISE_OVERLAY_PEER")
        or ""
    ).strip()
    if not raw:
        return True
    return raw.lower() not in _FALSE_ENV_VALUES


def _multiaddr_host_prefix(host: str) -> str | None:
    normalized_host = str(host or "").strip()
    if not normalized_host:
        return None

    host_without_zone = normalized_host.split("%", 1)[0]
    try:
        ip = ipaddress.ip_address(host_without_zone)
    except ValueError:
        return f"/dns/{normalized_host}"

    if ip.version == 4:
        return f"/ip4/{host_without_zone}"
    if ip.version == 6:
        return f"/ip6/{host_without_zone}"
    return None


async def _resolve_advertised_bootstrap_peers(
    node_id: NodeId,
    *,
    listen_port: int,
) -> tuple[Multiaddr, ...]:
    explicit_peers = _parse_advertised_bootstrap_peers()
    if explicit_peers:
        return explicit_peers

    if not _auto_advertise_overlay_peer_enabled():
        return ()

    preferred_host = (
        os.getenv("CAI_PUBLIC_DATA_HOST")
        or os.getenv("CAI_PUBLIC_API_HOST")
    )
    resolved_host = await resolve_advertised_host(preferred_host)
    host_prefix = _multiaddr_host_prefix(resolved_host or "")
    if host_prefix is None or listen_port <= 0:
        return ()

    return (
        Multiaddr(address=f"{host_prefix}/tcp/{listen_port}"),
    )


@dataclass
class Node:
    router: Router
    event_router: EventRouter
    download_coordinator: DownloadCoordinator | None
    worker: Worker | None
    info_publisher: "NodeInfoPublisher | None"
    election: Election  # Every node participates in election, as we do want a node to become master even if it isn't a master candidate if no master candidates are present.
    election_result_receiver: Receiver[ElectionResult]
    master: Master | None
    api: API | None

    node_id: NodeId
    offline: bool
    _api_port: int
    _tg: TaskGroup = field(init=False, default_factory=TaskGroup)
    _connected_overlay_peers: set[NodeId] = field(
        init=False, default_factory=set
    )
    _peer_event_sender: Sender[CAIEvent] | None = field(init=False, default=None)
    _sync_system_id: SystemId = field(init=False, default_factory=SystemId)
    _advertised_bootstrap_peers: tuple[Multiaddr, ...] = field(
        init=False, default=()
    )

    @classmethod
    async def create(cls, args: "Args") -> Self:
        keypair = get_node_id_keypair()
        node_id = NodeId(keypair.to_node_id())
        session_id = SessionId(
            master_node_id=node_id,
            election_clock=_new_boot_session_clock(),
        )
        router = Router.create(
            keypair,
            bootstrap_peers=args.bootstrap_peers,
            listen_port=args.libp2p_port,
        )
        await router.register_topic(topics.GLOBAL_EVENTS)
        await router.register_topic(topics.LOCAL_EVENTS)
        await router.register_topic(topics.COMMANDS)
        await router.register_topic(topics.ELECTION_MESSAGES)
        await router.register_topic(topics.CONNECTION_MESSAGES)
        await router.register_topic(topics.DOWNLOAD_COMMANDS)
        await router.register_topic(topics.CAI_OWNED_TRANSPORT_MESSAGES)
        event_router = EventRouter(
            session_id,
            command_sender=router.sender(topics.COMMANDS),
            external_outbound=router.sender(topics.LOCAL_EVENTS),
            external_inbound=router.receiver(topics.GLOBAL_EVENTS),
        )

        logger.info(f"Starting node {node_id}")
        is_master_candidate, master_candidate_reason = _resolve_master_candidate_enabled(
            bootstrap_peers=args.bootstrap_peers,
            worker_runtime_enabled=not args.no_worker,
            force_master=args.force_master,
        )
        logger.info(
            "Master candidacy {} ({})",
            "enabled" if is_master_candidate else "disabled",
            master_candidate_reason,
        )

        # Create DownloadCoordinator (unless --no-downloads)
        if not args.no_downloads:
            download_coordinator = DownloadCoordinator(
                node_id,
                cai_shard_downloader(offline=args.offline),
                event_sender=event_router.sender(),
                download_command_receiver=router.receiver(topics.DOWNLOAD_COMMANDS),
                offline=args.offline,
            )
        else:
            download_coordinator = None

        if args.spawn_api:
            api = API(
                node_id,
                port=args.api_port,
                event_receiver=event_router.receiver(),
                command_sender=router.sender(topics.COMMANDS),
                download_command_sender=router.sender(topics.DOWNLOAD_COMMANDS),
                election_receiver=router.receiver(topics.ELECTION_MESSAGES),
                cai_owned_transport_message_sender=router.sender(
                    topics.CAI_OWNED_TRANSPORT_MESSAGES,
                ),
                cai_owned_transport_message_receiver=router.receiver(
                    topics.CAI_OWNED_TRANSPORT_MESSAGES,
                ),
            )
        else:
            api = None

        if not args.no_worker:
            worker = Worker(
                node_id,
                event_receiver=event_router.receiver(),
                event_sender=event_router.sender(),
                command_sender=router.sender(topics.COMMANDS),
                download_command_sender=router.sender(topics.DOWNLOAD_COMMANDS),
                api_port=args.api_port,
            )
        else:
            worker = None
        info_publisher = (
            NodeInfoPublisher(
                node_id=node_id,
                api_port=args.api_port,
                event_sender=event_router.sender(),
            )
            if worker is None
            else None
        )

        # We start every node with a master
        master = Master(
            node_id,
            session_id,
            event_sender=event_router.sender(),
            global_event_sender=router.sender(topics.GLOBAL_EVENTS),
            local_event_receiver=router.receiver(topics.LOCAL_EVENTS),
            command_receiver=router.receiver(topics.COMMANDS),
            download_command_sender=router.sender(topics.DOWNLOAD_COMMANDS),
        )

        er_send, er_recv = channel[ElectionResult]()
        election = Election(
            node_id,
            # If someone manages to assemble 1 MILLION devices into an cai cluster then. well done. good job champ.
            seniority=1_000_000 if args.force_master else 0,
            is_candidate=is_master_candidate,
            # nb: this DOES feedback right now. i have thoughts on how to address this,
            # but ultimately it seems not worth the complexity
            election_message_sender=router.sender(topics.ELECTION_MESSAGES),
            election_message_receiver=router.receiver(topics.ELECTION_MESSAGES),
            connection_message_receiver=router.receiver(topics.CONNECTION_MESSAGES),
            command_receiver=router.receiver(topics.COMMANDS),
            election_result_sender=er_send,
            initial_session_id=session_id,
        )

        node = cls(
            router,
            event_router,
            download_coordinator,
            worker,
            info_publisher,
            election,
            er_recv,
            master,
            api,
            node_id,
            args.offline,
            args.api_port,
        )
        node._peer_event_sender = node.event_router.sender()
        node._advertised_bootstrap_peers = await _resolve_advertised_bootstrap_peers(
            node_id,
            listen_port=args.libp2p_port,
        )
        return node

    async def run(self):
        async with self._tg as tg:
            signal.signal(signal.SIGINT, lambda _, __: self.shutdown())
            signal.signal(signal.SIGTERM, lambda _, __: self.shutdown())
            tg.start_soon(self.router.run)
            tg.start_soon(self.event_router.run)
            tg.start_soon(self._track_overlay_connections)
            tg.start_soon(self._advertise_overlay_bootstrap_peers)
            tg.start_soon(self.election.run)
            if self.download_coordinator:
                tg.start_soon(self.download_coordinator.run)
            if self.worker:
                tg.start_soon(self.worker.run)
            if self.info_publisher:
                tg.start_soon(self.info_publisher.run)
            if self.master:
                tg.start_soon(self.master.run)
            if self.api:
                tg.start_soon(self.api.run)
            tg.start_soon(self._elect_loop)

    def shutdown(self):
        # if this is our second call to shutdown, just sys.exit
        if self._tg.cancel_called():
            import sys

            sys.exit(1)
        self._tg.cancel_tasks()

    async def _track_overlay_connections(self):
        with self.router.receiver(topics.CONNECTION_MESSAGES) as messages:
            async for message in messages:
                if message.node_id == self.node_id:
                    continue

                if message.connected:
                    if message.node_id in self._connected_overlay_peers:
                        continue
                    self._connected_overlay_peers.add(message.node_id)
                    if self._peer_event_sender is not None:
                        await self._peer_event_sender.send(
                            OverlayPeerConnected(
                                local_node_id=self.node_id,
                                remote_node_id=message.node_id,
                            )
                        )
                        await self._send_overlay_bootstrap_peers_advertised()
                    if (
                        self.election.current_session.master_node_id
                        == message.node_id
                        and message.node_id != self.node_id
                    ):
                        await self._request_full_sync("overlay_master_connected")
                    continue

                if message.node_id not in self._connected_overlay_peers:
                    continue

                self._connected_overlay_peers.remove(message.node_id)
                if self._peer_event_sender is not None:
                    await self._peer_event_sender.send(
                        OverlayPeerDisconnected(
                            local_node_id=self.node_id,
                            remote_node_id=message.node_id,
                        )
                    )

    async def _request_full_sync(self, reason: str):
        logger.info(f"Requesting full event-log sync: {reason}")
        await self.router.sender(topics.COMMANDS).send(
            ForwarderCommand(
                origin=self._sync_system_id,
                command=RequestEventLog(since_idx=0),
            )
        )

    async def _replay_overlay_peer_state(self):
        if self._peer_event_sender is None:
            return

        for peer_id in sorted(self._connected_overlay_peers):
            await self._peer_event_sender.send(
                OverlayPeerConnected(
                    local_node_id=self.node_id,
                    remote_node_id=peer_id,
                )
            )

    async def _advertise_overlay_bootstrap_peers(self):
        if (
            self._peer_event_sender is None
            or not self._advertised_bootstrap_peers
        ):
            return

        for delay in _overlay_advertise_retry_delays():
            if delay > 0:
                await anyio.sleep(delay)
            await self._send_overlay_bootstrap_peers_advertised()

    async def _send_overlay_bootstrap_peers_advertised(self):
        if (
            self._peer_event_sender is None
            or not self._advertised_bootstrap_peers
        ):
            return

        await self._peer_event_sender.send(
            OverlayBootstrapPeersAdvertised(
                node_id=self.node_id,
                peers=list(self._advertised_bootstrap_peers),
            )
        )

    async def _elect_loop(self):
        with self.election_result_receiver as results:
            async for result in results:
                # This function continues to have a lot of very specific entangled logic
                # At least it's somewhat contained

                # I don't like this duplication, but it's manageable for now.
                # TODO: This function needs refactoring generally

                # Ok:
                # On new master:
                # - Elect master locally if necessary
                # - Shutdown and re-create the worker
                # - Shut down and re-create the API

                current_session_id = self.event_router.session_id
                result = await _stabilize_election_result(
                    results,
                    result,
                    current_session_id=current_session_id,
                )
                requires_session_transition = (
                    result.session_id != self.event_router.session_id
                )

                if requires_session_transition:
                    if self.info_publisher:
                        await self.info_publisher.shutdown()
                    logger.info(
                        "Applying stabilized session transition: {} -> {}",
                        self.event_router.session_id,
                        result.session_id,
                    )
                    await anyio.sleep(0)
                    self.event_router.shutdown()
                    self.event_router = EventRouter(
                        result.session_id,
                        self.router.sender(topics.COMMANDS),
                        self.router.receiver(topics.GLOBAL_EVENTS),
                        self.router.sender(topics.LOCAL_EVENTS),
                    )
                    self._peer_event_sender = self.event_router.sender()

                if (
                    result.session_id.master_node_id == self.node_id
                    and self.master is not None
                ):
                    logger.info("Node elected Master")
                elif (
                    result.session_id.master_node_id == self.node_id
                    and self.master is None
                ):
                    logger.info("Node elected Master - promoting self")
                    self.master = Master(
                        self.node_id,
                        result.session_id,
                        event_sender=self.event_router.sender(),
                        global_event_sender=self.router.sender(topics.GLOBAL_EVENTS),
                        local_event_receiver=self.router.receiver(topics.LOCAL_EVENTS),
                        command_receiver=self.router.receiver(topics.COMMANDS),
                        download_command_sender=self.router.sender(
                            topics.DOWNLOAD_COMMANDS
                        ),
                    )
                    self._tg.start_soon(self.master.run)
                elif (
                    result.session_id.master_node_id != self.node_id
                    and self.master is not None
                ):
                    logger.info(
                        f"Node {result.session_id.master_node_id} elected master - demoting self"
                    )
                    await self.master.shutdown()
                    self.master = None
                else:
                    logger.info(
                        f"Node {result.session_id.master_node_id} elected master"
                    )
                if requires_session_transition:
                    if self.download_coordinator:
                        await self.download_coordinator.shutdown()
                        self.download_coordinator = DownloadCoordinator(
                            self.node_id,
                            cai_shard_downloader(offline=self.offline),
                            event_sender=self.event_router.sender(),
                            download_command_receiver=self.router.receiver(
                                topics.DOWNLOAD_COMMANDS
                            ),
                            offline=self.offline,
                        )
                        self._tg.start_soon(self.download_coordinator.run)
                    if self.worker:
                        await self.worker.shutdown()
                        # TODO: add profiling etc to resource monitor
                        self.worker = Worker(
                            self.node_id,
                            event_receiver=self.event_router.receiver(),
                            event_sender=self.event_router.sender(),
                            command_sender=self.router.sender(topics.COMMANDS),
                            download_command_sender=self.router.sender(
                                topics.DOWNLOAD_COMMANDS
                            ),
                            api_port=self._api_port,
                        )
                        self._tg.start_soon(self.worker.run)
                    elif self.info_publisher is not None:
                        self.info_publisher = NodeInfoPublisher(
                            node_id=self.node_id,
                            api_port=self._api_port,
                            event_sender=self.event_router.sender(),
                        )
                        self._tg.start_soon(self.info_publisher.run)
                    if self.api:
                        self.api.reset(
                            result.won_clock,
                            self.event_router.receiver(),
                            master_node_id=result.session_id.master_node_id,
                        )
                    self._tg.start_soon(self.event_router.run)
                    await self._replay_overlay_peer_state()
                    await self._advertise_overlay_bootstrap_peers()
                    if result.session_id.master_node_id != self.node_id:
                        await self._request_full_sync("session_changed_to_remote_master")
                else:
                    if self.api:
                        self.api.unpause(
                            result.won_clock,
                            master_node_id=result.session_id.master_node_id,
                        )


@dataclass
class NodeInfoPublisher:
    node_id: NodeId
    api_port: int
    event_sender: Sender[CAIEvent]
    _tg: TaskGroup = field(init=False, default_factory=TaskGroup)

    async def run(self):
        logger.info("Starting node info publisher")
        info_send, info_recv = channel[GatheredInfo]()
        info_gatherer = InfoGatherer(
            info_send,
            api_port=self.api_port,
            node_id=str(self.node_id),
        )

        try:
            async with self._tg as tg:
                tg.start_soon(info_gatherer.run)
                tg.start_soon(self._forward_info, info_recv)
        finally:
            logger.info("Stopping node info publisher")

    async def shutdown(self) -> None:
        self._tg.cancel_tasks()

    async def _forward_info(self, recv: Receiver[GatheredInfo]) -> None:
        with recv as info_stream:
            async for info in info_stream:
                try:
                    await self.event_sender.send(
                        NodeGatheredInfo(
                            node_id=self.node_id,
                            when=str(datetime.now(tz=timezone.utc)),
                            info=info,
                        )
                    )
                except (BrokenResourceError, ClosedResourceError):
                    logger.debug(
                        "node info publisher stopping because event channel is closed"
                    )
                    return


def main():
    args = Args.parse()
    _patch_windows_spawn_no_console()
    if resource is not None:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(max(soft, 65535), hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))

    mp.set_start_method("spawn", force=True)
    # TODO: Refactor the current verbosity system
    logger_setup(CAI_LOG, args.verbosity)
    logger.info(f"{'=' * 40}")
    logger.info(f"Starting CAI | pid={os.getpid()}")
    logger.info(f"{'=' * 40}")
    logger.info(
        "CAI_LIBP2P_NAMESPACE: {}",
        os.getenv("CAI_LIBP2P_NAMESPACE"),
    )

    instance_guard: NodeSingleInstanceGuard | None = None
    if should_enforce_node_single_instance():
        instance_guard = NodeSingleInstanceGuard()
        if not instance_guard.acquire():
            message = node_instance_error_message()
            print(message, file=sys.stderr)
            logger.error(message)
            logger_cleanup()
            return
        instance_guard.write_state(
            {
                "command": list(sys.argv),
                "apiPort": int(args.api_port),
                "libp2pPort": int(args.libp2p_port),
                "dashboardUrl": f"http://127.0.0.1:{args.api_port}/",
                "workerRuntimeEnabled": not args.no_worker,
                "offline": bool(args.offline),
            }
        )

    if args.offline:
        logger.info("Running in OFFLINE mode — no internet checks, local models only")

    if args.bootstrap_peers:
        logger.info(f"Bootstrap peers: {args.bootstrap_peers}")

    if args.no_batch:
        os.environ["CAI_NO_BATCH"] = "1"
        logger.info("Continuous batching disabled (--no-batch)")

    # Set FAST_SYNCH override env var for runner subprocesses
    if args.fast_synch is True:
        os.environ["CAI_FAST_SYNCH"] = "true"
        logger.info("FAST_SYNCH forced ON")
    elif args.fast_synch is False:
        os.environ["CAI_FAST_SYNCH"] = "false"
        logger.info("FAST_SYNCH forced OFF")

    try:
        node = anyio.run(Node.create, args)
        anyio.run(node.run)
    except BaseException as exception:
        logger.opt(exception=exception).critical(
            "CAI terminated due to unhandled exception"
        )
        raise
    finally:
        if instance_guard is not None:
            instance_guard.clear_state()
            instance_guard.release()
        logger.info("CAI shutdown complete")
        logger_cleanup()


class Args(CamelCaseModel):
    verbosity: int = 0
    force_master: bool = False
    spawn_api: bool = False
    api_port: PositiveInt = 52415
    tb_only: bool = False
    no_worker: bool = False
    no_downloads: bool = False
    offline: bool = (
        os.getenv("CAI_OFFLINE", None) or os.getenv("CAI_OFFLINE", "false")
    ).lower() == "true"
    no_batch: bool = False
    fast_synch: bool | None = None  # None = auto, True = force on, False = force off
    bootstrap_peers: list[str] = []
    libp2p_port: int

    @classmethod
    def parse(cls) -> Self:
        parser = argparse.ArgumentParser(prog="CAI")
        default_verbosity = 0
        parser.add_argument(
            "-q",
            "--quiet",
            action="store_const",
            const=-1,
            dest="verbosity",
            default=default_verbosity,
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="count",
            dest="verbosity",
            default=default_verbosity,
        )
        parser.add_argument(
            "-m",
            "--force-master",
            action="store_true",
            dest="force_master",
        )
        parser.add_argument(
            "--no-api",
            action="store_false",
            dest="spawn_api",
        )
        parser.add_argument(
            "--api-port",
            type=int,
            dest="api_port",
            default=52415,
        )
        parser.add_argument(
            "--no-worker",
            action="store_true",
        )
        parser.add_argument(
            "--no-downloads",
            action="store_true",
            help="Disable the download coordinator (node won't download models)",
        )
        parser.add_argument(
            "--offline",
            action="store_true",
            default=(
                os.getenv("CAI_OFFLINE", "false")
            ).lower() == "true",
            help="Run in offline/air-gapped mode: skip internet checks, use only pre-staged local models",
        )
        parser.add_argument(
            "--no-batch",
            action="store_true",
            help="Disable continuous batching, use sequential generation",
        )
        parser.add_argument(
            "--bootstrap-peers",
            type=lambda s: [p for p in s.split(",") if p],
            default=(
                os.getenv("CAI_BOOTSTRAP_PEERS")
                or ""
            ).split(",")
            if (
                os.getenv("CAI_BOOTSTRAP_PEERS")
            )
            else [],
            dest="bootstrap_peers",
            help="Comma-separated libp2p multiaddrs to dial on startup (env: CAI_BOOTSTRAP_PEERS)",
        )
        parser.add_argument(
            "--libp2p-port",
            type=int,
            default=0,
            dest="libp2p_port",
            help="Fixed TCP port for libp2p to listen on (0 = OS-assigned).",
        )
        fast_synch_group = parser.add_mutually_exclusive_group()
        fast_synch_group.add_argument(
            "--fast-synch",
            action="store_true",
            dest="fast_synch",
            default=None,
            help="Force MLX FAST_SYNCH on (for JACCL backend)",
        )
        fast_synch_group.add_argument(
            "--no-fast-synch",
            action="store_false",
            dest="fast_synch",
            help="Force MLX FAST_SYNCH off",
        )

        args = parser.parse_args()
        return cls(**vars(args))  # pyright: ignore[reportAny] - We are intentionally validating here, we can't do it statically


def _parse_advertised_bootstrap_peers() -> tuple[Multiaddr, ...]:
    raw_peers = os.getenv("CAI_ADVERTISE_PEERS") or os.getenv(
        "CAI_ADVERTISE_PEERS", ""
    )
    if not raw_peers.strip():
        return ()

    peers: list[Multiaddr] = []
    for line in raw_peers.splitlines():
        for part in line.split(","):
            candidate = part.strip()
            if not candidate:
                continue
            peers.append(Multiaddr(address=candidate))

    return tuple(peers)
