# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import os
import struct
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import suppress
from typing import cast
from urllib.parse import urlencode

import aiohttp
from aiohttp import ClientError, ClientSession, WSMsgType
from loguru import logger

from cai.shared.types.common import Host, NodeId
from cai.shared.types.worker.instances import LlamaCppRelayRoute


DEFAULT_RELAY_CONNECT_TIMEOUT_SECONDS = 5.0
DEFAULT_RELAY_DIRECT_CONNECT_TIMEOUT_SECONDS = max(
    float(os.getenv("CAI_RELAY_DIRECT_CONNECT_TIMEOUT_SECONDS", "1.5") or "1.5"),
    0.1,
)
DEFAULT_RELAY_STREAM_CHUNK_SIZE = max(
    int(os.getenv("CAI_RELAY_STREAM_CHUNK_SIZE", "16384") or "16384"),
    1024,
)
DEFAULT_REVERSE_RELAY_POOL_SIZE = max(
    int(os.getenv("CAI_REVERSE_RELAY_POOL_SIZE", "32") or "32"),
    1,
)
_RELAY_EOF_MESSAGE = "__cai_relay_eof__"
_RELAY_TARGET_CONNECTED_MESSAGE = "__cai_relay_target_connected__"
_LLAMA_CPP_RPC_CMD_HELLO = 14
_LLAMA_CPP_RPC_CONN_CAPS_SIZE = 24
_LLAMA_CPP_RPC_HELLO_RESPONSE_SIZE = 4 + _LLAMA_CPP_RPC_CONN_CAPS_SIZE


def _llama_cpp_rpc_hello_payload() -> bytes:
    return (
        bytes([_LLAMA_CPP_RPC_CMD_HELLO])
        + struct.pack("<Q", _LLAMA_CPP_RPC_CONN_CAPS_SIZE)
        + bytes(_LLAMA_CPP_RPC_CONN_CAPS_SIZE)
    )


def _parse_llama_cpp_rpc_hello_response(response: bytes) -> str:
    if len(response) != _LLAMA_CPP_RPC_HELLO_RESPONSE_SIZE:
        raise RuntimeError(f"invalid HELLO response length {len(response)}")
    major, minor, patch = response[0], response[1], response[2]
    if major <= 0:
        raise RuntimeError("invalid HELLO protocol version")
    return f"{major}.{minor}.{patch}"


async def _read_stream_exact(
    reader: asyncio.StreamReader,
    size: int,
    *,
    timeout: float,
) -> bytes:
    try:
        return await asyncio.wait_for(reader.readexactly(size), timeout=timeout)
    except asyncio.IncompleteReadError as exc:
        raise RuntimeError("connection closed during HELLO") from exc


async def _probe_llama_cpp_rpc_hello_stream(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    timeout: float,
) -> str:
    writer.write(_llama_cpp_rpc_hello_payload())
    await asyncio.wait_for(writer.drain(), timeout=timeout)
    response_size = struct.unpack(
        "<Q",
        await _read_stream_exact(reader, 8, timeout=timeout),
    )[0]
    if response_size != _LLAMA_CPP_RPC_HELLO_RESPONSE_SIZE:
        raise RuntimeError(f"unexpected HELLO response size {response_size}")
    response = await _read_stream_exact(reader, response_size, timeout=timeout)
    return _parse_llama_cpp_rpc_hello_response(response)


class LlamaCppRelayTunnelManager:
    def __init__(self, routes: Sequence[LlamaCppRelayRoute]):
        self._routes_by_sink: dict[NodeId, list[LlamaCppRelayRoute]] = {}
        for route in _dedupe_routes(routes):
            self._routes_by_sink.setdefault(route.sink_node_id, []).append(route)
        self._local_endpoints: dict[NodeId, Host] = {}
        self._servers: dict[NodeId, asyncio.AbstractServer] = {}
        self._preferred_route_by_sink: dict[
            NodeId,
            tuple[str, LlamaCppRelayRoute],
        ] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        self._started = threading.Event()
        self._stopped = threading.Event()
        self._startup_error: BaseException | None = None

    def start(self) -> None:
        if not self._routes_by_sink:
            return
        if self._thread is not None:
            return

        self._thread = threading.Thread(
            target=self._thread_main,
            name="llama-cpp-relay-tunnels",
            daemon=True,
        )
        self._thread.start()
        self._started.wait(timeout=10)
        if self._startup_error is not None:
            raise RuntimeError("Failed to start relay tunnel manager") from self._startup_error
        if self._loop is None:
            raise RuntimeError("Relay tunnel manager did not initialize an event loop")

    def stop(self) -> None:
        if self._thread is None or self._loop is None:
            return

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._async_shutdown(),
                self._loop,
            )
            future.result(timeout=10)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._stopped.wait(timeout=10)
            self._thread.join(timeout=10)
            self._thread = None
            self._loop = None
            self._session = None
            self._servers = {}
            self._local_endpoints = {}

    def local_endpoint_for_sink(self, sink_node_id: NodeId) -> Host | None:
        return self._local_endpoints.get(sink_node_id)

    def local_endpoints(self) -> Mapping[NodeId, Host]:
        return dict(self._local_endpoints)

    def selected_route_for_sink(
        self,
        sink_node_id: NodeId,
    ) -> tuple[str, LlamaCppRelayRoute] | None:
        return self._preferred_route_by_sink.get(sink_node_id)

    def probe_route(self, sink_node_id: NodeId, *, timeout: float) -> None:
        routes = self._routes_by_sink.get(sink_node_id)
        if not routes:
            raise ValueError(f"Unknown relay sink node: {sink_node_id}")
        if self._loop is None:
            raise RuntimeError("Relay tunnel manager is not running")

        errors: list[str] = []
        for route in routes:
            future = asyncio.run_coroutine_threadsafe(
                self._async_probe_route(route, timeout=timeout),
                self._loop,
            )
            try:
                future.result(timeout=timeout)
                return
            except FutureTimeoutError:
                future.cancel()
                errors.append(
                    f"{route.source_node_id}->{route.sink_node_id} via "
                    f"{route.transit_node_id}: timed out"
                )
            except Exception as exc:
                errors.append(
                    f"{route.source_node_id}->{route.sink_node_id} via "
                    f"{route.transit_node_id}: {exc}"
                )
        raise RuntimeError(
            f"No relay path to {sink_node_id} is ready: {'; '.join(errors)}"
        )

    def probe_llama_cpp_rpc_route(self, sink_node_id: NodeId, *, timeout: float) -> str:
        if sink_node_id not in self._local_endpoints:
            raise ValueError(f"Unknown local relay endpoint for sink node: {sink_node_id}")
        if self._loop is None:
            raise RuntimeError("Relay tunnel manager is not running")

        future = asyncio.run_coroutine_threadsafe(
            self._async_probe_llama_cpp_rpc_route(
                sink_node_id,
                timeout=timeout,
            ),
            self._loop,
        )
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            future.cancel()
            raise RuntimeError(
                f"Timed out probing llama.cpp RPC route to {sink_node_id}"
            ) from exc

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_start())
            self._started.set()
            loop.run_forever()
        except BaseException as exc:
            self._startup_error = exc
            self._started.set()
            raise
        finally:
            with suppress(Exception):
                loop.run_until_complete(self._async_shutdown())
            loop.close()
            self._stopped.set()

    async def _async_start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None)
        )
        for sink_node_id in self._routes_by_sink:
            server = await asyncio.start_server(
                lambda reader, writer, current_sink=sink_node_id: self._handle_client(
                    current_sink, reader, writer
                ),
                host="127.0.0.1",
                port=0,
            )
            sockets = server.sockets or []
            if not sockets:
                raise RuntimeError(
                    f"Relay tunnel listener did not expose a local socket for {sink_node_id}"
                )
            local_port = int(sockets[0].getsockname()[1])
            self._servers[sink_node_id] = server
            self._local_endpoints[sink_node_id] = Host(
                ip="127.0.0.1",
                port=local_port,
            )

    async def _async_shutdown(self) -> None:
        for server in self._servers.values():
            server.close()
        for server in self._servers.values():
            with suppress(Exception):
                await server.wait_closed()
        self._servers = {}
        if self._session is not None:
            with suppress(Exception):
                await self._session.close()
            self._session = None

    async def _async_probe_route(
        self,
        route: LlamaCppRelayRoute,
        *,
        timeout: float,
    ) -> None:
        try:
            _reader, writer = await self._connect_direct_stream(
                route,
                timeout=min(timeout, DEFAULT_RELAY_DIRECT_CONNECT_TIMEOUT_SECONDS),
            )
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()
            return
        except Exception:
            pass

        session = self._require_session()
        try:
            async with session.get(
                self._relay_probe_url(route),
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                if response.status < 200 or response.status >= 300:
                    message = await response.text()
                    raise RuntimeError(message or f"HTTP {response.status}")
                payload = await response.json()
                if not isinstance(payload, dict) or not payload.get("ready"):
                    raise RuntimeError(f"Relay probe returned not ready: {payload}")
        except Exception as exc:
            raise RuntimeError(
                f"Direct target and relay path {route.source_node_id}->{route.transit_node_id}->{route.sink_node_id} are not ready"
            ) from exc

    async def _async_probe_llama_cpp_rpc_route(
        self,
        sink_node_id: NodeId,
        *,
        timeout: float,
    ) -> str:
        await self._async_select_llama_cpp_rpc_route(
            sink_node_id,
            timeout=timeout,
        )
        return await self._async_probe_local_llama_cpp_rpc_route(
            sink_node_id,
            timeout=timeout,
        )

    async def _async_select_llama_cpp_rpc_route(
        self,
        sink_node_id: NodeId,
        *,
        timeout: float,
    ) -> str:
        routes = self._routes_by_sink.get(sink_node_id)
        if not routes:
            raise ValueError(f"Unknown relay sink node: {sink_node_id}")

        errors: list[str] = []
        for route in routes:
            try:
                reader, writer = await self._connect_direct_stream(
                    route,
                    timeout=min(timeout, DEFAULT_RELAY_DIRECT_CONNECT_TIMEOUT_SECONDS),
                )
                try:
                    version = await _probe_llama_cpp_rpc_hello_stream(
                        reader,
                        writer,
                        timeout=timeout,
                    )
                finally:
                    writer.close()
                    with suppress(Exception):
                        await writer.wait_closed()
                self._preferred_route_by_sink[sink_node_id] = ("direct", route)
                return version
            except Exception as exc:
                errors.append(
                    f"direct {route.target_host}:{route.target_port}: {exc}"
                )

            try:
                version = await self._async_probe_relay_llama_cpp_rpc_route(
                    route,
                    timeout=timeout,
                )
                self._preferred_route_by_sink[sink_node_id] = ("relay", route)
                return version
            except Exception as exc:
                errors.append(
                    f"relay {route.source_node_id}->{route.transit_node_id}->{route.sink_node_id}: {exc}"
                )

        raise RuntimeError(
            f"No llama.cpp RPC protocol-ready path to {sink_node_id}: "
            + "; ".join(errors)
        )

    async def _async_probe_relay_llama_cpp_rpc_route(
        self,
        route: LlamaCppRelayRoute,
        *,
        timeout: float,
    ) -> str:
        session = self._require_session()
        async with session.get(
            self._relay_probe_url(route, protocol="llama_cpp_rpc"),
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            if response.status < 200 or response.status >= 300:
                message = await response.text()
                raise RuntimeError(message or f"HTTP {response.status}")
            payload = await response.json()
            if not isinstance(payload, dict) or not payload.get("ready"):
                raise RuntimeError(f"Relay probe returned not ready: {payload}")
            if not payload.get("protocolReady"):
                raise RuntimeError(f"Relay RPC protocol probe failed: {payload}")
            return str(payload.get("protocolVersion") or "unknown")

    async def _async_probe_local_llama_cpp_rpc_route(
        self,
        sink_node_id: NodeId,
        *,
        timeout: float,
    ) -> str:
        endpoint = self._local_endpoints.get(sink_node_id)
        if endpoint is None:
            raise ValueError(f"Unknown local relay endpoint for sink node: {sink_node_id}")

        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(endpoint.ip, endpoint.port),
            timeout=timeout,
        )
        try:
            return await _probe_llama_cpp_rpc_hello_stream(
                reader,
                writer,
                timeout=timeout,
            )
        finally:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()

    async def _handle_client(
        self,
        sink_node_id: NodeId,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        routes = self._routes_by_sink.get(sink_node_id, [])
        try:
            preferred = self._preferred_route_by_sink.get(sink_node_id)
            if preferred is not None:
                mode, preferred_route = preferred
                preferred_routes = [preferred_route] + [
                    route for route in routes if route != preferred_route
                ]
                if mode == "relay":
                    websocket = await self._connect_first_ready_websocket(
                        preferred_routes
                    )
                    try:
                        async with asyncio.TaskGroup() as tg:
                            tg.create_task(self._pipe_local_to_websocket(reader, websocket))
                            tg.create_task(self._pipe_websocket_to_local(websocket, writer))
                    finally:
                        await websocket.close()
                    return

                if mode == "direct":
                    try:
                        remote_reader, remote_writer = await self._connect_direct_stream(
                            preferred_route,
                            timeout=DEFAULT_RELAY_DIRECT_CONNECT_TIMEOUT_SECONDS,
                        )
                        try:
                            async with asyncio.TaskGroup() as tg:
                                tg.create_task(
                                    self._pipe_stream_to_stream(reader, remote_writer)
                                )
                                tg.create_task(
                                    self._pipe_stream_to_stream(remote_reader, writer)
                                )
                        finally:
                            remote_writer.close()
                            with suppress(Exception):
                                await remote_writer.wait_closed()
                        return
                    except Exception as exc:
                        logger.debug(
                            "Preferred direct llama.cpp RPC route {}->{} failed: {}",
                            preferred_route.source_node_id,
                            preferred_route.sink_node_id,
                            exc,
                        )

            direct_stream = await self._connect_first_ready_direct_stream(routes)
            if direct_stream is not None:
                remote_reader, remote_writer = direct_stream
                try:
                    async with asyncio.TaskGroup() as tg:
                        tg.create_task(self._pipe_stream_to_stream(reader, remote_writer))
                        tg.create_task(self._pipe_stream_to_stream(remote_reader, writer))
                finally:
                    remote_writer.close()
                    with suppress(Exception):
                        await remote_writer.wait_closed()
                return

            websocket = await self._connect_first_ready_websocket(routes)
            try:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self._pipe_local_to_websocket(reader, websocket))
                    tg.create_task(self._pipe_websocket_to_local(websocket, writer))
            finally:
                await websocket.close()
        finally:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()

    async def _connect_first_ready_direct_stream(
        self,
        routes: Sequence[LlamaCppRelayRoute],
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter] | None:
        errors: list[str] = []
        for route in routes:
            try:
                stream = await self._connect_direct_stream(
                    route,
                    timeout=DEFAULT_RELAY_DIRECT_CONNECT_TIMEOUT_SECONDS,
                )
                logger.debug(
                    "Using direct llama.cpp RPC target {}:{} for {}->{}",
                    route.target_host,
                    route.target_port,
                    route.source_node_id,
                    route.sink_node_id,
                )
                return stream
            except Exception as exc:
                errors.append(
                    f"{route.source_node_id}->{route.sink_node_id} direct "
                    f"{route.target_host}:{route.target_port}: {exc}"
                )
        if errors:
            logger.debug(
                "No direct llama.cpp RPC target is ready, falling back to relay: {}",
                "; ".join(errors),
            )
        return None

    async def _connect_direct_stream(
        self,
        route: LlamaCppRelayRoute,
        *,
        timeout: float,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        host = str(route.target_host or "").strip()
        port = int(route.target_port or 0)
        if not _is_dialable_direct_target(host, port):
            raise RuntimeError(f"{host}:{port} is not a dialable direct target")
        return await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout,
        )

    async def _connect_first_ready_websocket(
        self,
        routes: Sequence[LlamaCppRelayRoute],
    ) -> aiohttp.ClientWebSocketResponse:
        session = self._require_session()
        errors: list[str] = []
        for route in routes:
            websocket: aiohttp.ClientWebSocketResponse | None = None
            try:
                websocket = await session.ws_connect(
                    self._relay_ws_url(route),
                    **_relay_ws_connect_kwargs(),
                )
                await self._expect_connected(
                    websocket,
                    timeout=DEFAULT_RELAY_CONNECT_TIMEOUT_SECONDS,
                )
                return websocket
            except Exception as exc:
                if websocket is not None:
                    with suppress(Exception):
                        await websocket.close()
                errors.append(
                    f"{route.source_node_id}->{route.sink_node_id} via "
                    f"{route.transit_node_id}: {exc}"
                )
        raise RuntimeError(f"No relay websocket path is ready: {'; '.join(errors)}")

    async def _pipe_stream_to_stream(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        while True:
            chunk = await reader.read(DEFAULT_RELAY_STREAM_CHUNK_SIZE)
            if not chunk:
                await _write_stream_eof(writer)
                return
            writer.write(chunk)
            await writer.drain()

    async def _pipe_local_to_websocket(
        self,
        reader: asyncio.StreamReader,
        websocket: aiohttp.ClientWebSocketResponse,
    ) -> None:
        while True:
            chunk = await reader.read(DEFAULT_RELAY_STREAM_CHUNK_SIZE)
            if not chunk:
                await websocket.send_str(_RELAY_EOF_MESSAGE)
                return
            await websocket.send_bytes(chunk)

    async def _pipe_websocket_to_local(
        self,
        websocket: aiohttp.ClientWebSocketResponse,
        writer: asyncio.StreamWriter,
    ) -> None:
        while True:
            message = await websocket.receive()
            if message.type == WSMsgType.BINARY:
                writer.write(cast(bytes, message.data))
                await writer.drain()
                continue
            if message.type in {WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED}:
                return
            if message.type == WSMsgType.ERROR:
                raise websocket.exception() or RuntimeError(
                    "Relay websocket closed with an error"
                )
            if message.type == WSMsgType.TEXT:
                payload = str(message.data or "").strip()
                if payload == _RELAY_EOF_MESSAGE:
                    await _write_stream_eof(writer)
                    return
                if payload.lower() == "connected":
                    continue
                raise RuntimeError(f"Unexpected relay websocket message: {payload}")
            if message.type in {WSMsgType.PING, WSMsgType.PONG}:
                continue
            raise RuntimeError(f"Unexpected relay websocket frame: {message.type}")

    async def _expect_connected(
        self,
        websocket: aiohttp.ClientWebSocketResponse,
        *,
        timeout: float | None = None,
    ) -> None:
        if timeout is None:
            message = await websocket.receive()
        else:
            message = await asyncio.wait_for(websocket.receive(), timeout=timeout)
        if message.type == WSMsgType.TEXT:
            payload = str(message.data or "").strip()
            if payload.lower() == "connected":
                return
            raise RuntimeError(payload or "Relay websocket returned a non-ready status")
        if message.type == WSMsgType.ERROR:
            raise websocket.exception() or RuntimeError(
                "Relay websocket failed before signaling readiness"
            )
        raise RuntimeError(
            f"Relay websocket did not confirm readiness: {message.type}"
        )

    def _relay_ws_url(self, route: LlamaCppRelayRoute) -> str:
        host = str(route.relay_api_host).strip()
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        query = urlencode(
            {
                "source_node_id": str(route.source_node_id),
                "transit_node_id": str(route.transit_node_id),
                "sink_node_id": str(route.sink_node_id),
                "target_host": route.target_host,
                "target_port": route.target_port,
            }
        )
        return f"ws://{host}:{route.relay_api_port}/v1/cai/relay/rpc/ws?{query}"

    def _relay_probe_url(
        self,
        route: LlamaCppRelayRoute,
        *,
        protocol: str | None = None,
    ) -> str:
        host = str(route.relay_api_host).strip()
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        query_params = {
            "source_node_id": str(route.source_node_id),
            "transit_node_id": str(route.transit_node_id),
            "sink_node_id": str(route.sink_node_id),
            "target_host": route.target_host,
            "target_port": route.target_port,
        }
        if protocol:
            query_params["protocol"] = protocol
        query = urlencode(query_params)
        return f"http://{host}:{route.relay_api_port}/v1/cai/relay/rpc/probe?{query}"

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("Relay tunnel manager session is not available")
        return self._session


class LlamaCppReverseRelayManager:
    def __init__(self, routes: Sequence[LlamaCppRelayRoute]):
        self._routes = list(_dedupe_routes(routes))
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._started = threading.Event()
        self._stopped = threading.Event()
        self._startup_error: BaseException | None = None
        self._closing = False

    def start(self) -> None:
        if not self._routes:
            return
        if self._thread is not None:
            return

        self._closing = False
        self._thread = threading.Thread(
            target=self._thread_main,
            name="llama-cpp-reverse-relay-tunnels",
            daemon=True,
        )
        self._thread.start()
        self._started.wait(timeout=10)
        if self._startup_error is not None:
            raise RuntimeError("Failed to start reverse relay tunnel manager") from self._startup_error
        if self._loop is None:
            raise RuntimeError("Reverse relay tunnel manager did not initialize an event loop")

    def stop(self) -> None:
        if self._thread is None or self._loop is None:
            return

        self._closing = True
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._async_shutdown(),
                self._loop,
            )
            future.result(timeout=10)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._stopped.wait(timeout=10)
            self._thread.join(timeout=10)
            self._thread = None
            self._loop = None
            self._session = None
            self._tasks = set()

    def route_count(self) -> int:
        return len(self._routes)

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_start())
            self._started.set()
            loop.run_forever()
        except BaseException as exc:
            self._startup_error = exc
            self._started.set()
            raise
        finally:
            with suppress(Exception):
                loop.run_until_complete(self._async_shutdown())
            loop.close()
            self._stopped.set()

    async def _async_start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None)
        )
        for route in self._routes:
            for _ in range(DEFAULT_REVERSE_RELAY_POOL_SIZE):
                task = asyncio.create_task(self._maintain_reverse_route(route))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

    async def _async_shutdown(self) -> None:
        self._closing = True
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            with suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks = set()
        if self._session is not None:
            with suppress(Exception):
                await self._session.close()
            self._session = None

    async def _maintain_reverse_route(self, route: LlamaCppRelayRoute) -> None:
        backoff_seconds = 0.5
        while not self._closing:
            try:
                await self._serve_reverse_route(route)
                backoff_seconds = 0.5
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._closing:
                    logger.debug(
                        "Reverse relay route {}->{} via {} disconnected: {}",
                        route.source_node_id,
                        route.sink_node_id,
                        route.transit_node_id,
                        exc,
                    )
                    await asyncio.sleep(backoff_seconds)
                    backoff_seconds = min(backoff_seconds * 2, 5.0)

    async def _serve_reverse_route(self, route: LlamaCppRelayRoute) -> None:
        session = self._require_session()
        async with session.ws_connect(
            self._reverse_relay_ws_url(route),
            **_relay_ws_connect_kwargs(),
        ) as websocket:
            connected_immediately = await self._expect_registered_or_connected(websocket)
            reader, writer = await asyncio.open_connection(
                "127.0.0.1",
                route.target_port,
            )
            await websocket.send_str(_RELAY_TARGET_CONNECTED_MESSAGE)
            if not connected_immediately:
                await self._expect_connected(websocket)
            try:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self._pipe_local_to_websocket(reader, websocket))
                    tg.create_task(self._pipe_websocket_to_local(websocket, writer))
            finally:
                writer.close()
                with suppress(Exception):
                    await writer.wait_closed()

    async def _pipe_local_to_websocket(
        self,
        reader: asyncio.StreamReader,
        websocket: aiohttp.ClientWebSocketResponse,
    ) -> None:
        while True:
            chunk = await reader.read(DEFAULT_RELAY_STREAM_CHUNK_SIZE)
            if not chunk:
                await websocket.send_str(_RELAY_EOF_MESSAGE)
                return
            await websocket.send_bytes(chunk)

    async def _pipe_websocket_to_local(
        self,
        websocket: aiohttp.ClientWebSocketResponse,
        writer: asyncio.StreamWriter,
    ) -> None:
        while True:
            message = await websocket.receive()
            if message.type == WSMsgType.BINARY:
                writer.write(cast(bytes, message.data))
                await writer.drain()
                continue
            if message.type in {WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED}:
                return
            if message.type == WSMsgType.ERROR:
                raise websocket.exception() or RuntimeError(
                    "Reverse relay websocket closed with an error"
                )
            if message.type == WSMsgType.TEXT:
                payload = str(message.data or "").strip()
                if payload == _RELAY_EOF_MESSAGE:
                    await _write_stream_eof(writer)
                    return
                if payload.lower() in {"connected", "registered"}:
                    continue
                raise RuntimeError(f"Unexpected reverse relay websocket message: {payload}")
            if message.type in {WSMsgType.PING, WSMsgType.PONG}:
                continue
            raise RuntimeError(f"Unexpected reverse relay websocket frame: {message.type}")

    async def _expect_registered_or_connected(
        self,
        websocket: aiohttp.ClientWebSocketResponse,
    ) -> bool:
        message = await websocket.receive()
        if message.type == WSMsgType.TEXT:
            payload = str(message.data or "").strip().lower()
            if payload == "connected":
                return True
            if payload == "registered":
                return False
            raise RuntimeError(payload or "Reverse relay websocket returned a non-ready status")
        if message.type == WSMsgType.ERROR:
            raise websocket.exception() or RuntimeError(
                "Reverse relay websocket failed before registration"
            )
        raise RuntimeError(
            f"Reverse relay websocket did not confirm registration: {message.type}"
        )

    async def _expect_connected(
        self,
        websocket: aiohttp.ClientWebSocketResponse,
    ) -> None:
        message = await websocket.receive()
        if message.type == WSMsgType.TEXT:
            payload = str(message.data or "").strip()
            if payload.lower() == "connected":
                return
            raise RuntimeError(payload or "Reverse relay websocket returned a non-ready status")
        if message.type == WSMsgType.ERROR:
            raise websocket.exception() or RuntimeError(
                "Reverse relay websocket failed before signaling readiness"
            )
        raise RuntimeError(
            f"Reverse relay websocket did not confirm readiness: {message.type}"
        )

    def _reverse_relay_ws_url(self, route: LlamaCppRelayRoute) -> str:
        host = str(route.relay_api_host).strip()
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        query = urlencode(
            {
                "source_node_id": str(route.source_node_id),
                "transit_node_id": str(route.transit_node_id),
                "sink_node_id": str(route.sink_node_id),
                "target_host": route.target_host,
                "target_port": route.target_port,
            }
        )
        return f"ws://{host}:{route.relay_api_port}/v1/cai/relay/rpc/reverse/ws?{query}"

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("Reverse relay tunnel manager session is not available")
        return self._session


def _dedupe_routes(routes: Sequence[LlamaCppRelayRoute]) -> list[LlamaCppRelayRoute]:
    deduped: list[LlamaCppRelayRoute] = []
    seen: set[tuple[str, int, str, str, int]] = set()
    for route in routes:
        key = (
            str(route.relay_api_host),
            int(route.relay_api_port),
            str(route.sink_node_id),
            str(route.target_host),
            int(route.target_port),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(route)
    return deduped


def _relay_ws_connect_kwargs() -> dict[str, object]:
    return {
        "heartbeat": 30,
        "max_msg_size": 0,
        "compress": 0,
    }


def _is_dialable_direct_target(host: str, port: int) -> bool:
    if port <= 0:
        return False
    if not host:
        return False
    if host in {"0.0.0.0", "::", "198.51.100.1"}:
        return False
    return True


async def _write_stream_eof(writer: asyncio.StreamWriter) -> None:
    if writer.can_write_eof():
        writer.write_eof()
        await writer.drain()
        return
    writer.close()
    with suppress(Exception):
        await writer.wait_closed()

