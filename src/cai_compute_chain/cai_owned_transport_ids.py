# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Any
import hashlib
import json

from .cai_owned_transport_protocol import (
    CAI_OWNED_TRANSPORT_PROTOCOL,
    CAI_OWNED_TRANSPORT_PROTOCOL_VERSION,
)


def cai_owned_transport_batch_id(
    *,
    session_id: str,
    phase: str,
    source_node_id: str,
    sink_node_id: str,
    sequence: int,
    payload_sha256_hex: str,
) -> str:
    batch_fingerprint = json.dumps(
        {
            "payloadSha256Hex": str(payload_sha256_hex or "").strip().lower(),
            "phase": str(phase or "").strip(),
            "sequence": max(0, int(sequence or 0)),
            "sessionId": str(session_id or "").strip(),
            "sinkNodeId": str(sink_node_id or "").strip(),
            "sourceNodeId": str(source_node_id or "").strip(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"caibatch_{hashlib.sha256(batch_fingerprint).hexdigest()[:24]}"


def cai_owned_transport_stage_id(
    *,
    session_id: str,
    phase: str,
    sequence: int,
    executor_node_id: str,
    layer_start: int,
    layer_end: int,
) -> str:
    stage_fingerprint = json.dumps(
        {
            "executorNodeId": str(executor_node_id or "").strip(),
            "layerEnd": int(layer_end),
            "layerStart": int(layer_start),
            "phase": str(phase or "").strip(),
            "protocol": CAI_OWNED_TRANSPORT_PROTOCOL,
            "protocolVersion": CAI_OWNED_TRANSPORT_PROTOCOL_VERSION,
            "sequence": max(0, int(sequence or 0)),
            "sessionId": str(session_id or "").strip(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"caistage_{hashlib.sha256(stage_fingerprint).hexdigest()[:24]}"


def cai_owned_transport_dag_hash(dag: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in dag.items()
        if key != "dagHashSha256Hex"
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
