# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import base64
from collections.abc import Sequence
import gzip
from typing import Any

from .cai_owned_transport_protocol import CAI_OWNED_TRANSPORT_PAYLOAD_COMPRESSIONS


def decode_cai_owned_transport_batch_payload(envelope: dict[str, Any]) -> bytes:
    encoded_payload_bytes = encoded_cai_owned_transport_payload_bytes_from_envelope(
        envelope,
    )
    compression = str(envelope.get("payloadCompression") or "").strip().lower()
    if compression and compression not in CAI_OWNED_TRANSPORT_PAYLOAD_COMPRESSIONS:
        raise ValueError(
            "CAI-owned transport batch envelope payload compression is unsupported."
        )
    return decode_cai_owned_transport_payload_bytes(
        encoded_payload_bytes,
        compression=compression or None,
    )


def encoded_cai_owned_transport_payload_bytes_from_envelope(
    envelope: dict[str, Any],
) -> bytes:
    payload_base64 = envelope.get("payloadBase64")
    payload_chunks_base64 = envelope.get("payloadChunksBase64")
    has_single_payload = payload_base64 is not None
    has_chunked_payload = payload_chunks_base64 is not None
    if has_single_payload and has_chunked_payload:
        raise ValueError("CAI-owned transport batch envelope payload is ambiguous.")
    if has_single_payload:
        if not isinstance(payload_base64, str):
            raise ValueError("CAI-owned transport batch envelope payload is missing.")
        return _decode_cai_owned_transport_base64_payload_chunk(payload_base64)
    if has_chunked_payload:
        if not isinstance(payload_chunks_base64, Sequence) or isinstance(
            payload_chunks_base64,
            (str, bytes, bytearray),
        ):
            raise ValueError(
                "CAI-owned transport batch envelope payload chunks are invalid."
            )
        chunks = list(payload_chunks_base64)
        if not chunks:
            raise ValueError(
                "CAI-owned transport batch envelope payload chunks are missing."
            )
        declared_count = _optional_int(envelope.get("payloadChunkCount"))
        if declared_count is not None and declared_count != len(chunks):
            raise ValueError(
                "CAI-owned transport batch envelope payload chunk count does not match."
            )
        decoded_chunks = [
            _decode_cai_owned_transport_base64_payload_chunk(chunk)
            for chunk in chunks
        ]
        return b"".join(decoded_chunks)
    raise ValueError("CAI-owned transport batch envelope payload is missing.")


def cai_owned_transport_encoded_payload_fields(
    encoded_payload_bytes: bytes,
    *,
    chunk_size_bytes: int | None,
) -> dict[str, Any]:
    payload_bytes = bytes(encoded_payload_bytes or b"")
    chunk_size = _optional_int(chunk_size_bytes)
    if chunk_size is None or chunk_size <= 0 or len(payload_bytes) <= chunk_size:
        return {
            "payloadBase64": base64.b64encode(payload_bytes).decode("ascii"),
        }
    chunks = [
        base64.b64encode(payload_bytes[index : index + chunk_size]).decode("ascii")
        for index in range(0, len(payload_bytes), chunk_size)
    ]
    return {
        "payloadChunksBase64": chunks,
        "payloadChunkCount": len(chunks),
        "payloadChunkSizeBytes": chunk_size,
    }


def normalize_cai_owned_transport_payload_compression(
    value: str | None,
) -> str | None:
    clean = str(value or "").strip().lower()
    if not clean or clean == "none":
        return None
    if clean not in CAI_OWNED_TRANSPORT_PAYLOAD_COMPRESSIONS:
        raise ValueError(
            "CAI-owned transport batch envelope payload compression is unsupported."
        )
    return clean


def encode_cai_owned_transport_payload_bytes(
    payload: bytes,
    *,
    compression: str | None,
) -> bytes:
    if compression is None:
        return bytes(payload or b"")
    if compression == "gzip":
        return gzip.compress(bytes(payload or b""))
    raise ValueError(
        "CAI-owned transport batch envelope payload compression is unsupported."
    )


def decode_cai_owned_transport_payload_bytes(
    payload: bytes,
    *,
    compression: str | None,
) -> bytes:
    if compression is None:
        return bytes(payload or b"")
    if compression == "gzip":
        try:
            return gzip.decompress(bytes(payload or b""))
        except Exception as exc:
            raise ValueError(
                "CAI-owned transport batch envelope payload compression is invalid."
            ) from exc
    raise ValueError(
        "CAI-owned transport batch envelope payload compression is unsupported."
    )


def _decode_cai_owned_transport_base64_payload_chunk(value: object) -> bytes:
    if not isinstance(value, str):
        raise ValueError(
            "CAI-owned transport batch envelope payload is not valid base64."
        )
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except Exception as exc:
        raise ValueError(
            "CAI-owned transport batch envelope payload is not valid base64."
        ) from exc


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
