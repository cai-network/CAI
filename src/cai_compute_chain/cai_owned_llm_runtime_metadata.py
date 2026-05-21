# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .gguf_shard_policy import (
    GGUF_SHARD_MODE_FULL_MODEL_LOCAL,
    GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED,
    GGUF_SHARD_MODE_UNSUPPORTED_FOR_SHARDING,
    gguf_shard_compatibility,
)


def cai_owned_transport_llm_runtime_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    model_id: str,
    total_layer_count: int,
    tokenizer_config_hash: str | None,
) -> dict[str, Any] | None:
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise ValueError("CAI-owned dispatch LLM runtime metadata is invalid.")
    resolved = dict(metadata)
    resolved.setdefault("modelId", str(model_id or "").strip())
    resolved.setdefault("totalLayerCount", int(total_layer_count))
    if tokenizer_config_hash:
        resolved.setdefault("tokenizerConfigHash", tokenizer_config_hash)
    return resolved


def runtime_metadata_text(metadata: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        raw = metadata.get(key)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    return None


def runtime_metadata_int(metadata: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        raw = metadata.get(key)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return None


def runtime_metadata_bool(metadata: Mapping[str, Any], *keys: str) -> bool | None:
    for key in keys:
        raw = metadata.get(key)
        if isinstance(raw, bool):
            return raw
        if raw is None or str(raw).strip() == "":
            continue
        value = str(raw).strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
    return None


def runtime_metadata_mapping(
    metadata: Mapping[str, Any],
    *keys: str,
) -> dict[str, Any] | None:
    for key in keys:
        raw = metadata.get(key)
        if raw is None:
            continue
        if not isinstance(raw, dict):
            raise ValueError(f"CAI-owned LLM runtime metadata {key} must be an object.")
        return _jsonable_dict(raw, field_name=f"runtimeMetadata.{key}")
    return None


def runtime_metadata_shape(metadata: Mapping[str, Any]) -> list[int] | None:
    raw = None
    for key in (
        "activationShape",
        "activation_shape",
        "tensorShape",
        "tensor_shape",
    ):
        if metadata.get(key) is not None:
            raw = metadata.get(key)
            break
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("CAI-owned LLM runtime metadata tensor shape is invalid.")
    shape: list[int] = []
    for item in raw:
        try:
            dimension = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "CAI-owned LLM runtime metadata tensor shape is invalid."
            ) from exc
        if dimension <= 0:
            raise ValueError(
                "CAI-owned LLM runtime metadata tensor shape is invalid."
            )
        shape.append(dimension)
    if not shape:
        raise ValueError("CAI-owned LLM runtime metadata tensor shape is invalid.")
    return shape


def require_runtime_metadata_layer_range_supported(
    runtime_metadata: Mapping[str, Any],
    *,
    model_id: str,
) -> None:
    shard_compatibility = runtime_metadata_text(
        runtime_metadata,
        "shardCompatibility",
        "shard_compatibility",
    )
    layer_range_supported = runtime_metadata_bool(
        runtime_metadata,
        "layerRangeSupported",
        "layer_range_supported",
    )
    if shard_compatibility == GGUF_SHARD_MODE_UNSUPPORTED_FOR_SHARDING:
        raise ValueError(
            "CAI-owned LLM runtime metadata is unsupported_for_sharding."
        )
    if shard_compatibility == GGUF_SHARD_MODE_FULL_MODEL_LOCAL:
        raise ValueError(
            "CAI-owned LLM runtime metadata is full_model_local; this is a "
            "single-node inference mode, not layer-range sharding."
        )
    if (
        shard_compatibility
        and shard_compatibility != GGUF_SHARD_MODE_LAYER_RANGE_SUPPORTED
    ):
        raise ValueError(
            "CAI-owned LLM runtime metadata does not support layer-range sharding."
        )
    if layer_range_supported is False:
        raise ValueError(
            "CAI-owned LLM runtime metadata layerRangeSupported is false."
        )

    gguf_architecture = runtime_metadata_text(
        runtime_metadata,
        "ggufArchitecture",
        "gguf_architecture",
    )
    if gguf_architecture:
        compatibility = gguf_shard_compatibility(
            model_id=model_id,
            gguf_architecture=gguf_architecture,
            family=runtime_metadata_text(runtime_metadata, "family"),
            filename=runtime_metadata_text(
                runtime_metadata,
                "preferredFilename",
                "preferred_filename",
            ),
        )
        if not compatibility.layer_range_supported:
            raise ValueError(
                "CAI-owned LLM runtime metadata is unsupported_for_sharding."
            )


def runtime_metadata_external_shard_descriptor(
    runtime_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in (
        "preferredFilename",
        "family",
        "quantization",
        "ggufArchitecture",
        "shardCompatibility",
        "layerRangeProbeAbi",
        "layerRangeProbeReport",
        "layerRangeEquivalenceProbeReport",
        "stateFormat",
        "activationStateFormat",
        "decodeStateFormat",
    ):
        value = runtime_metadata_text(runtime_metadata, key)
        if value:
            output[key] = value
    layer_range_supported = runtime_metadata_bool(
        runtime_metadata,
        "layerRangeSupported",
        "layer_range_supported",
    )
    if layer_range_supported is not None:
        output["layerRangeSupported"] = layer_range_supported
    context_length = runtime_metadata_int(
        runtime_metadata,
        "contextLength",
        "context_length",
        "nCtx",
        "n_ctx",
    )
    if context_length is not None and context_length > 0:
        output["contextLength"] = context_length
    return output


def _jsonable_dict(value: dict[str, Any], *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"CAI-owned transport {field_name} must be an object.")
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"CAI-owned transport {field_name} must be JSON serializable."
        ) from exc
