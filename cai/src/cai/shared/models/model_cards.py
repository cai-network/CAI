# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Mapping
from enum import Enum
from pathlib import Path as FsPath
from typing import Annotated, Any

import aiofiles
import aiofiles.os as aios
import tomlkit
from anyio import Path, open_file
from huggingface_hub import model_info
from loguru import logger
from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    PositiveInt,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from tomlkit.exceptions import TOMLKitError

from cai_compute_chain.gguf_shard_policy import gguf_shard_compatibility

from cai.shared.constants import (
    CAI_CUSTOM_MODEL_CARDS_DIR,
    CAI_DATA_HOME,
    CAI_ENABLE_IMAGE_MODELS,
    CAI_MODELS_DIRS,
    RESOURCES_DIR,
)
from cai.shared.types.common import ModelId
from cai.shared.types.memory import Memory
from cai.utils.pydantic_ext import CamelCaseModel

_custom_cards_dir = Path(str(CAI_CUSTOM_MODEL_CARDS_DIR))
_custom_model_paths_file = FsPath(CAI_DATA_HOME) / "custom_model_paths.json"
_BUILTIN_CARD_DIRS = [
    Path(RESOURCES_DIR) / "inference_model_cards",
]

_card_cache: dict[ModelId, "ModelCard"] = {}
_SUPPORTED_BACKENDS = {"llama_cpp"}
_GGUF_SHARD_POLICY_FIELD_DEFAULTS: dict[str, object] = {
    "gguf_architecture": "",
    "shard_compatibility": "",
    "layer_range_supported": False,
    "layer_range_probe_abi": None,
    "layer_range_probe_report": None,
    "layer_range_equivalence_probe_report": None,
    "state_format": None,
    "activation_state_format": None,
    "decode_state_format": None,
    "shard_compatibility_reason": "",
}


def _normalize_local_model_source_path(local_path: str | FsPath) -> FsPath:
    raw = str(local_path).strip()
    if not raw:
        raise ValueError("Local model path cannot be empty")

    expanded = os.path.expanduser(raw)
    if os.name != "nt":
        windows_path_match = re.match(r"^([A-Za-z]):[\\/](.+)$", expanded)
        if windows_path_match:
            drive = windows_path_match.group(1).lower()
            rest = windows_path_match.group(2).replace("\\", "/")
            expanded = f"/mnt/{drive}/{rest}"

    return FsPath(expanded).expanduser()


def _load_custom_model_paths() -> dict[str, str]:
    if not _custom_model_paths_file.exists():
        return {}
    try:
        payload = json.loads(_custom_model_paths_file.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return {
                str(key): str(value)
                for key, value in payload.items()
                if str(key).strip() and str(value).strip()
            }
    except Exception:
        logger.warning(
            "Failed to parse custom model path registry: {}", _custom_model_paths_file
        )
    return {}


def _save_custom_model_paths(entries: dict[str, str]) -> None:
    _custom_model_paths_file.parent.mkdir(parents=True, exist_ok=True)
    _custom_model_paths_file.write_text(
        json.dumps(entries, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def get_custom_model_local_path(model_id: ModelId) -> FsPath | None:
    stored = _load_custom_model_paths().get(str(model_id))
    if not stored:
        return None
    return _normalize_local_model_source_path(stored)


def set_custom_model_local_path(model_id: ModelId, local_path: str | FsPath) -> FsPath:
    resolved = _normalize_local_model_source_path(local_path)
    if not resolved.exists():
        raise ValueError(f"Local model path does not exist: {resolved}")
    if not resolved.is_dir() and not resolved.is_file():
        raise ValueError(f"Local model path is not a file or directory: {resolved}")

    entries = _load_custom_model_paths()
    entries[str(ModelId(model_id))] = str(resolved)
    _save_custom_model_paths(entries)
    return resolved


def delete_custom_model_local_path(model_id: ModelId) -> bool:
    entries = _load_custom_model_paths()
    removed = entries.pop(str(ModelId(model_id)), None)
    if removed is None:
        return False
    _save_custom_model_paths(entries)
    return True


def derive_custom_model_id_from_local_path(local_path: str | FsPath) -> ModelId:
    resolved = _normalize_local_model_source_path(local_path)
    stem = resolved.name or "model"
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-_.").lower()
    return ModelId(f"local/{safe_stem or 'model'}")


def _find_local_gguf_path(local_path: FsPath) -> FsPath | None:
    if local_path.is_file():
        return local_path if local_path.suffix.lower() == ".gguf" else None

    gguf_candidates = sorted(local_path.glob("*.gguf"))
    if gguf_candidates:
        return gguf_candidates[0]

    recursive_candidates = sorted(local_path.glob("**/*.gguf"))
    if recursive_candidates:
        return recursive_candidates[0]

    return None


def _derive_family_from_filename(name: str) -> str:
    lowered = name.lower()
    for family in ("qwen", "llama", "deepseek", "mistral", "gemma", "phi"):
        if family in lowered:
            return family
    return ""


def _derive_quantization_from_filename(name: str) -> str:
    lowered = name.lower()
    for pattern in (
        r"(iq\d(?:_[a-z0-9]+)+)",
        r"(q\d(?:_[a-z0-9]+)+)",
        r"(fp16|fp8|bf16|f16|f32)",
    ):
        match = re.search(pattern, lowered)
        if match:
            return match.group(1).upper()
    return ""


def _extract_base_model_from_tags(tags: list[str] | None) -> str:
    if not tags:
        return ""

    quantized_fallback = ""
    for raw_tag in tags:
        tag = str(raw_tag)
        if tag.startswith("base_model:quantized:"):
            quantized_fallback = tag.removeprefix("base_model:quantized:")
        elif tag.startswith("base_model:"):
            return tag.removeprefix("base_model:")

    return quantized_fallback


def _gguf_shard_policy_fields(
    *,
    model_id: ModelId | str,
    family: str = "",
    gguf_architecture: str | None = None,
    filename: str | None = None,
) -> dict[str, object]:
    compatibility = gguf_shard_compatibility(
        model_id=model_id,
        gguf_architecture=gguf_architecture or family,
        family=family,
        filename=filename,
        allow_full_model_local=False,
    )
    metadata = compatibility.to_metadata()
    fields = dict(_GGUF_SHARD_POLICY_FIELD_DEFAULTS)
    for key, value in metadata.items():
        if key in fields:
            fields[key] = value
    return fields


def _select_preferred_gguf_filename(filenames: list[str]) -> str | None:
    if not filenames:
        return None

    preference_order = (
        "q4_k_m",
        "q4_k_s",
        "q4_0",
        "q5_k_m",
        "q5_0",
        "q6_k",
        "q8_0",
        "q3_k_m",
        "q3_k_s",
        "q2_k",
        "fp16",
    )
    lowered_filenames = [(name, name.lower()) for name in filenames]
    for preferred_quant in preference_order:
        for original_name, lowered_name in lowered_filenames:
            if preferred_quant in lowered_name:
                return original_name
    return sorted(filenames)[0]


def _infer_gguf_layer_count(
    *,
    model_id: ModelId,
    architecture: str,
    filename: str,
    metadata: Mapping[str, Any] | None = None,
) -> int:
    metadata = metadata or {}
    for key in (
        "block_count",
        "n_layer",
        "layers",
        "num_hidden_layers",
        "transformer.block_count",
    ):
        value = metadata.get(key)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = 0
        if parsed > 1:
            return parsed

    normalized = " ".join(
        part.lower()
        for part in (architecture, filename, str(model_id))
        if part
    )
    if any(
        token in normalized
        for token in ("qwen2", "qwen2.5", "qwen2_5", "qwen1.5", "qwen1_5")
    ):
        return 24
    if "qwen3" in normalized or "qwen" in normalized:
        return 28
    if "gemma" in normalized:
        return 28
    if any(token in normalized for token in ("llama", "mistral", "phi")):
        return 32
    return 32


def _build_hf_gguf_model_card(model_id: ModelId, info: Any) -> "ModelCard":
    gguf_files = [
        sibling
        for sibling in getattr(info, "siblings", [])
        if str(getattr(sibling, "rfilename", "")).lower().endswith(".gguf")
    ]
    preferred_filename = _select_preferred_gguf_filename(
        [str(sibling.rfilename) for sibling in gguf_files]
    )
    if preferred_filename is None:
        raise ValueError(f"No GGUF files found for {model_id}")

    preferred_sibling = next(
        (
            sibling
            for sibling in gguf_files
            if str(getattr(sibling, "rfilename", "")) == preferred_filename
        ),
        None,
    )
    gguf_metadata = getattr(info, "gguf", {}) or {}
    sibling_size = getattr(preferred_sibling, "size", None)
    if sibling_size is None and preferred_sibling is not None:
        sibling_size = getattr(getattr(preferred_sibling, "lfs", None), "size", None)
    size_bytes = int(sibling_size or gguf_metadata.get("total") or 0)

    family = str(gguf_metadata.get("architecture", "") or "")
    if not family:
        family = _derive_family_from_filename(preferred_filename)

    return ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_bytes(size_bytes),
        n_layers=_infer_gguf_layer_count(
            model_id=model_id,
            architecture=family,
            filename=preferred_filename,
            metadata=gguf_metadata,
        ),
        hidden_size=1,
        supports_tensor=False,
        context_length=int(gguf_metadata.get("context_length", 0) or 0),
        tasks=[ModelTask.TextGeneration],
        trust_remote_code=False,
        is_custom=False,
        inference_backend=InferenceBackend.LlamaCpp,
        family=family,
        quantization=_derive_quantization_from_filename(preferred_filename),
        base_model=_extract_base_model_from_tags(getattr(info, "tags", None)),
        preferred_filename=preferred_filename,
    )


async def _load_cards_from_dir(directory: Path, *, is_custom: bool) -> None:
    if not await directory.exists():
        return
    async for toml_file in directory.rglob("*.toml"):
        try:
            card = await ModelCard.load_from_path(toml_file)
            if is_custom and get_custom_model_local_path(card.model_id) is not None:
                card = card.model_copy(update={"is_custom": True})
            if card.model_id not in _card_cache:
                _card_cache[card.model_id] = card
        except (ValidationError, TOMLKitError, ValueError):
            pass


async def _refresh_card_cache() -> None:
    for path in _BUILTIN_CARD_DIRS:
        await _load_cards_from_dir(path, is_custom=False)
    await _load_cards_from_dir(_custom_cards_dir, is_custom=True)


def _is_image_card(card: "ModelCard") -> bool:
    return any(t in (ModelTask.TextToImage, ModelTask.ImageToImage) for t in card.tasks)


def _allowed_inference_backend_values() -> set[str] | None:
    raw = (os.environ.get("CAI_ALLOWED_INFERENCE_BACKENDS") or "").strip()
    if not raw:
        return set(_SUPPORTED_BACKENDS)
    allowed = {
        backend.strip().lower()
        for backend in raw.split(",")
        if backend.strip()
    }
    return (allowed or set(_SUPPORTED_BACKENDS)) & _SUPPORTED_BACKENDS


def _card_matches_allowed_inference_backends(card: "ModelCard") -> bool:
    allowed_backends = _allowed_inference_backend_values()
    return str(card.inference_backend.value).strip().lower() in allowed_backends


def get_card(model_id: ModelId) -> "ModelCard | None":
    return _card_cache.get(model_id)


async def get_model_cards() -> list["ModelCard"]:
    if len(_card_cache) == 0:
        await _refresh_card_cache()
    cards = [
        card
        for card in _card_cache.values()
        if _card_matches_allowed_inference_backends(card)
    ]
    if CAI_ENABLE_IMAGE_MODELS:
        return cards
    return [card for card in cards if not _is_image_card(card)]


class ModelTask(str, Enum):
    TextGeneration = "TextGeneration"
    TextToImage = "TextToImage"
    ImageToImage = "ImageToImage"


class InferenceBackend(str, Enum):
    LlamaCpp = "llama_cpp"


class ComponentInfo(CamelCaseModel):
    component_name: str
    component_path: str
    storage_size: Memory
    n_layers: PositiveInt | None = None
    can_shard: bool
    safetensors_index_filename: str | None = None


class VisionCardConfig(CamelCaseModel):
    image_token_id: int
    model_type: str
    weights_repo: str = ""
    image_token: str | None = None
    processor_repo: str | None = None


class ModelCard(CamelCaseModel):
    model_id: ModelId
    storage_size: Memory
    n_layers: PositiveInt
    hidden_size: PositiveInt
    supports_tensor: bool
    num_key_value_heads: PositiveInt | None = None
    tasks: list[ModelTask]
    components: list[ComponentInfo] | None = None
    family: str = ""
    quantization: str = ""
    base_model: str = ""
    capabilities: list[str] = Field(default_factory=list)
    context_length: int = 0
    uses_cfg: bool = False
    trust_remote_code: bool = False
    is_custom: bool = False
    inference_backend: InferenceBackend = InferenceBackend.LlamaCpp
    preferred_filename: str | None = None
    vision: VisionCardConfig | None = None
    gguf_architecture: str = ""
    shard_compatibility: str = ""
    layer_range_supported: bool = False
    layer_range_probe_abi: str | None = None
    layer_range_probe_report: str | None = None
    layer_range_equivalence_probe_report: str | None = None
    state_format: str | None = None
    activation_state_format: str | None = None
    decode_state_format: str | None = None
    shard_compatibility_reason: str = ""

    @model_validator(mode="after")
    def _fill_gguf_shard_policy(self) -> "ModelCard":
        if self.inference_backend != InferenceBackend.LlamaCpp:
            return self
        policy = _gguf_shard_policy_fields(
            model_id=self.model_id,
            family=self.family,
            gguf_architecture=self.gguf_architecture,
            filename=self.preferred_filename,
        )
        for key, value in policy.items():
            object.__setattr__(self, key, value)
        return self

    @field_validator("tasks", mode="before")
    @classmethod
    def _validate_tasks(cls, value: list[str | ModelTask]) -> list[ModelTask]:
        return [item if isinstance(item, ModelTask) else ModelTask(item) for item in value]

    @field_validator("inference_backend", mode="before")
    @classmethod
    def _validate_inference_backend(
        cls, value: str | InferenceBackend
    ) -> InferenceBackend:
        return value if isinstance(value, InferenceBackend) else InferenceBackend(value)

    async def save(self, path: Path) -> None:
        async with await open_file(path, "w") as file:
            data = tomlkit.dumps(self.model_dump(exclude_none=True))
            await file.write(data)

    async def save_to_custom_dir(self) -> None:
        await aios.makedirs(str(_custom_cards_dir), exist_ok=True)
        await self.save(_custom_cards_dir / (self.model_id.normalize() + ".toml"))

    @staticmethod
    async def load_from_path(path: Path) -> "ModelCard":
        async with await open_file(path, "r") as file:
            return ModelCard.model_validate(tomlkit.loads(await file.read()))

    @staticmethod
    async def load(model_id: ModelId) -> "ModelCard":
        if model_id not in _card_cache:
            await _refresh_card_cache()
        if (card := _card_cache.get(model_id)) is not None:
            return card

        card = await ModelCard.fetch_from_hf(model_id)
        await card.save_to_custom_dir()
        _card_cache[model_id] = card
        return card

    @staticmethod
    async def fetch_from_hf(model_id: ModelId) -> "ModelCard":
        info = await asyncio.to_thread(model_info, model_id, files_metadata=True)
        if getattr(info, "gguf", None) or any(
            str(getattr(sibling, "rfilename", "")).lower().endswith(".gguf")
            for sibling in getattr(info, "siblings", [])
        ):
            return _build_hf_gguf_model_card(model_id, info)
        raise ValueError(
            f"Unsupported model format for {model_id}: CAI currently supports GGUF/llama.cpp models only"
        )

    @staticmethod
    async def load_from_local_directory(
        model_id: ModelId,
        local_path: str | FsPath,
    ) -> "ModelCard":
        resolved_path = _normalize_local_model_source_path(local_path)
        if not resolved_path.exists():
            raise ValueError(f"Local model path does not exist: {resolved_path}")
        if not resolved_path.is_dir() and not resolved_path.is_file():
            raise ValueError(
                f"Local model path is not a file or directory: {resolved_path}"
            )

        gguf_path = _find_local_gguf_path(resolved_path)
        if gguf_path is None:
            raise ValueError(
                "Only GGUF local model sources are supported by this CAI build"
            )

        filename = gguf_path.name
        family = _derive_family_from_filename(filename)
        return ModelCard(
            model_id=ModelId(model_id),
            storage_size=Memory.from_bytes(gguf_path.stat().st_size),
            n_layers=_infer_gguf_layer_count(
                model_id=model_id,
                architecture=family,
                filename=filename,
            ),
            hidden_size=1,
            supports_tensor=False,
            context_length=0,
            tasks=[ModelTask.TextGeneration],
            trust_remote_code=False,
            is_custom=True,
            inference_backend=InferenceBackend.LlamaCpp,
            family=family,
            quantization=_derive_quantization_from_filename(filename),
            base_model=gguf_path.stem,
            preferred_filename=filename,
        )


def add_to_card_cache(card: "ModelCard") -> None:
    _card_cache[card.model_id] = card


async def delete_custom_card(model_id: ModelId) -> bool:
    card_path = _custom_cards_dir / (ModelId(model_id).normalize() + ".toml")
    delete_custom_model_local_path(model_id)
    if await card_path.exists():
        await card_path.unlink()
        _card_cache.pop(model_id, None)
        return True
    return False


class ConfigData(BaseModel):
    model_config = {"extra": "ignore"}

    architectures: list[str] | None = None
    hidden_size: Annotated[int, Field(ge=0)] | None = None
    num_key_value_heads: PositiveInt | None = None
    layer_count: int = Field(
        validation_alias=AliasChoices(
            "num_hidden_layers",
            "num_layers",
            "n_layer",
            "n_layers",
            "num_decoder_layers",
            "decoder_layers",
        )
    )
    max_position_embeddings: int = 0
    vision: VisionCardConfig | None = None

    @property
    def supports_tensor(self) -> bool:
        return False

    @model_validator(mode="before")
    @classmethod
    def defer_to_text_config(cls, data: dict[str, Any], info: ValidationInfo):
        text_config = data.get("text_config")
        if text_config is not None:
            for field in [
                "architectures",
                "hidden_size",
                "num_key_value_heads",
                "max_position_embeddings",
                "num_hidden_layers",
                "num_layers",
                "n_layer",
                "n_layers",
                "num_decoder_layers",
                "decoder_layers",
            ]:
                if (value := text_config.get(field)) is not None:
                    data[field] = value

        vision_config = data.get("vision_config")
        image_token_id = data.get("image_token_id")
        if vision_config is not None and image_token_id is not None:
            model_type = str(data.get("model_type", vision_config.get("model_type", "")))
            assert info.context is not None
            data["vision"] = VisionCardConfig(
                image_token_id=int(image_token_id),
                model_type=model_type,
                weights_repo=str(info.context["model_id"]),
            )

        return data


async def fetch_config_data(model_id: ModelId) -> ConfigData:
    raise ValueError(
        f"Unsupported model format for {model_id}: CAI currently supports GGUF/llama.cpp models only"
    )


async def fetch_safetensors_size(model_id: ModelId) -> Memory:
    raise ValueError(
        f"Unsupported model format for {model_id}: CAI currently supports GGUF/llama.cpp models only"
    )
