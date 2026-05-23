# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Iterable, Mapping

from cai_compute_chain.model import CuratedNetworkModel

from cai.api.types import ModelList, ModelListModel
from cai.shared.models.model_cards import InferenceBackend, ModelCard, ModelId
from cai.shared.types.worker.downloads import DownloadCompleted


def _downloaded_model_ids(downloads_by_node: Mapping[object, Iterable[object]]) -> set[str]:
    downloaded_model_ids: set[str] = set()
    for node_downloads in downloads_by_node.values():
        for download in node_downloads:
            if isinstance(download, DownloadCompleted):
                downloaded_model_ids.add(
                    str(download.shard_metadata.model_card.model_id)
                )
    return downloaded_model_ids


def _curated_aliases_by_model_id(
    curated_models: Iterable[CuratedNetworkModel],
) -> dict[str, set[str]]:
    return {
        model.model_id: {
            model.model_id,
            model.execution_model_id,
            *model.runtime_model_ids,
        }
        for model in curated_models
    }


def _visible_curated_models(
    curated_models: Iterable[CuratedNetworkModel],
) -> list[CuratedNetworkModel]:
    curated_models = tuple(curated_models)
    private_execution_ids = {
        model.execution_model_id for model in curated_models if model.private_network
    }
    return [
        model
        for model in curated_models
        if model.private_network or model.model_id not in private_execution_ids
    ]


def _selected_model_cards(
    all_cards: Iterable[ModelCard],
    curated_models: Iterable[CuratedNetworkModel],
) -> list[tuple[ModelCard, str, str | None]]:
    llama_cards = [
        card for card in all_cards if card.inference_backend == InferenceBackend.LlamaCpp
    ]
    cards_by_id = {str(card.model_id): card for card in llama_cards}
    selected_cards: list[tuple[ModelCard, str, str | None]] = []
    selected_ids: set[str] = set()

    for model in _visible_curated_models(curated_models):
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

    return selected_cards


def _model_list_item(
    card: ModelCard,
    *,
    model_id: str,
    display_name: str | None,
) -> ModelListModel:
    return ModelListModel(
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
        layer_range_equivalence_probe_report=card.layer_range_equivalence_probe_report,
        state_format=card.state_format,
        activation_state_format=card.activation_state_format,
        decode_state_format=card.decode_state_format,
        shard_compatibility_reason=card.shard_compatibility_reason,
    )


def build_model_list_response(
    *,
    all_cards: Iterable[ModelCard],
    curated_models: Iterable[CuratedNetworkModel],
    downloads_by_node: Mapping[object, Iterable[object]],
    status: str | None = None,
) -> ModelList:
    curated_models = tuple(curated_models)
    selected_cards = _selected_model_cards(all_cards, curated_models)

    if status == "downloaded":
        downloaded_model_ids = _downloaded_model_ids(downloads_by_node)
        aliases_by_model_id = _curated_aliases_by_model_id(curated_models)
        selected_cards = [
            item
            for item in selected_cards
            if aliases_by_model_id.get(item[1], {item[1]}) & downloaded_model_ids
        ]

    return ModelList(
        data=[
            _model_list_item(card, model_id=model_id, display_name=display_name)
            for card, model_id, display_name in selected_cards
        ]
    )
