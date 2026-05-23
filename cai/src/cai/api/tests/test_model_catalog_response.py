# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from cai_compute_chain.model import CuratedNetworkModel

from cai.api.model_catalog_response import build_model_list_response
from cai.shared.models.model_cards import ModelCard, ModelId, ModelTask
from cai.shared.types.common import NodeId
from cai.shared.types.memory import Memory
from cai.shared.types.worker.downloads import DownloadCompleted
from cai.shared.types.worker.shards import TensorShardMetadata


def _card(model_id: str, *, is_custom: bool = False) -> ModelCard:
    return ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_mb(512),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        is_custom=is_custom,
    )


def _curated(
    model_id: str,
    *,
    execution_model_id: str | None = None,
    runtime_model_ids: tuple[str, ...] = (),
    display_name: str = "Curated model",
    private_network: bool = True,
) -> CuratedNetworkModel:
    return CuratedNetworkModel(
        model_id=model_id,
        execution_model_id=execution_model_id or model_id,
        display_name=display_name,
        source_repo_id=model_id,
        preferred_filename="model.gguf",
        runtime_model_ids=runtime_model_ids,
        private_network=private_network,
    )


def _completed_download(card: ModelCard) -> DownloadCompleted:
    return DownloadCompleted(
        node_id=NodeId("node-1"),
        shard_metadata=TensorShardMetadata(
            model_card=card,
            device_rank=0,
            world_size=1,
            start_layer=0,
            end_layer=1,
            n_layers=1,
        ),
        total=card.storage_size,
    )


def test_build_model_list_response_prefers_curated_alias_display_name() -> None:
    card = _card("Qwen/Qwen3-0.6B-GGUF")
    response = build_model_list_response(
        all_cards=[card],
        curated_models=[
            _curated(
                "cai/qwen-small",
                execution_model_id=str(card.model_id),
                display_name="Qwen Small",
            )
        ],
        downloads_by_node={},
    )

    assert [model.id for model in response.data] == ["cai/qwen-small"]
    assert response.data[0].name == "Qwen Small"
    assert response.data[0].hugging_face_id == "cai/qwen-small"


def test_build_model_list_response_includes_custom_llama_cards() -> None:
    response = build_model_list_response(
        all_cards=[_card("local/custom", is_custom=True)],
        curated_models=[],
        downloads_by_node={},
    )

    assert [model.id for model in response.data] == ["local/custom"]
    assert response.data[0].is_custom is True


def test_build_model_list_response_downloaded_uses_curated_runtime_aliases() -> None:
    execution_card = _card("Qwen/Qwen3-0.6B-GGUF")
    response = build_model_list_response(
        all_cards=[execution_card],
        curated_models=[
            _curated(
                "cai/qwen-small",
                execution_model_id=str(execution_card.model_id),
                runtime_model_ids=("cai/qwen-small-runtime",),
            )
        ],
        downloads_by_node={"node-1": [_completed_download(execution_card)]},
        status="downloaded",
    )

    assert [model.id for model in response.data] == ["cai/qwen-small"]
