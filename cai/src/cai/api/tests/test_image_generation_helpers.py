# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from cai.api.image_generation_helpers import ensure_seed, format_to_content_type
from cai.api.types import AdvancedImageParams


def test_format_to_content_type_defaults_to_png() -> None:
    assert format_to_content_type(None) == "image/png"
    assert format_to_content_type("jpeg") == "image/jpeg"
    assert format_to_content_type("webp") == "image/webp"


def test_ensure_seed_creates_params_when_missing() -> None:
    params = ensure_seed(None)

    assert isinstance(params, AdvancedImageParams)
    assert params.seed is not None
    assert 0 <= params.seed <= 2**32 - 1


def test_ensure_seed_preserves_existing_seed_and_fields() -> None:
    params = AdvancedImageParams(seed=123, guidance=7.5)

    assert ensure_seed(params) is params


def test_ensure_seed_fills_missing_seed_without_mutating_original() -> None:
    params = AdvancedImageParams(guidance=7.5)
    updated = ensure_seed(params)

    assert params.seed is None
    assert updated.seed is not None
    assert updated.guidance == 7.5
