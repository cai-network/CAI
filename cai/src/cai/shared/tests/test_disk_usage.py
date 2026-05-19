# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import shutil

from cai.shared.types.profiling import DiskUsage


def test_disk_usage_from_path_uses_existing_parent(tmp_path) -> None:
    missing_models_dir = tmp_path / "nested" / "models"

    usage = DiskUsage.from_path(missing_models_dir)
    expected = shutil.disk_usage(tmp_path)

    assert usage.total.in_bytes == expected.total
    assert usage.available.in_bytes == expected.free

