# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import asyncio

from cai.utils.info_gatherer import system_info


def test_get_total_vram_bytes_prefers_dxgi_on_windows(
    monkeypatch,
) -> None:
    monkeypatch.setattr(system_info.sys, "platform", "win32")
    monkeypatch.setattr(
        system_info,
        "_get_total_vram_bytes_dxgi",
        lambda: 8 * 1024**3,
    )
    monkeypatch.setenv("CAI_DISABLE_NVIDIA_SMI_VRAM_PROBE", "1")

    assert asyncio.run(system_info.get_total_vram_bytes()) == 8 * 1024**3


def test_get_total_vram_bytes_skips_nvidia_probe_when_disabled(
    monkeypatch,
) -> None:
    monkeypatch.setattr(system_info.sys, "platform", "win32")
    monkeypatch.setattr(system_info, "_get_total_vram_bytes_dxgi", lambda: None)
    monkeypatch.setenv("CAI_DISABLE_NVIDIA_SMI_VRAM_PROBE", "1")
    monkeypatch.setattr(
        system_info.shutil,
        "which",
        lambda _name: (_ for _ in ()).throw(AssertionError("nvidia-smi probe should stay disabled")),
    )

    assert asyncio.run(system_info.get_total_vram_bytes()) is None

