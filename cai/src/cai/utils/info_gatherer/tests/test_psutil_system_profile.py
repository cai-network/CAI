# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from cai.shared.types.profiling import SystemPerformanceProfile


def test_system_performance_profile_from_psutil_normalizes_cpu_usage(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "cai.shared.types.profiling.psutil.cpu_percent", lambda interval=None: 37.5
    )
    monkeypatch.setattr(
        "cai.shared.types.profiling.psutil.sensors_temperatures",
        lambda: {"coretemp": [type("Temp", (), {"current": 61.0})()]},
    )

    profile = SystemPerformanceProfile.from_psutil()

    assert profile.pcpu_usage == 0.375
    assert profile.temp == 61.0
    assert profile.gpu_usage == 0.0

