# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import sys

import pytest

from cai.shared.types.thunderbolt import (
    ThunderboltConnectivity,
)
from cai.utils.info_gatherer.info_gatherer import (
    _gather_iface_map,  # pyright: ignore[reportPrivateUsage]
)


@pytest.mark.anyio
@pytest.mark.skipif(
    sys.platform != "darwin", reason="Thunderbolt info can only be gathered on macos"
)
async def test_tb_parsing():
    data = await ThunderboltConnectivity.gather()
    ifaces = await _gather_iface_map()
    assert ifaces
    assert data
    for datum in data:
        datum.ident(ifaces)
        datum.conn()

