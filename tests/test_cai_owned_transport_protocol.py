# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import unittest

from cai_compute_chain import cai_owned_transport_peer_urls as peer_urls
from cai_compute_chain import cai_owned_transport_protocol as protocol
from cai_compute_chain import cai_owned_transport_storage as storage
from cai_compute_chain import decentralized_compute


class CaiOwnedTransportProtocolTests(unittest.TestCase):
    def test_protocol_constants_are_shared_by_compatibility_modules(self) -> None:
        self.assertEqual(
            decentralized_compute.CAI_OWNED_TRANSPORT_PROTOCOL,
            protocol.CAI_OWNED_TRANSPORT_PROTOCOL,
        )
        self.assertEqual(
            storage.CAI_OWNED_TRANSPORT_PROTOCOL,
            protocol.CAI_OWNED_TRANSPORT_PROTOCOL,
        )
        self.assertEqual(
            peer_urls.CAI_OWNED_TRANSPORT_OVERLAY_URL_PREFIX,
            protocol.CAI_OWNED_TRANSPORT_OVERLAY_URL_PREFIX,
        )
        self.assertEqual(
            decentralized_compute.EXECUTION_MODE_CAI_OWNED_TRANSPORT_REQUIRED,
            protocol.EXECUTION_MODE_CAI_OWNED_TRANSPORT_REQUIRED,
        )


if __name__ == "__main__":
    unittest.main()
