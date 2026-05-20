# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import unittest

from cai_compute_chain import cai_owned_transport_peer_urls as peer_urls
from cai_compute_chain import cai_owned_transport_common as common
from cai_compute_chain import cai_owned_transport_protocol as protocol
from cai_compute_chain import cai_owned_transport_storage as storage
from cai_compute_chain import cai_owned_transport_auth as transport_auth
from cai_compute_chain import decentralized_compute
from cai_compute_chain.model import ChainNetwork, WalletPolicy


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
        self.assertIs(
            decentralized_compute.sign_cai_owned_transport_session_offer,
            transport_auth.sign_cai_owned_transport_session_offer,
        )
        self.assertIs(
            decentralized_compute.validate_cai_owned_transport_payload_signature,
            transport_auth.validate_cai_owned_transport_payload_signature,
        )
        self.assertIs(
            decentralized_compute.cai_owned_transport_auth_headers,
            transport_auth.cai_owned_transport_auth_headers,
        )
        self.assertIs(
            decentralized_compute.validate_cai_owned_transport_request_auth,
            transport_auth.validate_cai_owned_transport_request_auth,
        )
        self.assertIs(
            decentralized_compute.validate_cai_owned_transport_local_runtime_auth,
            transport_auth.validate_cai_owned_transport_local_runtime_auth,
        )

    def test_common_helpers_match_legacy_transport_expectations(self) -> None:
        self.assertEqual(
            common.clean_node_ids([" node-a ", "", "node-a", "node-b"]),
            ["node-a", "node-b"],
        )
        self.assertEqual(
            common.cai_owned_transport_chain_id(chain_id=" MAINNET "),
            "mainnet",
        )
        self.assertEqual(
            common.cai_owned_transport_chain_id(
                WalletPolicy(chain_network=ChainNetwork.TESTNET)
            ),
            "testnet",
        )

        parsed = common.parse_cai_owned_transport_datetime("2026-05-20T12:00:00")
        self.assertIsNotNone(parsed)
        self.assertIsNotNone(parsed.tzinfo)

        self.assertTrue(
            common.is_safe_transport_file_id("caibatch_abc-123", prefix="caibatch_")
        )
        self.assertFalse(
            common.is_safe_transport_file_id("../caibatch_abc", prefix="caibatch_")
        )
        self.assertEqual(
            common.require_safe_transport_file_id(
                " caistage_stage-1 ",
                prefix="caistage_",
            ),
            "caistage_stage-1",
        )


if __name__ == "__main__":
    unittest.main()
