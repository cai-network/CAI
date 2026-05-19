# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT

import pytest

from cai_compute_chain.chain import make_genesis_block, validate_chain_blocks
from cai_compute_chain.model import MoneyPolicy
from cai_compute_chain.peer_payload import (
    add_peer_payload_metadata,
    validate_peer_payload_network,
)


def test_chain_validation_rejects_genesis_that_does_not_match_policy() -> None:
    current_genesis = make_genesis_block(MoneyPolicy())
    stale_genesis = make_genesis_block(
        MoneyPolicy(developer_treasury_address="1" * 64)
    )

    assert stale_genesis.block_hash != current_genesis.block_hash

    errors = validate_chain_blocks([stale_genesis])

    assert any("genesis_hash" in error for error in errors)


def test_peer_metadata_preserves_chain_genesis_instead_of_masking_it() -> None:
    stale_genesis = make_genesis_block(
        MoneyPolicy(developer_treasury_address="1" * 64)
    )
    payload = {
        "chain": {
            "network": "mainnet",
            "chain_id": "mainnet",
            "genesis_hash": stale_genesis.block_hash,
            "blocks": [{"block_hash": stale_genesis.block_hash}],
        }
    }

    decorated = add_peer_payload_metadata(payload)

    assert decorated["genesis_hash"] == stale_genesis.block_hash
    with pytest.raises(ValueError, match="genesis_hash"):
        validate_peer_payload_network(decorated, payload_name="chain")


def test_peer_network_validation_rejects_conflicting_genesis_metadata() -> None:
    current_genesis = make_genesis_block(MoneyPolicy())
    stale_genesis = make_genesis_block(
        MoneyPolicy(developer_treasury_address="1" * 64)
    )
    payload = {
        "network": "mainnet",
        "chain_id": "mainnet",
        "genesis_hash": current_genesis.block_hash,
        "chain": {
            "network": "mainnet",
            "chain_id": "mainnet",
            "genesis_hash": stale_genesis.block_hash,
            "blocks": [{"block_hash": stale_genesis.block_hash}],
        },
    }

    with pytest.raises(ValueError, match="conflicting genesis_hash"):
        validate_peer_payload_network(payload, payload_name="chain")
