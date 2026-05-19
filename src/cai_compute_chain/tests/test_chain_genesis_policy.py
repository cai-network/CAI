# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT

import pytest

from cai_compute_chain.chain import (
    ChainBlock,
    compute_block_hash,
    compute_chain_state_root,
    compute_transaction_root,
    ensure_chain_genesis,
    list_chain_blocks,
    make_chain_transaction,
    make_genesis_block,
    save_chain_blocks,
    validate_chain_blocks,
)
from cai_compute_chain.model import MoneyPolicy, WalletPolicy
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


def test_ensure_chain_genesis_rejects_active_mismatched_chain(tmp_path) -> None:
    policy = WalletPolicy(wallet_data_dirname=str(tmp_path))
    current_genesis = make_genesis_block(MoneyPolicy())
    stale_genesis = make_genesis_block(
        MoneyPolicy(developer_treasury_address="1" * 64)
    )
    tx = make_chain_transaction(
        tx_type="test_activity",
        address="system:test",
        delta_atomic=0,
        nonce="test-active-stale-chain",
        chain_id="mainnet",
    )
    activity_block = ChainBlock(
        block_id="test-stale-activity",
        height=1,
        created_at="2026-01-01T00:01:00+00:00",
        previous_hash=stale_genesis.block_hash,
        validator_id="test-validator",
        transactions=[tx],
        block_hash="",
        chain_id="mainnet",
        tx_root=compute_transaction_root([tx]),
    )
    activity_block.state_root = compute_chain_state_root(
        [stale_genesis, activity_block]
    )
    activity_block.block_hash = compute_block_hash(activity_block)
    save_chain_blocks([stale_genesis, activity_block], policy)

    with pytest.raises(ValueError, match="Local chain genesis_hash"):
        ensure_chain_genesis(policy=policy)

    assert list_chain_blocks(policy)[0].block_hash == stale_genesis.block_hash
    assert list_chain_blocks(policy)[0].block_hash != current_genesis.block_hash
