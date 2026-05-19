# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cai_compute_chain.chain import ensure_chain_genesis, chain_summary
from cai_compute_chain.cli import handle_ledger
from cai_compute_chain.model import MoneyPolicy, WalletPolicy
from cai_compute_chain.wallet import (
    coins_to_atomic,
    load_or_create_ledger,
    save_ledger,
)


class CliLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo_patch = patch(
            "cai_compute_chain.wallet.repo_root",
            return_value=Path(self.tempdir.name),
        )
        self.repo_patch.start()

    def tearDown(self) -> None:
        self.repo_patch.stop()
        self.tempdir.cleanup()

    def test_handle_ledger_initializes_chain_and_ignores_ledger_cache(self) -> None:
        money_policy = MoneyPolicy()
        ledger = load_or_create_ledger(money_policy)
        ledger.compute_reserve_balance_atomic = coins_to_atomic("7.00000000")
        save_ledger(ledger)

        output = handle_ledger()

        chain_reserve = chain_summary()["computeReserveBalanceCoins"]
        self.assertIn("- balance_source=chain", output)
        self.assertIn("- chain_block_count=1", output)
        self.assertIn(f"- compute_reserve_balance={chain_reserve}", output)
        self.assertNotIn("- compute_reserve_balance=7.00000000", output)

    def test_handle_ledger_uses_chain_balances_after_chain_init(self) -> None:
        money_policy = MoneyPolicy()
        wallet_policy = WalletPolicy()
        ensure_chain_genesis(policy=wallet_policy, money_policy=money_policy)
        chain_reserve = chain_summary(wallet_policy)["computeReserveBalanceCoins"]
        ledger = load_or_create_ledger(money_policy, wallet_policy)
        ledger.compute_reserve_balance_atomic = coins_to_atomic("7.00000000")
        save_ledger(ledger, wallet_policy)

        output = handle_ledger()

        self.assertIn("- balance_source=chain", output)
        self.assertIn("- chain_block_count=1", output)
        self.assertIn(f"- compute_reserve_balance={chain_reserve}", output)
        self.assertNotIn("- compute_reserve_balance=7.00000000", output)


if __name__ == "__main__":
    unittest.main()
