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

from cai_compute_chain.chain import (
    append_chain_block,
    chain_balance_atomic,
    compute_reserve_chain_address,
    ensure_chain_genesis,
    make_chain_transaction,
)
from cai_compute_chain.cli import _handle_job_quote, _handle_wallet_send, handle_wallet_status
from cai_compute_chain.model import MoneyPolicy, PaymentPreference
from cai_compute_chain.wallet import (
    coins_to_atomic,
    create_wallet,
    credit_wallet,
    load_or_create_ledger,
    save_ledger,
    unlock_wallet,
)


class CliWalletTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo_patch = patch(
            "cai_compute_chain.wallet.repo_root",
            return_value=Path(self.tempdir.name),
        )
        self.repo_patch.start()
        self._chain_credit_counter = 0

    def tearDown(self) -> None:
        self.repo_patch.stop()
        self.tempdir.cleanup()

    def _credit_wallet_from_reserve(self, wallet, amount_atomic: int) -> None:
        money_policy = MoneyPolicy()
        ensure_chain_genesis(money_policy=money_policy)
        self._chain_credit_counter += 1
        credit_id = f"test-cli-wallet-credit-{wallet.wallet_id}-{self._chain_credit_counter}"
        append_chain_block(
            [
                make_chain_transaction(
                    tx_type="test_reserve_debit",
                    address=compute_reserve_chain_address(money_policy),
                    delta_atomic=-amount_atomic,
                    wallet_id=f"system-compute-reserve-{money_policy.chain_network.value}",
                    counterparty_address=wallet.address,
                    nonce=f"{credit_id}:reserve-debit",
                ),
                make_chain_transaction(
                    tx_type="test_wallet_credit",
                    address=wallet.address,
                    delta_atomic=amount_atomic,
                    wallet_id=wallet.wallet_id,
                    counterparty_address=compute_reserve_chain_address(money_policy),
                    nonce=f"{credit_id}:wallet-credit",
                ),
            ],
            validator_id="test-cli",
        )

    def test_wallet_status_shows_chain_balance_not_local_cache(self) -> None:
        wallet = create_wallet("main", "pass", select=True)
        credit_wallet(wallet.wallet_id, coins_to_atomic("5.00000000"))

        output = handle_wallet_status()

        self.assertIn("- balance=0.00000000", output)
        self.assertIn("- balance_source=chain", output)
        self.assertIn("- local_cached_balance=5.00000000", output)
        self.assertEqual(chain_balance_atomic(wallet.address), 0)

    def test_job_quote_wallet_only_ignores_local_only_balance(self) -> None:
        wallet = create_wallet("main", "pass", select=True)
        credit_wallet(wallet.wallet_id, coins_to_atomic("5.00000000"))
        unlock_wallet("pass", selector=wallet.wallet_id)

        output = _handle_job_quote(
            amount="1.00000000",
            payment=PaymentPreference.WALLET_ONLY.value,
            prompt=None,
            model=None,
            cai_url="http://127.0.0.1:52415",
        )

        self.assertIn("- can_fund=False", output)
        self.assertIn("Wallet balance is insufficient", output)
        self.assertIn("- wallet_before=0.00000000", output)

    def test_wallet_send_ignores_local_only_balance(self) -> None:
        sender = create_wallet("sender", "pass", select=True)
        recipient = create_wallet("recipient", "pass")
        credit_wallet(sender.wallet_id, coins_to_atomic("5.00000000"))
        unlock_wallet("pass", selector=sender.wallet_id)

        with self.assertRaisesRegex(ValueError, "Wallet chain balance is insufficient"):
            _handle_wallet_send(
                recipient_address=recipient.address,
                amount="1.00000000",
            )

        self.assertEqual(chain_balance_atomic(sender.address), 0)
        self.assertEqual(chain_balance_atomic(recipient.address), 0)

    def test_job_quote_reserve_uses_chain_reserve_over_ledger_cache(self) -> None:
        money_policy = MoneyPolicy()
        wallet = create_wallet("main", "pass", select=True)
        self._credit_wallet_from_reserve(wallet, coins_to_atomic("1.00000000"))
        unlock_wallet("pass", selector=wallet.wallet_id)
        ledger = load_or_create_ledger(money_policy)
        ledger.compute_reserve_balance_atomic = 0
        save_ledger(ledger)

        output = _handle_job_quote(
            amount="0.10000000",
            payment=PaymentPreference.RESERVE_ONLY.value,
            prompt=None,
            model=None,
            cai_url="http://127.0.0.1:52415",
        )

        self.assertIn("- can_fund=True", output)
        self.assertIn("- funding_source=reserve", output)
        self.assertNotIn("- reserve_before=0.00000000", output)


if __name__ == "__main__":
    unittest.main()
