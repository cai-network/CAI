# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from cai_compute_chain.cai_desktop_app import CaiDesktopConfig, wallet_home_path
from cai_compute_chain.model import WalletPolicy
from cai_compute_chain.wallet import data_root


def test_wallet_data_root_prefers_cai_wallet_home(monkeypatch, tmp_path) -> None:
    cai_wallet_home = tmp_path / "portable" / "data" / ".cai-local"
    monkeypatch.setenv("CAI_WALLET_HOME", str(cai_wallet_home))
    monkeypatch.setenv("CAI_REPO_ROOT", str(tmp_path / "portable"))

    assert data_root() == cai_wallet_home.resolve()


def test_wallet_data_root_falls_back_to_repo_policy_dir(monkeypatch, tmp_path) -> None:
    repo_root = tmp_path / "repo"
    monkeypatch.delenv("CAI_WALLET_HOME", raising=False)
    monkeypatch.setenv("CAI_REPO_ROOT", str(repo_root))

    assert data_root(WalletPolicy(wallet_data_dirname=".wallets")) == (
        repo_root / ".wallets"
    ).resolve()


def test_portable_wallet_home_uses_wallet_directory_not_runtime_home(tmp_path) -> None:
    portable = tmp_path / "CAI-portable"
    (portable / "_internal").mkdir(parents=True)
    (portable / "CAI.exe").write_text("", encoding="utf-8")

    assert wallet_home_path(CaiDesktopConfig(repo_root=portable)) == (
        portable / "data" / ".cai-local"
    )
