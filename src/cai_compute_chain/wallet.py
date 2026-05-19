# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .local_json_store import (
    atomic_write_json_array_file,
    atomic_write_json_object_file,
    read_json_array_file,
    read_json_object_file,
)
from .model import MoneyPolicy, WalletPolicy
from .seed_phrase import (
    DEFAULT_SEED_WORD_COUNT,
    derive_seed_wallet_id,
    generate_seed_phrase,
    seed_fingerprint,
    validate_seed_phrase,
)
from .wallet_signing import (
    ADDRESS_SCHEME_FIXED_WALLET_ID_SHA256,
    ADDRESS_SCHEME_ED25519,
    DEFAULT_WALLET_HYBRID_ADDRESS_SCHEME,
    HYBRID_ADDRESS_SCHEMES,
    SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65,
    SIGNING_SCHEME_ML_DSA_65,
    SIGNING_SCHEME_ED25519,
    decrypt_signing_seed,
    derive_pq_signing_seed_from_seed_phrase,
    derive_signing_seed_from_seed_phrase,
    encode_bytes,
    encrypt_signing_seed,
    generate_signing_seed,
    generate_mldsa65_keypair_b64,
    hybrid_address_from_public_keys_b64,
    mldsa65_keypair_b64_from_seed,
    public_key_b64_from_seed,
    sign_payload_mldsa65_b64,
    sign_payload_b64,
)


def repo_root() -> Path:
    configured = str(os.getenv("CAI_REPO_ROOT") or os.getenv("CAI_RUNTIME_REPO") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def data_root(policy: WalletPolicy | None = None) -> Path:
    active_policy = policy or WalletPolicy()
    configured_home = str(os.getenv("CAI_WALLET_HOME") or "").strip()
    root = (
        Path(configured_home).expanduser().resolve()
        if configured_home
        else repo_root() / active_policy.wallet_data_dirname
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass
class WalletRecord:
    wallet_id: str
    name: str
    address: str
    created_at: str
    password_salt_b64: str
    password_hash_b64: str
    spendable_balance_atomic: int = 0
    validator_reserved_atomic: int = 0
    seed_fingerprint: str | None = None
    seed_word_count: int | None = None
    signing_scheme: str | None = None
    address_scheme: str | None = None
    public_key_b64: str | None = None
    encrypted_private_key_b64: str | None = None
    private_key_salt_b64: str | None = None
    private_key_nonce_b64: str | None = None
    pq_signing_scheme: str | None = None
    pq_public_key_b64: str | None = None
    encrypted_pq_private_key_b64: str | None = None
    pq_private_key_salt_b64: str | None = None
    pq_private_key_nonce_b64: str | None = None


@dataclass
class WalletSession:
    active_wallet_id: str | None = None
    unlocked_wallet_id: str | None = None
    unlocked_at: str | None = None


@dataclass(frozen=True)
class SeedWalletIdentity:
    wallet_id: str
    address: str
    address_scheme: str
    public_key_b64: str
    pq_public_key_b64: str


@dataclass
class LedgerState:
    compute_reserve_balance_atomic: int
    project_treasury_balance_atomic: int
    developer_treasury_wallet_id: str | None = None
    developer_treasury_address: str | None = None
    developer_treasury_allocated_atomic: int = 0
    developer_treasury_provisioned_locally: bool = False
    developer_treasury_seed_file: str | None = None
    developer_treasury_password_file: str | None = None
    ai_development_wallet_id: str | None = None
    ai_development_address: str | None = None
    ai_development_provisioned_locally: bool = False
    ai_development_seed_file: str | None = None
    ai_development_password_file: str | None = None
    ai_development_fee_pool_atomic: int = 0
    validator_fee_pool_atomic: int = 0
    validator_slashed_atomic: int = 0
    tx_fee_pool_atomic: int = 0
    worker_distributed_atomic: int = 0
    settlements_applied: int = 0


@dataclass
class JournalEntry:
    entry_id: str
    event_type: str
    created_at: str
    wallet_id: str | None = None
    counterparty_address: str | None = None
    amount_atomic: int | None = None
    funding_source: str | None = None
    compute_cost_atomic: int | None = None
    tx_fee_atomic: int | None = None
    settlement_fee_atomic: int | None = None
    worker_reward_atomic: int | None = None
    note: str | None = None


def wallets_file_path(policy: WalletPolicy | None = None) -> Path:
    active_policy = policy or WalletPolicy()
    return data_root(active_policy) / active_policy.wallet_file_name


def session_file_path(policy: WalletPolicy | None = None) -> Path:
    active_policy = policy or WalletPolicy()
    return data_root(active_policy) / active_policy.session_file_name


def ledger_file_path(policy: WalletPolicy | None = None) -> Path:
    active_policy = policy or WalletPolicy()
    return data_root(active_policy) / active_policy.ledger_file_name


def journal_file_path(policy: WalletPolicy | None = None) -> Path:
    active_policy = policy or WalletPolicy()
    return data_root(active_policy) / active_policy.journal_file_name


def secrets_root(policy: WalletPolicy | None = None) -> Path:
    active_policy = policy or WalletPolicy()
    root = data_root(active_policy) / active_policy.secret_dir_name
    root.mkdir(parents=True, exist_ok=True)
    return root


def developer_treasury_seed_file_path(policy: WalletPolicy | None = None) -> Path:
    active_policy = policy or WalletPolicy()
    return secrets_root(active_policy) / active_policy.developer_treasury_seed_file_name


def developer_treasury_password_file_path(policy: WalletPolicy | None = None) -> Path:
    active_policy = policy or WalletPolicy()
    return secrets_root(active_policy) / active_policy.developer_treasury_password_file_name


def ai_development_seed_file_path(policy: WalletPolicy | None = None) -> Path:
    active_policy = policy or WalletPolicy()
    return secrets_root(active_policy) / active_policy.ai_development_seed_file_name


def ai_development_password_file_path(policy: WalletPolicy | None = None) -> Path:
    active_policy = policy or WalletPolicy()
    return secrets_root(active_policy) / active_policy.ai_development_password_file_name


def unlocked_signing_key_file_path(policy: WalletPolicy | None = None) -> Path:
    return secrets_root(policy) / "unlocked-wallet-signing-key.json"


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    if isinstance(default, list):
        return read_json_array_file(path, heal_corrupt=True)
    if isinstance(default, dict):
        return read_json_object_file(path, heal_corrupt=True)
    if default is None:
        raw = read_json_object_file(path, heal_corrupt=True)
        return raw or None
    return default


def _write_json(path: Path, payload: Any) -> None:
    if isinstance(payload, list):
        atomic_write_json_array_file(path, payload)
        return
    if isinstance(payload, dict):
        atomic_write_json_object_file(path, payload)
        return
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_secret_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def normalize_address(address: str) -> str:
    normalized = address.strip().lower()
    if normalized.startswith("cai_"):
        return normalized[4:]
    return normalized


def derive_seed_wallet_identity(seed_phrase: str) -> SeedWalletIdentity:
    normalized_seed = validate_seed_phrase(seed_phrase)
    wallet_id = derive_seed_wallet_id(normalized_seed)
    signing_seed = derive_signing_seed_from_seed_phrase(normalized_seed)
    public_key_b64 = public_key_b64_from_seed(signing_seed)
    pq_signing_seed = derive_pq_signing_seed_from_seed_phrase(normalized_seed)
    pq_public_key_b64, _pq_private_key_b64 = mldsa65_keypair_b64_from_seed(
        pq_signing_seed
    )
    address = hybrid_address_from_public_keys_b64(
        ed25519_public_key_b64=public_key_b64,
        pq_public_key_b64=pq_public_key_b64,
        address_scheme=DEFAULT_WALLET_HYBRID_ADDRESS_SCHEME,
    )
    return SeedWalletIdentity(
        wallet_id=wallet_id,
        address=address,
        address_scheme=DEFAULT_WALLET_HYBRID_ADDRESS_SCHEME,
        public_key_b64=public_key_b64,
        pq_public_key_b64=pq_public_key_b64,
    )


def _new_wallet_signing_material(
    *,
    wallet_id: str,
    password: str,
    wallet_policy: WalletPolicy,
    signing_seed: bytes,
    pq_signing_seed: bytes | None = None,
    forced_address: str | None = None,
) -> dict[str, str]:
    public_key_b64 = public_key_b64_from_seed(signing_seed)
    if pq_signing_seed is None:
        pq_public_key_b64, pq_private_key_b64 = generate_mldsa65_keypair_b64()
    else:
        pq_public_key_b64, pq_private_key_b64 = mldsa65_keypair_b64_from_seed(
            pq_signing_seed
        )
    fixed_address = normalize_address(forced_address) if forced_address else None
    address = fixed_address or hybrid_address_from_public_keys_b64(
        ed25519_public_key_b64=public_key_b64,
        pq_public_key_b64=pq_public_key_b64,
        address_scheme=DEFAULT_WALLET_HYBRID_ADDRESS_SCHEME,
    )
    address_scheme = (
        ADDRESS_SCHEME_FIXED_WALLET_ID_SHA256
        if fixed_address
        else DEFAULT_WALLET_HYBRID_ADDRESS_SCHEME
    )
    encrypted_private_key_b64, private_key_salt_b64, private_key_nonce_b64 = (
        encrypt_signing_seed(
            signing_seed=signing_seed,
            password=password,
            wallet_id=wallet_id,
            address=address,
            kdf_rounds=wallet_policy.password_kdf_rounds,
        )
    )
    (
        encrypted_pq_private_key_b64,
        pq_private_key_salt_b64,
        pq_private_key_nonce_b64,
    ) = encrypt_signing_seed(
        signing_seed=_decode_bytes(pq_private_key_b64),
        password=password,
        wallet_id=wallet_id,
        address=address,
        kdf_rounds=wallet_policy.password_kdf_rounds,
    )
    return {
        "address": address,
        "address_scheme": address_scheme,
        "signing_scheme": SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65,
        "public_key_b64": public_key_b64,
        "encrypted_private_key_b64": encrypted_private_key_b64,
        "private_key_salt_b64": private_key_salt_b64,
        "private_key_nonce_b64": private_key_nonce_b64,
        "pq_signing_scheme": SIGNING_SCHEME_ML_DSA_65,
        "pq_public_key_b64": pq_public_key_b64,
        "encrypted_pq_private_key_b64": encrypted_pq_private_key_b64,
        "pq_private_key_salt_b64": pq_private_key_salt_b64,
        "pq_private_key_nonce_b64": pq_private_key_nonce_b64,
    }


def _hash_password(password: str, *, salt: bytes, rounds: int) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        rounds,
    )


def _encode_bytes(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def _decode_bytes(payload: str) -> bytes:
    return base64.b64decode(payload.encode("ascii"))


def list_wallets(policy: WalletPolicy | None = None) -> list[WalletRecord]:
    raw = _read_json(wallets_file_path(policy), default=[])
    wallets = [WalletRecord(**item) for item in raw]
    changed = False
    for wallet in wallets:
        normalized_address = normalize_address(wallet.address)
        if wallet.address != normalized_address:
            wallet.address = normalized_address
            changed = True
    if changed:
        save_wallets(wallets, policy)
    return wallets


def save_wallets(wallets: list[WalletRecord], policy: WalletPolicy | None = None) -> None:
    _write_json(wallets_file_path(policy), [asdict(wallet) for wallet in wallets])


def load_session(policy: WalletPolicy | None = None) -> WalletSession:
    raw = _read_json(session_file_path(policy), default={})
    return WalletSession(**raw)


def save_session(session: WalletSession, policy: WalletPolicy | None = None) -> None:
    _write_json(session_file_path(policy), asdict(session))


def load_or_create_ledger(
    money_policy: MoneyPolicy | None = None,
    wallet_policy: WalletPolicy | None = None,
) -> LedgerState:
    active_money_policy = money_policy or MoneyPolicy()
    raw = _read_json(ledger_file_path(wallet_policy), default=None)
    if raw is not None:
        ledger = LedgerState(**raw)
        changed = False
        expected_treasury_atomic = coins_to_atomic(
            str(active_money_policy.developer_treasury_coins), active_money_policy
        )
        if not ledger.developer_treasury_wallet_id:
            ledger.developer_treasury_wallet_id = active_money_policy.developer_treasury_wallet_id
            changed = True
        if not ledger.developer_treasury_address:
            ledger.developer_treasury_address = active_money_policy.developer_treasury_address
            changed = True
        if ledger.developer_treasury_allocated_atomic <= 0:
            if ledger.project_treasury_balance_atomic == expected_treasury_atomic:
                ledger.project_treasury_balance_atomic = 0
            ledger.developer_treasury_allocated_atomic = expected_treasury_atomic
            changed = True
        if ledger.developer_treasury_seed_file is None:
            ledger.developer_treasury_seed_file = str(
                developer_treasury_seed_file_path(wallet_policy)
            )
            changed = True
        if ledger.developer_treasury_password_file is None:
            ledger.developer_treasury_password_file = str(
                developer_treasury_password_file_path(wallet_policy)
            )
            changed = True
        if not ledger.ai_development_wallet_id:
            ledger.ai_development_wallet_id = active_money_policy.ai_development_wallet_id
            changed = True
        if not ledger.ai_development_address:
            ledger.ai_development_address = active_money_policy.ai_development_address
            changed = True
        if ledger.ai_development_seed_file is None:
            ledger.ai_development_seed_file = str(
                ai_development_seed_file_path(wallet_policy)
            )
            changed = True
        if ledger.ai_development_password_file is None:
            ledger.ai_development_password_file = str(
                ai_development_password_file_path(wallet_policy)
            )
            changed = True
        if changed:
            save_ledger(ledger, wallet_policy)
        return ledger

    ledger = LedgerState(
        compute_reserve_balance_atomic=coins_to_atomic(
            str(active_money_policy.compute_reserve_coins), active_money_policy
        ),
        project_treasury_balance_atomic=0,
        developer_treasury_wallet_id=active_money_policy.developer_treasury_wallet_id,
        developer_treasury_address=active_money_policy.developer_treasury_address,
        developer_treasury_allocated_atomic=coins_to_atomic(
            str(active_money_policy.developer_treasury_coins), active_money_policy
        ),
        developer_treasury_provisioned_locally=False,
        developer_treasury_seed_file=str(
            developer_treasury_seed_file_path(wallet_policy)
        ),
        developer_treasury_password_file=str(
            developer_treasury_password_file_path(wallet_policy)
        ),
        ai_development_wallet_id=active_money_policy.ai_development_wallet_id,
        ai_development_address=active_money_policy.ai_development_address,
        ai_development_provisioned_locally=False,
        ai_development_seed_file=str(
            ai_development_seed_file_path(wallet_policy)
        ),
        ai_development_password_file=str(
            ai_development_password_file_path(wallet_policy)
        ),
    )
    save_ledger(ledger, wallet_policy)
    return ledger


def save_ledger(ledger: LedgerState, wallet_policy: WalletPolicy | None = None) -> None:
    _write_json(ledger_file_path(wallet_policy), asdict(ledger))


def append_journal_entry(
    entry: JournalEntry, wallet_policy: WalletPolicy | None = None
) -> None:
    path = journal_file_path(wallet_policy)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


def list_journal_entries(
    *,
    wallet_id: str | None = None,
    limit: int | None = None,
    wallet_policy: WalletPolicy | None = None,
) -> list[JournalEntry]:
    path = journal_file_path(wallet_policy)
    if not path.exists():
        return []

    entries: list[JournalEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = JournalEntry(**json.loads(line))
        if wallet_id is not None and entry.wallet_id != wallet_id:
            continue
        entries.append(entry)

    entries.sort(key=lambda item: item.created_at, reverse=True)
    if limit is not None:
        return entries[:limit]
    return entries


def create_wallet(
    name: str,
    password: str,
    *,
    select: bool = False,
    wallet_policy: WalletPolicy | None = None,
) -> WalletRecord:
    active_wallet_policy = wallet_policy or WalletPolicy()
    wallets = list_wallets(active_wallet_policy)
    wallet_id = secrets.token_hex(16)
    salt = secrets.token_bytes(16)
    digest = _hash_password(
        password,
        salt=salt,
        rounds=active_wallet_policy.password_kdf_rounds,
    )
    signing_material = _new_wallet_signing_material(
        wallet_id=wallet_id,
        password=password,
        wallet_policy=active_wallet_policy,
        signing_seed=generate_signing_seed(),
    )
    wallet = WalletRecord(
        wallet_id=wallet_id,
        name=name,
        address=signing_material["address"],
        created_at=_now_iso(),
        password_salt_b64=_encode_bytes(salt),
        password_hash_b64=_encode_bytes(digest),
        signing_scheme=signing_material["signing_scheme"],
        address_scheme=signing_material["address_scheme"],
        public_key_b64=signing_material["public_key_b64"],
        encrypted_private_key_b64=signing_material["encrypted_private_key_b64"],
        private_key_salt_b64=signing_material["private_key_salt_b64"],
        private_key_nonce_b64=signing_material["private_key_nonce_b64"],
        pq_signing_scheme=signing_material["pq_signing_scheme"],
        pq_public_key_b64=signing_material["pq_public_key_b64"],
        encrypted_pq_private_key_b64=signing_material["encrypted_pq_private_key_b64"],
        pq_private_key_salt_b64=signing_material["pq_private_key_salt_b64"],
        pq_private_key_nonce_b64=signing_material["pq_private_key_nonce_b64"],
    )
    wallets.append(wallet)
    save_wallets(wallets, active_wallet_policy)

    session = load_session(active_wallet_policy)
    if select or session.active_wallet_id is None:
        session.active_wallet_id = wallet.wallet_id
        save_session(session, active_wallet_policy)
    append_journal_entry(
        JournalEntry(
            entry_id=secrets.token_hex(12),
            event_type="wallet_created",
            created_at=_now_iso(),
            wallet_id=wallet.wallet_id,
            note=f"Wallet '{wallet.name}' created.",
        ),
        active_wallet_policy,
    )
    return wallet


def create_seed_wallet(
    name: str,
    password: str,
    *,
    select: bool = False,
    wallet_policy: WalletPolicy | None = None,
) -> tuple[WalletRecord, str]:
    seed_phrase = generate_seed_phrase(DEFAULT_SEED_WORD_COUNT)
    wallet = restore_wallet_from_seed(
        name,
        password,
        seed_phrase=seed_phrase,
        select=select,
        wallet_policy=wallet_policy,
    )
    append_journal_entry(
        JournalEntry(
            entry_id=secrets.token_hex(12),
            event_type="wallet_seed_generated",
            created_at=_now_iso(),
            wallet_id=wallet.wallet_id,
            note="Seed phrase generated for wallet creation.",
        ),
        wallet_policy,
    )
    return wallet, seed_phrase


def restore_wallet_from_seed(
    name: str,
    password: str,
    *,
    seed_phrase: str,
    select: bool = False,
    wallet_policy: WalletPolicy | None = None,
    forced_address: str | None = None,
) -> WalletRecord:
    active_wallet_policy = wallet_policy or WalletPolicy()
    wallets = list_wallets(active_wallet_policy)
    normalized_seed = validate_seed_phrase(seed_phrase)
    wallet_id = derive_seed_wallet_id(normalized_seed)
    if any(existing.wallet_id == wallet_id for existing in wallets):
        raise ValueError("A wallet restored from this seed phrase already exists.")

    salt = secrets.token_bytes(16)
    digest = _hash_password(
        password,
        salt=salt,
        rounds=active_wallet_policy.password_kdf_rounds,
    )
    signing_seed = derive_signing_seed_from_seed_phrase(normalized_seed)
    pq_signing_seed = derive_pq_signing_seed_from_seed_phrase(normalized_seed)
    signing_material = _new_wallet_signing_material(
        wallet_id=wallet_id,
        password=password,
        wallet_policy=active_wallet_policy,
        signing_seed=signing_seed,
        pq_signing_seed=pq_signing_seed,
        forced_address=forced_address,
    )
    wallet = WalletRecord(
        wallet_id=wallet_id,
        name=name,
        address=signing_material["address"],
        created_at=_now_iso(),
        password_salt_b64=_encode_bytes(salt),
        password_hash_b64=_encode_bytes(digest),
        seed_fingerprint=seed_fingerprint(normalized_seed),
        seed_word_count=len(normalized_seed.split()),
        signing_scheme=signing_material["signing_scheme"],
        address_scheme=signing_material["address_scheme"],
        public_key_b64=signing_material["public_key_b64"],
        encrypted_private_key_b64=signing_material["encrypted_private_key_b64"],
        private_key_salt_b64=signing_material["private_key_salt_b64"],
        private_key_nonce_b64=signing_material["private_key_nonce_b64"],
        pq_signing_scheme=signing_material["pq_signing_scheme"],
        pq_public_key_b64=signing_material["pq_public_key_b64"],
        encrypted_pq_private_key_b64=signing_material["encrypted_pq_private_key_b64"],
        pq_private_key_salt_b64=signing_material["pq_private_key_salt_b64"],
        pq_private_key_nonce_b64=signing_material["pq_private_key_nonce_b64"],
    )
    wallets.append(wallet)
    save_wallets(wallets, active_wallet_policy)

    session = load_session(active_wallet_policy)
    if select or session.active_wallet_id is None:
        session.active_wallet_id = wallet.wallet_id
        save_session(session, active_wallet_policy)
    append_journal_entry(
        JournalEntry(
            entry_id=secrets.token_hex(12),
            event_type="wallet_restored",
            created_at=_now_iso(),
            wallet_id=wallet.wallet_id,
            note=f"Wallet '{wallet.name}' restored from seed phrase.",
        ),
        active_wallet_policy,
    )
    return wallet


def resolve_wallet(selector: str, wallets: list[WalletRecord]) -> WalletRecord | None:
    selector_lower = selector.lower()
    normalized_address = normalize_address(selector)
    for wallet in wallets:
        if (
            wallet.wallet_id.lower() == selector_lower
            or normalize_address(wallet.address) == normalized_address
            or wallet.name.lower() == selector_lower
        ):
            return wallet
    return None


def get_active_wallet(wallet_policy: WalletPolicy | None = None) -> WalletRecord | None:
    session = load_session(wallet_policy)
    if session.active_wallet_id is None:
        return None
    wallets = list_wallets(wallet_policy)
    return next(
        (wallet for wallet in wallets if wallet.wallet_id == session.active_wallet_id),
        None,
    )


def select_active_wallet(
    selector: str, wallet_policy: WalletPolicy | None = None
) -> WalletRecord:
    active_wallet_policy = wallet_policy or WalletPolicy()
    wallets = list_wallets(active_wallet_policy)
    wallet = resolve_wallet(selector, wallets)
    if wallet is None:
        raise ValueError(f"Wallet '{selector}' not found.")
    session = load_session(active_wallet_policy)
    if session.unlocked_wallet_id != wallet.wallet_id:
        session.unlocked_wallet_id = None
        session.unlocked_at = None
        _delete_unlocked_signing_key(active_wallet_policy)
    session.active_wallet_id = wallet.wallet_id
    save_session(session, active_wallet_policy)
    return wallet


def verify_wallet_password(
    wallet: WalletRecord,
    password: str,
    *,
    wallet_policy: WalletPolicy | None = None,
) -> bool:
    active_wallet_policy = wallet_policy or WalletPolicy()
    digest = _hash_password(
        password,
        salt=_decode_bytes(wallet.password_salt_b64),
        rounds=active_wallet_policy.password_kdf_rounds,
    )
    return secrets.compare_digest(
        digest,
        _decode_bytes(wallet.password_hash_b64),
    )


def _wallet_has_signing_key(wallet: WalletRecord) -> bool:
    base_key_available = bool(
        wallet.public_key_b64
        and wallet.encrypted_private_key_b64
        and wallet.private_key_salt_b64
        and wallet.private_key_nonce_b64
    )
    if wallet.signing_scheme == SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65:
        return bool(
            base_key_available
            and wallet.pq_signing_scheme == SIGNING_SCHEME_ML_DSA_65
            and wallet.pq_public_key_b64
            and wallet.encrypted_pq_private_key_b64
            and wallet.pq_private_key_salt_b64
            and wallet.pq_private_key_nonce_b64
        )
    return bool(
        wallet.signing_scheme == SIGNING_SCHEME_ED25519
        and base_key_available
    )


def _write_unlocked_signing_key(
    wallet: WalletRecord,
    password: str,
    *,
    wallet_policy: WalletPolicy,
) -> None:
    if not _wallet_has_signing_key(wallet):
        _delete_unlocked_signing_key(wallet_policy)
        return
    assert wallet.encrypted_private_key_b64 is not None
    assert wallet.private_key_salt_b64 is not None
    assert wallet.private_key_nonce_b64 is not None
    signing_seed = decrypt_signing_seed(
        encrypted_private_key_b64=wallet.encrypted_private_key_b64,
        private_key_salt_b64=wallet.private_key_salt_b64,
        private_key_nonce_b64=wallet.private_key_nonce_b64,
        password=password,
        wallet_id=wallet.wallet_id,
        address=wallet.address,
        kdf_rounds=wallet_policy.password_kdf_rounds,
    )
    unlocked_payload = {
        "wallet_id": wallet.wallet_id,
        "address": normalize_address(wallet.address),
        "address_scheme": wallet.address_scheme,
        "signing_scheme": wallet.signing_scheme,
        "public_key_b64": wallet.public_key_b64,
        "signing_seed_b64": encode_bytes(signing_seed),
        "unlocked_at": _now_iso(),
    }
    if wallet.signing_scheme == SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65:
        assert wallet.encrypted_pq_private_key_b64 is not None
        assert wallet.pq_private_key_salt_b64 is not None
        assert wallet.pq_private_key_nonce_b64 is not None
        pq_private_key = decrypt_signing_seed(
            encrypted_private_key_b64=wallet.encrypted_pq_private_key_b64,
            private_key_salt_b64=wallet.pq_private_key_salt_b64,
            private_key_nonce_b64=wallet.pq_private_key_nonce_b64,
            password=password,
            wallet_id=wallet.wallet_id,
            address=wallet.address,
            kdf_rounds=wallet_policy.password_kdf_rounds,
        )
        unlocked_payload.update(
            {
                "pq_signing_scheme": wallet.pq_signing_scheme,
                "pq_public_key_b64": wallet.pq_public_key_b64,
                "pq_private_key_b64": encode_bytes(pq_private_key),
            }
        )
    _write_secret_text(
        unlocked_signing_key_file_path(wallet_policy),
        json.dumps(unlocked_payload, ensure_ascii=False, indent=2),
    )


def _delete_unlocked_signing_key(wallet_policy: WalletPolicy | None = None) -> None:
    try:
        unlocked_signing_key_file_path(wallet_policy).unlink(missing_ok=True)
    except OSError:
        pass


def _load_unlocked_signing_key(
    wallet: WalletRecord,
    wallet_policy: WalletPolicy | None = None,
) -> dict[str, Any] | None:
    path = unlocked_signing_key_file_path(wallet_policy)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if str(payload.get("wallet_id") or "") != wallet.wallet_id:
        return None
    if normalize_address(str(payload.get("address") or "")) != normalize_address(
        wallet.address
    ):
        return None
    return payload


def load_unlocked_wallet_signing_material(
    wallet: WalletRecord,
    wallet_policy: WalletPolicy | None = None,
) -> dict[str, Any] | None:
    return _load_unlocked_signing_key(wallet, wallet_policy)


def unlock_wallet(
    password: str,
    *,
    selector: str | None = None,
    wallet_policy: WalletPolicy | None = None,
) -> WalletRecord:
    active_wallet_policy = wallet_policy or WalletPolicy()
    wallets = list_wallets(active_wallet_policy)
    wallet = (
        resolve_wallet(selector, wallets)
        if selector is not None
        else get_active_wallet(active_wallet_policy)
    )
    if wallet is None:
        raise ValueError("Active wallet is not set.")
    if not verify_wallet_password(wallet, password, wallet_policy=active_wallet_policy):
        raise ValueError("Invalid wallet password.")

    session = load_session(active_wallet_policy)
    session.active_wallet_id = wallet.wallet_id
    session.unlocked_wallet_id = wallet.wallet_id
    session.unlocked_at = _now_iso()
    save_session(session, active_wallet_policy)
    _write_unlocked_signing_key(
        wallet,
        password,
        wallet_policy=active_wallet_policy,
    )
    return wallet


def lock_wallet(wallet_policy: WalletPolicy | None = None) -> WalletSession:
    session = load_session(wallet_policy)
    session.unlocked_wallet_id = None
    session.unlocked_at = None
    save_session(session, wallet_policy)
    _delete_unlocked_signing_key(wallet_policy)
    return session


def logout_wallet(wallet_policy: WalletPolicy | None = None) -> WalletSession:
    session = load_session(wallet_policy)
    session.active_wallet_id = None
    session.unlocked_wallet_id = None
    session.unlocked_at = None
    save_session(session, wallet_policy)
    _delete_unlocked_signing_key(wallet_policy)
    return session


def credit_wallet(
    selector: str,
    amount_atomic: int,
    *,
    wallet_policy: WalletPolicy | None = None,
) -> WalletRecord:
    active_wallet_policy = wallet_policy or WalletPolicy()
    wallets = list_wallets(active_wallet_policy)
    wallet = resolve_wallet(selector, wallets)
    if wallet is None:
        raise ValueError(f"Wallet '{selector}' not found.")
    wallet.spendable_balance_atomic += amount_atomic
    save_wallets(wallets, active_wallet_policy)
    append_journal_entry(
        JournalEntry(
            entry_id=secrets.token_hex(12),
            event_type="wallet_credit",
            created_at=_now_iso(),
            wallet_id=wallet.wallet_id,
            amount_atomic=amount_atomic,
            note="Local prototype credit applied.",
        ),
        active_wallet_policy,
    )
    return wallet


def ensure_local_developer_treasury_wallet(
    *,
    money_policy: MoneyPolicy | None = None,
    wallet_policy: WalletPolicy | None = None,
) -> WalletRecord:
    active_money_policy = money_policy or MoneyPolicy()
    active_wallet_policy = wallet_policy or WalletPolicy()
    ledger = load_or_create_ledger(active_money_policy, active_wallet_policy)
    seed_path = developer_treasury_seed_file_path(active_wallet_policy)
    if not seed_path.exists():
        raise ValueError(
            f"Developer treasury seed file is missing: {seed_path}"
        )
    seed_phrase = seed_path.read_text(encoding="utf-8").strip()
    seed_identity = derive_seed_wallet_identity(seed_phrase)
    if seed_identity.wallet_id != active_money_policy.developer_treasury_wallet_id:
        raise ValueError(
            "Developer treasury seed does not match the configured wallet id."
        )

    password_path = developer_treasury_password_file_path(active_wallet_policy)
    if password_path.exists():
        password = password_path.read_text(encoding="utf-8").strip()
    else:
        password = secrets.token_urlsafe(32)
        _write_secret_text(password_path, password)

    configured_address = normalize_address(
        active_money_policy.developer_treasury_address
    )
    forced_address = (
        None
        if configured_address == seed_identity.address
        else active_money_policy.developer_treasury_address
    )
    wallet = find_wallet_by_id(
        active_money_policy.developer_treasury_wallet_id,
        active_wallet_policy,
    )
    if wallet is None:
        wallet = restore_wallet_from_seed(
            f"Developer Treasury ({active_money_policy.chain_network.value})",
            password,
            seed_phrase=seed_phrase,
            select=False,
            wallet_policy=active_wallet_policy,
            forced_address=forced_address,
        )

    if wallet.wallet_id != active_money_policy.developer_treasury_wallet_id:
        raise ValueError(
            "Developer treasury wallet does not match the configured wallet id."
        )
    if normalize_address(wallet.address) != configured_address:
        raise ValueError("Developer treasury seed does not match the configured address.")
    if (
        forced_address is None
        and wallet.address_scheme != seed_identity.address_scheme
    ):
        raise ValueError(
            "Developer treasury wallet must use the configured hybrid address scheme."
        )

    if not ledger.developer_treasury_provisioned_locally:
        wallet.spendable_balance_atomic += ledger.developer_treasury_allocated_atomic
        update_wallet(wallet, active_wallet_policy)
        ledger.developer_treasury_provisioned_locally = True
        ledger.developer_treasury_wallet_id = wallet.wallet_id
        ledger.developer_treasury_address = wallet.address
        ledger.developer_treasury_seed_file = str(seed_path)
        ledger.developer_treasury_password_file = str(password_path)
        save_ledger(ledger, active_wallet_policy)
        append_journal_entry(
            JournalEntry(
                entry_id=secrets.token_hex(12),
                event_type="developer_treasury_provisioned",
                created_at=_now_iso(),
                wallet_id=wallet.wallet_id,
                amount_atomic=ledger.developer_treasury_allocated_atomic,
                note=(
                    f"Developer treasury genesis allocation provisioned for "
                    f"{active_money_policy.chain_network.value}."
                ),
            ),
            active_wallet_policy,
        )
    return wallet


def ensure_local_ai_development_wallet(
    *,
    money_policy: MoneyPolicy | None = None,
    wallet_policy: WalletPolicy | None = None,
) -> WalletRecord:
    active_money_policy = money_policy or MoneyPolicy()
    active_wallet_policy = wallet_policy or WalletPolicy()
    ledger = load_or_create_ledger(active_money_policy, active_wallet_policy)
    seed_path = ai_development_seed_file_path(active_wallet_policy)
    if not seed_path.exists():
        raise ValueError(
            f"AI development seed file is missing: {seed_path}"
        )
    seed_phrase = seed_path.read_text(encoding="utf-8").strip()
    seed_identity = derive_seed_wallet_identity(seed_phrase)
    if seed_identity.wallet_id != active_money_policy.ai_development_wallet_id:
        raise ValueError(
            "AI development seed does not match the configured wallet id."
        )

    password_path = ai_development_password_file_path(active_wallet_policy)
    if password_path.exists():
        password = password_path.read_text(encoding="utf-8").strip()
    else:
        password = secrets.token_urlsafe(32)
        _write_secret_text(password_path, password)

    configured_address = normalize_address(active_money_policy.ai_development_address)
    forced_address = (
        None
        if configured_address == seed_identity.address
        else active_money_policy.ai_development_address
    )
    wallet = find_wallet_by_id(
        active_money_policy.ai_development_wallet_id,
        active_wallet_policy,
    )
    if wallet is None:
        wallet = restore_wallet_from_seed(
            f"AI Development ({active_money_policy.chain_network.value})",
            password,
            seed_phrase=seed_phrase,
            select=False,
            wallet_policy=active_wallet_policy,
            forced_address=forced_address,
        )

    if wallet.wallet_id != active_money_policy.ai_development_wallet_id:
        raise ValueError(
            "AI development wallet does not match the configured wallet id."
        )
    if normalize_address(wallet.address) != configured_address:
        raise ValueError("AI development seed does not match the configured address.")
    if (
        forced_address is None
        and wallet.address_scheme != seed_identity.address_scheme
    ):
        raise ValueError(
            "AI development wallet must use the configured hybrid address scheme."
        )

    if not ledger.ai_development_provisioned_locally:
        ledger.ai_development_provisioned_locally = True
        ledger.ai_development_wallet_id = wallet.wallet_id
        ledger.ai_development_address = wallet.address
        ledger.ai_development_seed_file = str(seed_path)
        ledger.ai_development_password_file = str(password_path)
        save_ledger(ledger, active_wallet_policy)
        append_journal_entry(
            JournalEntry(
                entry_id=secrets.token_hex(12),
                event_type="ai_development_wallet_provisioned",
                created_at=_now_iso(),
                wallet_id=wallet.wallet_id,
                amount_atomic=0,
                note=(
                    f"AI development wallet provisioned for "
                    f"{active_money_policy.chain_network.value}."
                ),
            ),
            active_wallet_policy,
        )
    return wallet


def update_wallet(wallet: WalletRecord, wallet_policy: WalletPolicy | None = None) -> None:
    active_wallet_policy = wallet_policy or WalletPolicy()
    wallets = list_wallets(active_wallet_policy)
    for index, existing in enumerate(wallets):
        if existing.wallet_id == wallet.wallet_id:
            wallets[index] = wallet
            save_wallets(wallets, active_wallet_policy)
            return
    raise ValueError(f"Wallet '{wallet.wallet_id}' not found.")


def find_wallet_by_address(
    address: str, wallet_policy: WalletPolicy | None = None
) -> WalletRecord | None:
    wallets = list_wallets(wallet_policy)
    address_lower = normalize_address(address)
    return next(
        (wallet for wallet in wallets if normalize_address(wallet.address) == address_lower),
        None,
    )


def find_wallet_by_id(
    wallet_id: str, wallet_policy: WalletPolicy | None = None
) -> WalletRecord | None:
    wallets = list_wallets(wallet_policy)
    return next((wallet for wallet in wallets if wallet.wallet_id == wallet_id), None)


def apply_wallet_transfer(
    *,
    sender_wallet_id: str,
    recipient_address: str,
    amount_atomic: int,
    tx_fee_atomic: int,
    wallet_policy: WalletPolicy | None = None,
) -> tuple[WalletRecord, WalletRecord | None]:
    active_wallet_policy = wallet_policy or WalletPolicy()
    wallets = list_wallets(active_wallet_policy)
    sender = next((wallet for wallet in wallets if wallet.wallet_id == sender_wallet_id), None)
    if sender is None:
        raise ValueError(f"Wallet '{sender_wallet_id}' not found.")
    if amount_atomic <= 0:
        raise ValueError("Transfer amount must be positive.")

    total_debit = amount_atomic + tx_fee_atomic
    recipient: WalletRecord | None = None
    recipient_lower = normalize_address(recipient_address)
    for wallet in wallets:
        if normalize_address(wallet.address) == recipient_lower:
            recipient = wallet
            break

    from .chain import (
        append_chain_block,
        chain_balance_atomic,
        chain_is_initialized,
        make_chain_transaction,
        transaction_signing_payload,
        tx_fee_pool_chain_address,
    )
    from .validators import (
        get_validator_record,
        select_validator_committee_snapshot,
        split_amount_by_validator_bond,
    )

    chain_backed_transfer = chain_is_initialized(active_wallet_policy)
    if chain_backed_transfer:
        active_money_policy = MoneyPolicy(chain_network=active_wallet_policy.chain_network)
        sender_chain_balance = chain_balance_atomic(sender.address, active_wallet_policy)
        if sender_chain_balance < total_debit:
            raise ValueError("Wallet chain balance is insufficient for transfer amount and tx fee.")
        transfer_id = secrets.token_hex(12)
        chain_id = active_money_policy.chain_network.value
        common_metadata = {
            "transfer_id": transfer_id,
            "amount_atomic": int(amount_atomic),
            "tx_fee_atomic": int(tx_fee_atomic),
            "reward_token_code": active_money_policy.reward_token_code,
        }
        debit_metadata = dict(common_metadata)
        public_key_b64 = None
        signature_b64 = None
        if sender.address_scheme in {
            ADDRESS_SCHEME_ED25519,
            ADDRESS_SCHEME_FIXED_WALLET_ID_SHA256,
            *HYBRID_ADDRESS_SCHEMES,
        }:
            signer = _load_unlocked_signing_key(sender, active_wallet_policy)
            if signer is None:
                raise ValueError(
                    "Active wallet must be unlocked again before signing an on-chain transfer."
                )
            public_key_b64 = str(signer.get("public_key_b64") or "")
            debit_metadata["signature_required"] = True
            debit_metadata["address_scheme"] = str(sender.address_scheme or "")
            if sender.address_scheme == ADDRESS_SCHEME_FIXED_WALLET_ID_SHA256:
                debit_metadata["signing_scheme"] = SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65
            if sender.address_scheme in HYBRID_ADDRESS_SCHEMES or (
                sender.address_scheme == ADDRESS_SCHEME_FIXED_WALLET_ID_SHA256
            ):
                debit_metadata["pq_signature_scheme"] = SIGNING_SCHEME_ML_DSA_65
                debit_metadata["pq_public_key_b64"] = str(
                    signer.get("pq_public_key_b64") or ""
                )
        debit_tx = make_chain_transaction(
            tx_type="wallet_transfer_debit",
            address=sender.address,
            delta_atomic=-total_debit,
            wallet_id=sender.wallet_id,
            counterparty_address=recipient_lower,
            nonce=f"{transfer_id}:debit",
            metadata=debit_metadata,
            chain_id=chain_id,
            public_key_b64=public_key_b64,
        )
        if public_key_b64:
            signing_seed_b64 = str(signer.get("signing_seed_b64") or "")
            signature_b64 = sign_payload_b64(
                _decode_bytes(signing_seed_b64),
                transaction_signing_payload(debit_tx),
            )
            debit_tx.signature_b64 = signature_b64
            if sender.address_scheme in HYBRID_ADDRESS_SCHEMES or (
                sender.address_scheme == ADDRESS_SCHEME_FIXED_WALLET_ID_SHA256
            ):
                pq_private_key_b64 = str(signer.get("pq_private_key_b64") or "")
                debit_tx.metadata["pq_signature_b64"] = sign_payload_mldsa65_b64(
                    pq_private_key_b64,
                    transaction_signing_payload(debit_tx),
                )
        fee_committee = select_validator_committee_snapshot(
            selection_seed=transfer_id,
            policy=active_wallet_policy,
            money_policy=active_money_policy,
        )
        validator_fee_shares = split_amount_by_validator_bond(
            tx_fee_atomic,
            dict(fee_committee.bonded_atomic_by_validator_id),
            validator_ids=list(fee_committee.validator_ids),
        )
        fee_transactions = []
        if validator_fee_shares:
            total_bonded_atomic = sum(
                int(item.bonded_atomic) for item in validator_fee_shares
            )
            for share in validator_fee_shares:
                validator_record = get_validator_record(
                    share.validator_id,
                    active_wallet_policy,
                )
                fee_transactions.append(
                    make_chain_transaction(
                        tx_type="wallet_transfer_validator_fee_payout",
                        address=share.validator_id,
                        delta_atomic=share.amount_atomic,
                        wallet_id=(
                            validator_record.wallet_id
                            if validator_record is not None
                            else None
                        ),
                        counterparty_address=sender.address,
                        nonce=f"{transfer_id}:validator-fee:{share.validator_id}",
                        metadata={
                            **common_metadata,
                            "validator_id": share.validator_id,
                            "validator_bonded_atomic": int(share.bonded_atomic),
                            "committee_total_bonded_atomic": int(total_bonded_atomic),
                            "committee_validator_ids": list(fee_committee.validator_ids),
                            "committee_quorum_bond_atomic": int(
                                fee_committee.quorum_bond_atomic
                            ),
                            "distribution": "committee_bond_weighted",
                        },
                        chain_id=chain_id,
                    )
                )
        else:
            fee_transactions.append(
                make_chain_transaction(
                    tx_type="wallet_transfer_fee_credit",
                    address=tx_fee_pool_chain_address(active_money_policy),
                    delta_atomic=tx_fee_atomic,
                    wallet_id=f"system-tx-fee-pool-{chain_id}",
                    counterparty_address=sender.address,
                    nonce=f"{transfer_id}:fee",
                    metadata={**common_metadata, "distribution": "legacy_pool"},
                    chain_id=chain_id,
                )
            )
        append_chain_block(
            [
                debit_tx,
                make_chain_transaction(
                    tx_type="wallet_transfer_credit",
                    address=recipient_lower,
                    delta_atomic=amount_atomic,
                    wallet_id=recipient.wallet_id if recipient is not None else None,
                    counterparty_address=sender.address,
                    nonce=f"{transfer_id}:credit",
                    metadata=common_metadata,
                    chain_id=chain_id,
                ),
                *fee_transactions,
            ],
            policy=active_wallet_policy,
        )
        affected_addresses = {
            normalize_address(sender.address),
            recipient_lower,
            *{
                normalize_address(str(tx.address or ""))
                for tx in fee_transactions
                if str(tx.address or "").strip()
            },
        }
        for wallet in wallets:
            if normalize_address(wallet.address) in affected_addresses:
                wallet.spendable_balance_atomic = chain_balance_atomic(
                    wallet.address,
                    active_wallet_policy,
                )
        save_wallets(wallets, active_wallet_policy)
        return sender, recipient

    if sender.spendable_balance_atomic < total_debit:
        raise ValueError("Wallet balance is insufficient for transfer amount and tx fee.")

    sender.spendable_balance_atomic -= total_debit
    if recipient is not None:
        recipient.spendable_balance_atomic += amount_atomic

    save_wallets(wallets, active_wallet_policy)
    return sender, recipient


def coins_to_atomic(value: str, money_policy: MoneyPolicy | None = None) -> int:
    active_money_policy = money_policy or MoneyPolicy()
    whole, _, fraction = value.partition(".")
    if not whole:
        whole = "0"
    fraction = (fraction + ("0" * active_money_policy.decimals))[
        : active_money_policy.decimals
    ]
    sign = -1 if whole.startswith("-") else 1
    whole_digits = whole[1:] if sign < 0 else whole
    return sign * (
        int(whole_digits) * (10**active_money_policy.decimals) + int(fraction or "0")
    )


def atomic_to_coins(value: int, money_policy: MoneyPolicy | None = None) -> str:
    active_money_policy = money_policy or MoneyPolicy()
    sign = "-" if value < 0 else ""
    abs_value = abs(value)
    scale = 10**active_money_policy.decimals
    whole = abs_value // scale
    fraction = abs_value % scale
    return f"{sign}{whole}.{fraction:0{active_money_policy.decimals}d}"

