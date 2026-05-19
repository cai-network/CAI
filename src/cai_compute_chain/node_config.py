# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import ipaddress
import json
import os
import secrets
import socket
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from urllib.parse import urlparse

from .chain import (
    append_chain_block,
    chain_balance_atomic,
    chain_is_initialized,
    ensure_chain_genesis,
    list_chain_blocks,
    make_chain_transaction,
    validator_bond_pool_chain_address,
    validator_locked_bond_index,
    validator_slash_pool_chain_address,
    wallet_chain_balance_or_local_atomic,
)
from .model import (
    LEGACY_PRIVATE_NETWORK_MODEL_ID,
    curated_model_registry,
    curated_worker_default_model_ids,
    MoneyPolicy,
    NetworkModelPolicy,
    normalize_network_model_id,
    ValidatorLifecycleState,
    WalletPolicy,
)
from .wallet import (
    JournalEntry,
    append_journal_entry,
    coins_to_atomic,
    data_root,
    find_wallet_by_id,
    get_active_wallet,
    load_or_create_ledger,
    load_session,
    normalize_address,
    save_ledger,
    update_wallet,
)
from .validators import (
    get_validator_record,
    sync_validator_record,
    validator_ha_lease_is_active,
)


def _coalesce_cai_url(cai_url: str | None = None, CAI_url: str | None = None) -> str | None:
    resolved = str(cai_url or CAI_url or "").strip()
    return resolved or None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def default_relay_enabled() -> bool:
    return _env_bool("CAI_RELAY_ENABLED_BY_DEFAULT", True)


VALIDATOR_HA_ROLE_STANDALONE = "standalone"
VALIDATOR_HA_ROLE_ACTIVE = "active"
VALIDATOR_HA_ROLE_PASSIVE = "passive"
VALIDATOR_HA_DEFAULT_LEASE_SECONDS = 90
VALIDATOR_HA_MIN_LEASE_SECONDS = 15
VALIDATOR_HA_MAX_LEASE_SECONDS = 3600
VALIDATOR_HA_ROLES = {
    VALIDATOR_HA_ROLE_STANDALONE,
    VALIDATOR_HA_ROLE_ACTIVE,
    VALIDATOR_HA_ROLE_PASSIVE,
}


def _normalize_validator_ha_lease_seconds(value: object) -> int:
    try:
        seconds = int(value)  # pyright: ignore[reportArgumentType]
    except (TypeError, ValueError):
        seconds = VALIDATOR_HA_DEFAULT_LEASE_SECONDS
    return min(
        VALIDATOR_HA_MAX_LEASE_SECONDS,
        max(VALIDATOR_HA_MIN_LEASE_SECONDS, seconds),
    )


@dataclass
class NodeRuntimeConfig:
    validator_enabled: bool = False
    validator_state: str = ValidatorLifecycleState.UNBONDED
    validator_wallet_id: str | None = None
    validator_address: str | None = None
    validator_bond_atomic: int = 0
    validator_static_ip_confirmed: bool = False
    validator_unbonding_started_at: str | None = None
    validator_unbonding_available_at: str | None = None
    validator_jailed_at: str | None = None
    validator_unjail_available_at: str | None = None
    validator_jail_reason: str | None = None
    validator_last_slash_atomic: int = 0
    validator_total_slashed_atomic: int = 0
    validator_ha_enabled: bool = False
    validator_ha_role: str = VALIDATOR_HA_ROLE_STANDALONE
    validator_ha_replica_id: str | None = None
    validator_ha_auto_failover_enabled: bool = True
    validator_ha_lease_seconds: int = VALIDATOR_HA_DEFAULT_LEASE_SECONDS
    worker_enabled: bool = False
    relay_enabled: bool = field(default_factory=default_relay_enabled)
    relay_mode_manually_configured: bool = False
    worker_allowed_model_ids: list[str] = field(default_factory=list)
    worker_max_parallel_jobs: int = 1
    worker_max_memory_mb: int | None = None
    worker_reward_address_by_node_id: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidatorModeStatus:
    can_enable: bool
    state: str
    reason: str
    required_bond_atomic: int
    current_spendable_atomic: int
    bonded_atomic: int
    active_wallet_id: str | None
    active_wallet_address: str | None
    active_wallet_unlocked: bool
    network_ok: bool = False
    static_ip_confirmed: bool = False
    current_node_id: str | None = None
    advertised_api_host: str | None = None
    advertised_data_host: str | None = None
    validator_ha_enabled: bool = False
    validator_ha_role: str = VALIDATOR_HA_ROLE_STANDALONE
    validator_ha_replica_id: str | None = None
    validator_ha_auto_failover_enabled: bool = True
    validator_ha_lease_seconds: int = VALIDATOR_HA_DEFAULT_LEASE_SECONDS


def node_config_file_path(policy: WalletPolicy | None = None):
    active_policy = policy or WalletPolicy()
    return data_root(active_policy) / active_policy.node_config_file_name


def _canonical_worker_allowed_model_id(model_id: str) -> str:
    normalized_model_id = normalize_network_model_id(str(model_id).strip())
    for registry_model in curated_model_registry():
        if normalized_model_id == normalize_network_model_id(registry_model.model_id):
            return registry_model.model_id
    for registry_model in curated_model_registry():
        normalized_runtime_ids = {
            normalize_network_model_id(item)
            for item in (
                registry_model.execution_model_id,
                *registry_model.runtime_model_ids,
            )
        }
        if normalized_model_id in normalized_runtime_ids:
            return registry_model.model_id
    return normalized_model_id


def _legacy_collapsed_worker_default_model_ids() -> list[str]:
    collapsed: list[str] = []
    for model_id in curated_worker_default_model_ids():
        normalized_model_id = normalize_network_model_id(str(model_id).strip())
        resolved = normalized_model_id
        for registry_model in curated_model_registry():
            normalized_runtime_ids = {
                normalize_network_model_id(item)
                for item in (
                    registry_model.model_id,
                    registry_model.execution_model_id,
                    *registry_model.runtime_model_ids,
                )
            }
            if normalized_model_id in normalized_runtime_ids:
                resolved = registry_model.model_id
                break
        if resolved not in collapsed:
            collapsed.append(resolved)
    return collapsed


def default_worker_allowed_model_ids() -> list[str]:
    return list(curated_worker_default_model_ids())


def load_or_create_node_config(policy: WalletPolicy | None = None) -> NodeRuntimeConfig:
    path = node_config_file_path(policy)
    if not path.exists():
        config = NodeRuntimeConfig(
            worker_allowed_model_ids=default_worker_allowed_model_ids()
        )
        save_node_config(config, policy)
        return config

    config = NodeRuntimeConfig(**json.loads(path.read_text(encoding="utf-8")))
    changed = False

    normalized_allowed_model_ids = list(
        dict.fromkeys(
            _canonical_worker_allowed_model_id(str(model_id).strip())
            for model_id in config.worker_allowed_model_ids
            if str(model_id).strip()
        )
    )
    if normalized_allowed_model_ids != config.worker_allowed_model_ids:
        config.worker_allowed_model_ids = normalized_allowed_model_ids
        changed = True

    legacy_default_model_ids = {
        _canonical_worker_allowed_model_id(LEGACY_PRIVATE_NETWORK_MODEL_ID),
        _canonical_worker_allowed_model_id(NetworkModelPolicy().network_default_model_id),
    }
    if config.worker_allowed_model_ids and set(
        config.worker_allowed_model_ids
    ).issubset(legacy_default_model_ids):
        default_allowed_model_ids = default_worker_allowed_model_ids()
        if config.worker_allowed_model_ids != default_allowed_model_ids:
            config.worker_allowed_model_ids = default_allowed_model_ids
            changed = True

    legacy_collapsed_default_allowed_model_ids = (
        _legacy_collapsed_worker_default_model_ids()
    )
    default_allowed_model_ids = default_worker_allowed_model_ids()
    if (
        config.worker_allowed_model_ids == legacy_collapsed_default_allowed_model_ids
        and config.worker_allowed_model_ids != default_allowed_model_ids
    ):
        config.worker_allowed_model_ids = default_allowed_model_ids
        changed = True

    if not config.relay_mode_manually_configured:
        configured_relay_default = default_relay_enabled()
        if config.relay_enabled != configured_relay_default:
            config.relay_enabled = configured_relay_default
            changed = True

    expected_enabled = config.validator_state == ValidatorLifecycleState.BONDED
    if config.validator_enabled != expected_enabled:
        config.validator_enabled = expected_enabled
        changed = True

    if config.validator_state not in {
        ValidatorLifecycleState.UNBONDED,
        ValidatorLifecycleState.BONDED,
        ValidatorLifecycleState.UNBONDING,
        ValidatorLifecycleState.JAILED,
    }:
        config.validator_state = (
            ValidatorLifecycleState.BONDED
            if config.validator_enabled
            else ValidatorLifecycleState.UNBONDED
        )
        changed = True

    if config.validator_ha_role not in VALIDATOR_HA_ROLES:
        config.validator_ha_role = VALIDATOR_HA_ROLE_STANDALONE
        changed = True

    if not config.validator_ha_enabled:
        if config.validator_ha_role != VALIDATOR_HA_ROLE_STANDALONE:
            config.validator_ha_role = VALIDATOR_HA_ROLE_STANDALONE
            changed = True
        if config.validator_ha_replica_id is not None:
            config.validator_ha_replica_id = None
            changed = True
    elif config.validator_ha_role == VALIDATOR_HA_ROLE_STANDALONE:
        config.validator_ha_role = VALIDATOR_HA_ROLE_PASSIVE
        changed = True

    if config.validator_ha_enabled and not config.validator_ha_replica_id:
        config.validator_ha_replica_id = secrets.token_hex(8)
        changed = True

    normalized_ha_lease_seconds = _normalize_validator_ha_lease_seconds(
        config.validator_ha_lease_seconds
    )
    if config.validator_ha_lease_seconds != normalized_ha_lease_seconds:
        config.validator_ha_lease_seconds = normalized_ha_lease_seconds
        changed = True

    if config.validator_address is not None:
        normalized_address = normalize_address(config.validator_address)
        if normalized_address != config.validator_address:
            config.validator_address = normalized_address
            changed = True

    if config.validator_state in {
        ValidatorLifecycleState.BONDED,
        ValidatorLifecycleState.UNBONDING,
    } and (
        not config.validator_wallet_id
        or not config.validator_address
        or config.validator_bond_atomic <= 0
    ):
        config.validator_enabled = False
        config.validator_state = ValidatorLifecycleState.UNBONDED
        config.validator_wallet_id = None
        config.validator_address = None
        config.validator_bond_atomic = 0
        config.validator_unbonding_started_at = None
        config.validator_unbonding_available_at = None
        changed = True

    if config.validator_state in {
        ValidatorLifecycleState.BONDED,
        ValidatorLifecycleState.UNBONDING,
    } and config.worker_enabled:
        config.worker_enabled = False
        changed = True

    if config.validator_state != ValidatorLifecycleState.UNBONDING:
        if config.validator_unbonding_started_at is not None:
            config.validator_unbonding_started_at = None
            changed = True
        if config.validator_unbonding_available_at is not None:
            config.validator_unbonding_available_at = None
            changed = True

    if config.validator_state == ValidatorLifecycleState.UNBONDED and (
        config.validator_wallet_id is not None
        or config.validator_address is not None
        or config.validator_bond_atomic != 0
        or config.validator_ha_enabled
        or config.validator_ha_role != VALIDATOR_HA_ROLE_STANDALONE
        or config.validator_ha_replica_id is not None
    ):
        config.validator_wallet_id = None
        config.validator_address = None
        config.validator_bond_atomic = 0
        config.validator_ha_enabled = False
        config.validator_ha_role = VALIDATOR_HA_ROLE_STANDALONE
        config.validator_ha_replica_id = None
        config.validator_ha_auto_failover_enabled = True
        config.validator_ha_lease_seconds = VALIDATOR_HA_DEFAULT_LEASE_SECONDS
        changed = True

    if (
        config.validator_state == ValidatorLifecycleState.BONDED
        and not _local_validator_config_is_chain_backed(config, policy)
    ):
        stale_validator_wallet_id = config.validator_wallet_id
        config.validator_enabled = False
        config.validator_state = ValidatorLifecycleState.UNBONDED
        config.validator_wallet_id = None
        config.validator_address = None
        config.validator_bond_atomic = 0
        config.validator_ha_enabled = False
        config.validator_ha_role = VALIDATOR_HA_ROLE_STANDALONE
        config.validator_ha_replica_id = None
        config.validator_unbonding_started_at = None
        config.validator_unbonding_available_at = None
        if stale_validator_wallet_id:
            stale_wallet = find_wallet_by_id(stale_validator_wallet_id, policy)
            if stale_wallet is not None and stale_wallet.validator_reserved_atomic:
                stale_wallet.validator_reserved_atomic = 0
                update_wallet(stale_wallet, policy)
        changed = True

    if config.validator_state != ValidatorLifecycleState.JAILED:
        if config.validator_jailed_at is not None:
            config.validator_jailed_at = None
            changed = True
        if config.validator_unjail_available_at is not None:
            config.validator_unjail_available_at = None
            changed = True
        if config.validator_jail_reason is not None:
            config.validator_jail_reason = None
            changed = True
        if config.validator_last_slash_atomic != 0:
            config.validator_last_slash_atomic = 0
            changed = True

    if changed:
        save_node_config(config, policy)
    return config


def save_node_config(
    config: NodeRuntimeConfig, policy: WalletPolicy | None = None
) -> None:
    path = node_config_file_path(policy)
    path.write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")


def _record_validator_bond_lock(
    *,
    wallet,
    bond_atomic: int,
    validator_address: str,
    policy: WalletPolicy | None,
    money_policy: MoneyPolicy,
) -> None:
    chain_id = money_policy.chain_network.value
    validator_id = normalize_address(validator_address)
    if bond_atomic <= 0:
        raise ValueError("Validator self-bond must be positive.")
    current_chain_balance = chain_balance_atomic(wallet.address, policy)
    if current_chain_balance < bond_atomic:
        raise ValueError(
            "Active wallet chain balance is too low for validator mode. "
            f"Minimum self-bond is {money_policy.validator_min_bond_coins}."
        )
    bond_id = secrets.token_hex(12)
    metadata = {
        "bond_id": bond_id,
        "validator_id": validator_id,
        "validator_wallet_id": wallet.wallet_id,
        "validator_address": validator_id,
        "bond_atomic": int(bond_atomic),
        "reward_token_code": money_policy.reward_token_code,
    }
    append_chain_block(
        [
            make_chain_transaction(
                tx_type="validator_bond_lock",
                address=wallet.address,
                delta_atomic=-bond_atomic,
                wallet_id=wallet.wallet_id,
                nonce=f"{bond_id}:wallet-lock",
                metadata=metadata,
                chain_id=chain_id,
            ),
            make_chain_transaction(
                tx_type="validator_bond_pool_credit",
                address=validator_bond_pool_chain_address(money_policy),
                delta_atomic=bond_atomic,
                wallet_id=f"system-validator-bond-pool-{chain_id}",
                counterparty_address=wallet.address,
                nonce=f"{bond_id}:pool-credit",
                metadata=metadata,
                chain_id=chain_id,
            ),
        ],
        validator_id=validator_id,
        policy=policy,
    )


def _validator_bond_is_chain_backed(
    *,
    wallet,
    config: NodeRuntimeConfig,
    policy: WalletPolicy | None,
    money_policy: MoneyPolicy,
) -> bool:
    required_bond_atomic = max(0, int(config.validator_bond_atomic or 0))
    if required_bond_atomic <= 0:
        return False
    validator_id = normalize_address(config.validator_address or wallet.address)
    locked_by_validator = validator_locked_bond_index(policy)
    locked_atomic = max(0, int(locked_by_validator.get(validator_id, 0) or 0))
    total_locked_atomic = sum(
        max(0, int(value or 0)) for value in locked_by_validator.values()
    )
    bond_pool_atomic = chain_balance_atomic(
        validator_bond_pool_chain_address(money_policy),
        policy,
    )
    return (
        chain_is_initialized(policy)
        and _validator_bond_lock_exists(wallet=wallet, config=config, policy=policy)
        and locked_atomic >= required_bond_atomic
        and bond_pool_atomic == total_locked_atomic
    )


def _local_validator_config_is_chain_backed(
    config: NodeRuntimeConfig,
    policy: WalletPolicy | None = None,
) -> bool:
    if (
        config.validator_state != ValidatorLifecycleState.BONDED
        or not config.validator_wallet_id
        or not config.validator_address
        or config.validator_bond_atomic <= 0
    ):
        return False
    wallet = find_wallet_by_id(config.validator_wallet_id, policy)
    if wallet is None:
        return False
    if normalize_address(wallet.address) != normalize_address(config.validator_address):
        return False
    money_policy = MoneyPolicy(chain_network=(policy or WalletPolicy()).chain_network)
    return _validator_bond_is_chain_backed(
        wallet=wallet,
        config=config,
        policy=policy,
        money_policy=money_policy,
    )


def _validator_bond_lock_exists(
    *,
    wallet,
    config: NodeRuntimeConfig,
    policy: WalletPolicy | None,
) -> bool:
    validator_id = normalize_address(config.validator_address or wallet.address)
    wallet_address = normalize_address(wallet.address)
    for block in list_chain_blocks(policy):
        for tx in block.transactions:
            if tx.tx_type != "validator_bond_lock":
                continue
            metadata = tx.metadata or {}
            if normalize_address(tx.address) != wallet_address:
                continue
            if str(metadata.get("validator_wallet_id") or "") != wallet.wallet_id:
                continue
            if normalize_address(metadata.get("validator_id") or "") != validator_id:
                continue
            return True
    return False


def _record_validator_bond_release(
    *,
    wallet,
    config: NodeRuntimeConfig,
    released_atomic: int,
    policy: WalletPolicy | None,
    money_policy: MoneyPolicy,
    note: str,
) -> None:
    if released_atomic <= 0:
        return
    chain_id = money_policy.chain_network.value
    validator_id = normalize_address(config.validator_address or wallet.address)
    release_id = secrets.token_hex(12)
    metadata = {
        "release_id": release_id,
        "validator_id": validator_id,
        "validator_wallet_id": wallet.wallet_id,
        "validator_address": validator_id,
        "released_atomic": int(released_atomic),
        "reward_token_code": money_policy.reward_token_code,
        "note": note,
    }
    append_chain_block(
        [
            make_chain_transaction(
                tx_type="validator_bond_pool_debit",
                address=validator_bond_pool_chain_address(money_policy),
                delta_atomic=-released_atomic,
                wallet_id=f"system-validator-bond-pool-{chain_id}",
                counterparty_address=wallet.address,
                nonce=f"{release_id}:pool-debit",
                metadata=metadata,
                chain_id=chain_id,
            ),
            make_chain_transaction(
                tx_type="validator_bond_release_credit",
                address=wallet.address,
                delta_atomic=released_atomic,
                wallet_id=wallet.wallet_id,
                nonce=f"{release_id}:wallet-credit",
                metadata=metadata,
                chain_id=chain_id,
            ),
        ],
        validator_id=validator_id,
        policy=policy,
    )


def _record_validator_bond_slash(
    *,
    wallet,
    config: NodeRuntimeConfig,
    slash_atomic: int,
    released_atomic: int,
    policy: WalletPolicy | None,
    money_policy: MoneyPolicy,
    reason: str,
) -> None:
    total_locked = max(0, int(slash_atomic) + int(released_atomic))
    if total_locked <= 0:
        return
    chain_id = money_policy.chain_network.value
    validator_id = normalize_address(config.validator_address or wallet.address)
    slash_id = secrets.token_hex(12)
    metadata = {
        "slash_id": slash_id,
        "validator_id": validator_id,
        "validator_wallet_id": wallet.wallet_id,
        "validator_address": validator_id,
        "slash_atomic": int(slash_atomic),
        "released_atomic": int(released_atomic),
        "reward_token_code": money_policy.reward_token_code,
        "reason": reason,
    }
    transactions = [
        make_chain_transaction(
            tx_type="validator_bond_pool_debit",
            address=validator_bond_pool_chain_address(money_policy),
            delta_atomic=-total_locked,
            wallet_id=f"system-validator-bond-pool-{chain_id}",
            counterparty_address=wallet.address,
            nonce=f"{slash_id}:pool-debit",
            metadata=metadata,
            chain_id=chain_id,
        )
    ]
    if released_atomic > 0:
        transactions.append(
            make_chain_transaction(
                tx_type="validator_bond_release_credit",
                address=wallet.address,
                delta_atomic=released_atomic,
                wallet_id=wallet.wallet_id,
                nonce=f"{slash_id}:wallet-credit",
                metadata=metadata,
                chain_id=chain_id,
            )
        )
    if slash_atomic > 0:
        transactions.append(
            make_chain_transaction(
                tx_type="validator_slash_pool_credit",
                address=validator_slash_pool_chain_address(money_policy),
                delta_atomic=slash_atomic,
                wallet_id=f"system-validator-slash-pool-{chain_id}",
                counterparty_address=wallet.address,
                nonce=f"{slash_id}:slash-credit",
                metadata=metadata,
                chain_id=chain_id,
            )
        )
    append_chain_block(transactions, validator_id=validator_id, policy=policy)


def set_validator_mode(
    enabled: bool,
    policy: WalletPolicy | None = None,
    money_policy: MoneyPolicy | None = None,
    state_payload: dict | None = None,
    cai_url: str | None = None,
    CAI_url: str | None = None,
) -> NodeRuntimeConfig:
    config = load_or_create_node_config(policy)
    active_money_policy = money_policy or MoneyPolicy()
    if enabled:
        status = get_validator_mode_status(
            policy=policy,
            money_policy=active_money_policy,
            state_payload=state_payload,
            cai_url=_coalesce_cai_url(cai_url, CAI_url),
        )
        if not status.can_enable:
            raise ValueError(status.reason)

        wallet = get_active_wallet(policy)
        if wallet is None:
            raise ValueError("Active wallet is required before enabling validator mode.")

        if (
            config.validator_state == ValidatorLifecycleState.BONDED
            and config.validator_wallet_id == wallet.wallet_id
            and config.validator_bond_atomic == status.required_bond_atomic
        ):
            return config

        if config.validator_state == ValidatorLifecycleState.UNBONDING:
            raise ValueError(status.reason)
        if config.validator_state == ValidatorLifecycleState.BONDED:
            raise ValueError("Validator is already bonded on this node. Disable it and wait for unbonding before switching validator wallets.")

        bond_atomic = status.required_bond_atomic
        _record_validator_bond_lock(
            wallet=wallet,
            bond_atomic=bond_atomic,
            validator_address=wallet.address,
            policy=policy,
            money_policy=active_money_policy,
        )
        wallet.spendable_balance_atomic = chain_balance_atomic(wallet.address, policy)
        wallet.validator_reserved_atomic = bond_atomic
        update_wallet(wallet, policy)
        append_journal_entry(
            JournalEntry(
                entry_id=secrets.token_hex(12),
                event_type="validator_bond_locked",
                created_at=_now_iso(),
                wallet_id=wallet.wallet_id,
                amount_atomic=bond_atomic,
                note="Validator mode enabled and self-bond locked on-chain.",
            ),
            policy,
        )

        config.validator_enabled = True
        config.validator_state = ValidatorLifecycleState.BONDED
        config.validator_wallet_id = wallet.wallet_id
        config.validator_address = normalize_address(wallet.address)
        config.validator_bond_atomic = bond_atomic
        config.validator_unbonding_started_at = None
        config.validator_unbonding_available_at = None
        sync_validator_id = config.validator_address
    else:
        sync_validator_id = config.validator_address
        if config.validator_state == ValidatorLifecycleState.BONDED:
            config = _start_validator_unbonding(
                config,
                policy=policy,
                money_policy=active_money_policy,
            )
        else:
            config.validator_enabled = False

    save_node_config(config, policy)
    _sync_local_validator_registry(
        config,
        validator_id_override=sync_validator_id,
        state_payload=state_payload,
        cai_url=_coalesce_cai_url(cai_url, CAI_url),
        policy=policy,
    )
    return config


def get_validator_identity(policy: WalletPolicy | None = None) -> str | None:
    config = load_or_create_node_config(policy)
    if config.validator_state != ValidatorLifecycleState.BONDED:
        return None
    if not config.validator_address:
        return None
    if not config.validator_wallet_id:
        return None
    wallet = find_wallet_by_id(config.validator_wallet_id, policy)
    if wallet is None:
        return None
    money_policy = MoneyPolicy(chain_network=(policy or WalletPolicy()).chain_network)
    if not _validator_bond_is_chain_backed(
        wallet=wallet,
        config=config,
        policy=policy,
        money_policy=money_policy,
    ):
        return None
    return normalize_address(config.validator_address)


def set_validator_ha_mode(
    *,
    enabled: bool,
    role: str | None = None,
    replica_id: str | None = None,
    auto_failover: bool | None = None,
    lease_seconds: int | None = None,
    state_payload: dict | None = None,
    cai_url: str | None = None,
    CAI_url: str | None = None,
    policy: WalletPolicy | None = None,
) -> NodeRuntimeConfig:
    config = load_or_create_node_config(policy)
    if auto_failover is not None:
        config.validator_ha_auto_failover_enabled = bool(auto_failover)
    if lease_seconds is not None:
        config.validator_ha_lease_seconds = _normalize_validator_ha_lease_seconds(
            lease_seconds
        )
    if not enabled:
        config.validator_ha_enabled = False
        config.validator_ha_role = VALIDATOR_HA_ROLE_STANDALONE
        config.validator_ha_replica_id = None
        config.validator_ha_auto_failover_enabled = True
        config.validator_ha_lease_seconds = VALIDATOR_HA_DEFAULT_LEASE_SECONDS
        save_node_config(config, policy)
        _sync_local_validator_registry(
            config,
            state_payload=state_payload,
            cai_url=_coalesce_cai_url(cai_url, CAI_url),
            policy=policy,
        )
        return config

    normalized_role = str(role or "").strip().lower()
    if normalized_role not in {VALIDATOR_HA_ROLE_ACTIVE, VALIDATOR_HA_ROLE_PASSIVE}:
        raise ValueError("Validator HA role must be 'active' or 'passive'.")

    normalized_replica_id = str(replica_id or "").strip()
    config.validator_ha_enabled = True
    config.validator_ha_role = normalized_role
    config.validator_ha_replica_id = normalized_replica_id or (
        config.validator_ha_replica_id or secrets.token_hex(8)
    )
    save_node_config(config, policy)
    _sync_local_validator_registry(
        config,
        state_payload=state_payload,
        cai_url=_coalesce_cai_url(cai_url, CAI_url),
        policy=policy,
    )
    if normalized_role == VALIDATOR_HA_ROLE_ACTIVE:
        return refresh_validator_ha_lease(
            state_payload=state_payload,
            cai_url=_coalesce_cai_url(cai_url, CAI_url),
            policy=policy,
            allow_failover=False,
        )
    return config


def refresh_validator_ha_lease(
    *,
    state_payload: dict | None = None,
    cai_url: str | None = None,
    CAI_url: str | None = None,
    policy: WalletPolicy | None = None,
    allow_failover: bool = True,
) -> NodeRuntimeConfig:
    config = load_or_create_node_config(policy)
    if (
        not config.validator_ha_enabled
        or not config.validator_address
        or config.validator_state != ValidatorLifecycleState.BONDED
    ):
        return config
    resolved_cai_url = _coalesce_cai_url(cai_url, CAI_url)
    if state_payload is None or resolved_cai_url is None:
        return config

    network_status = assess_validator_network_status(
        state_payload=state_payload,
        cai_url=resolved_cai_url,
        policy=policy,
    )
    if not network_status.can_enable or not network_status.current_node_id:
        return config

    validator_id = normalize_address(config.validator_address)
    record = get_validator_record(validator_id, policy)
    active_node_id = str(getattr(record, "active_replica_node_id", "") or "").strip()
    current_node_id = str(network_status.current_node_id or "").strip()
    lease_is_active = validator_ha_lease_is_active(record)
    current_node_holds_lease = bool(active_node_id and active_node_id == current_node_id)

    should_be_active = False
    if config.validator_ha_role == VALIDATOR_HA_ROLE_ACTIVE:
        if lease_is_active and active_node_id and not current_node_holds_lease:
            config.validator_ha_role = VALIDATOR_HA_ROLE_PASSIVE
            save_node_config(config, policy)
            _sync_local_validator_registry(
                config,
                state_payload=state_payload,
                cai_url=resolved_cai_url,
                policy=policy,
            )
            return config
        should_be_active = True
    elif allow_failover and config.validator_ha_auto_failover_enabled:
        should_be_active = (
            current_node_holds_lease
            or not active_node_id
            or not lease_is_active
        )

    if should_be_active:
        if config.validator_ha_role != VALIDATOR_HA_ROLE_ACTIVE:
            config.validator_ha_role = VALIDATOR_HA_ROLE_ACTIVE
            save_node_config(config, policy)
        _sync_local_validator_registry(
            config,
            state_payload=state_payload,
            cai_url=resolved_cai_url,
            active_replica_lease_until=_validator_ha_lease_until(config),
            policy=policy,
        )
        return config

    _sync_local_validator_registry(
        config,
        state_payload=state_payload,
        cai_url=resolved_cai_url,
        policy=policy,
    )
    return config


def get_validator_mode_status(
    *,
    policy: WalletPolicy | None = None,
    money_policy: MoneyPolicy | None = None,
    state_payload: dict | None = None,
    cai_url: str | None = None,
    CAI_url: str | None = None,
) -> ValidatorModeStatus:
    config = load_or_create_node_config(policy)
    active_money_policy = money_policy or MoneyPolicy()
    ensure_chain_genesis(policy=policy, money_policy=active_money_policy)
    required_bond_atomic = coins_to_atomic(
        active_money_policy.validator_min_bond_coins, active_money_policy
    )
    wallet = get_active_wallet(policy)
    wallet_spendable_atomic = (
        wallet_chain_balance_or_local_atomic(wallet, policy) if wallet is not None else 0
    )
    session = load_session(policy)
    network_status = assess_validator_network_status(
        state_payload=state_payload,
        cai_url=_coalesce_cai_url(cai_url, CAI_url),
        policy=policy,
    )

    if config.validator_state == ValidatorLifecycleState.JAILED:
        jail_reason = "Validator is jailed and cannot be re-enabled yet."
        if config.validator_jail_reason:
            jail_reason = f"{jail_reason} Last reason: {config.validator_jail_reason}"
        return ValidatorModeStatus(
            can_enable=False,
            state=config.validator_state,
            reason=jail_reason,
            required_bond_atomic=required_bond_atomic,
            current_spendable_atomic=wallet_spendable_atomic,
            bonded_atomic=config.validator_bond_atomic,
            active_wallet_id=getattr(wallet, "wallet_id", None),
            active_wallet_address=getattr(wallet, "address", None),
            active_wallet_unlocked=bool(
                wallet and session.unlocked_wallet_id == wallet.wallet_id
            ),
            network_ok=False,
            static_ip_confirmed=config.validator_static_ip_confirmed,
            current_node_id=network_status.current_node_id,
            advertised_api_host=network_status.advertised_api_host,
            advertised_data_host=network_status.advertised_data_host,
        )

    if config.validator_state == ValidatorLifecycleState.UNBONDING:
        unbond_reason = "Validator is currently unbonding and cannot be re-enabled yet."
        available_at = _parse_iso_datetime(config.validator_unbonding_available_at)
        current_time = datetime.now(tz=UTC)
        if available_at is not None:
            if current_time < available_at:
                remaining = int((available_at - current_time).total_seconds())
                unbond_reason = (
                    f"{unbond_reason} Bond release becomes available at "
                    f"{available_at.isoformat()} ({max(1, remaining)} second(s) remaining)."
                )
            else:
                unbond_reason = (
                    f"{unbond_reason} Bond release is ready; run validator unbond completion "
                    f"to move back to unbonded state."
                )
        return ValidatorModeStatus(
            can_enable=False,
            state=config.validator_state,
            reason=unbond_reason,
            required_bond_atomic=required_bond_atomic,
            current_spendable_atomic=wallet_spendable_atomic,
            bonded_atomic=config.validator_bond_atomic,
            active_wallet_id=getattr(wallet, "wallet_id", None),
            active_wallet_address=getattr(wallet, "address", None),
            active_wallet_unlocked=bool(
                wallet and session.unlocked_wallet_id == wallet.wallet_id
            ),
            network_ok=False,
            static_ip_confirmed=config.validator_static_ip_confirmed,
            current_node_id=network_status.current_node_id,
            advertised_api_host=network_status.advertised_api_host,
            advertised_data_host=network_status.advertised_data_host,
        )

    if config.validator_state == ValidatorLifecycleState.BONDED:
        validator_wallet = (
            find_wallet_by_id(config.validator_wallet_id, policy)
            if config.validator_wallet_id
            else None
        )
        active_wallet_matches = bool(
            wallet is not None
            and validator_wallet is not None
            and wallet.wallet_id == validator_wallet.wallet_id
        )
        wallet_bond_ok = bool(
            validator_wallet is not None
            and normalize_address(validator_wallet.address)
            == normalize_address(config.validator_address or "")
            and _validator_bond_is_chain_backed(
                wallet=validator_wallet,
                config=config,
                policy=policy,
                money_policy=active_money_policy,
            )
        )
        bond_policy_ok = config.validator_bond_atomic == required_bond_atomic
        can_reuse_bonded_mode = (
            active_wallet_matches
            and wallet_bond_ok
            and bond_policy_ok
            and network_status.can_enable
        )
        if can_reuse_bonded_mode:
            bonded_reason = "Validator mode is already bonded on this node."
        elif validator_wallet is None:
            bonded_reason = "Validator wallet could not be found locally."
        elif not active_wallet_matches:
            bonded_reason = "Select the bonded validator wallet before changing validator mode."
        elif not wallet_bond_ok:
            bonded_reason = "Validator self-bond is not backed by an on-chain bond lock."
        elif not bond_policy_ok:
            bonded_reason = "Validator bond no longer matches the current minimum self-bond policy."
        else:
            bonded_reason = network_status.reason
        return ValidatorModeStatus(
            can_enable=can_reuse_bonded_mode,
            state=config.validator_state,
            reason=bonded_reason,
            required_bond_atomic=required_bond_atomic,
            current_spendable_atomic=wallet_spendable_atomic,
            bonded_atomic=config.validator_bond_atomic,
            active_wallet_id=getattr(wallet, "wallet_id", None),
            active_wallet_address=(
                normalize_address(wallet.address) if wallet is not None else None
            ),
            active_wallet_unlocked=bool(
                wallet and session.unlocked_wallet_id == wallet.wallet_id
            ),
            network_ok=network_status.can_enable,
            static_ip_confirmed=config.validator_static_ip_confirmed,
            current_node_id=network_status.current_node_id,
            advertised_api_host=network_status.advertised_api_host,
            advertised_data_host=network_status.advertised_data_host,
        )

    if config.worker_enabled:
        return ValidatorModeStatus(
            can_enable=False,
            state=config.validator_state,
            reason="Disable worker mode before enabling validator mode.",
            required_bond_atomic=required_bond_atomic,
            current_spendable_atomic=wallet_spendable_atomic,
            bonded_atomic=config.validator_bond_atomic,
            active_wallet_id=getattr(wallet, "wallet_id", None),
            active_wallet_address=getattr(wallet, "address", None),
            active_wallet_unlocked=bool(
                wallet and session.unlocked_wallet_id == wallet.wallet_id
            ),
            network_ok=False,
            static_ip_confirmed=config.validator_static_ip_confirmed,
            current_node_id=network_status.current_node_id,
            advertised_api_host=network_status.advertised_api_host,
            advertised_data_host=network_status.advertised_data_host,
        )

    if wallet is None:
        return ValidatorModeStatus(
            can_enable=False,
            state=config.validator_state,
            reason="Select an active wallet before enabling validator mode.",
            required_bond_atomic=required_bond_atomic,
            current_spendable_atomic=0,
            bonded_atomic=config.validator_bond_atomic,
            active_wallet_id=None,
            active_wallet_address=None,
            active_wallet_unlocked=False,
            network_ok=network_status.can_enable,
            static_ip_confirmed=config.validator_static_ip_confirmed,
            current_node_id=network_status.current_node_id,
            advertised_api_host=network_status.advertised_api_host,
            advertised_data_host=network_status.advertised_data_host,
        )

    if session.unlocked_wallet_id != wallet.wallet_id:
        return ValidatorModeStatus(
            can_enable=False,
            state=config.validator_state,
            reason="Unlock the active wallet before enabling validator mode.",
            required_bond_atomic=required_bond_atomic,
            current_spendable_atomic=wallet_spendable_atomic,
            bonded_atomic=config.validator_bond_atomic,
            active_wallet_id=wallet.wallet_id,
            active_wallet_address=normalize_address(wallet.address),
            active_wallet_unlocked=False,
            network_ok=network_status.can_enable,
            static_ip_confirmed=config.validator_static_ip_confirmed,
            current_node_id=network_status.current_node_id,
            advertised_api_host=network_status.advertised_api_host,
            advertised_data_host=network_status.advertised_data_host,
        )

    if not network_status.can_enable:
        return ValidatorModeStatus(
            can_enable=False,
            state=config.validator_state,
            reason=network_status.reason,
            required_bond_atomic=required_bond_atomic,
            current_spendable_atomic=wallet_spendable_atomic,
            bonded_atomic=config.validator_bond_atomic,
            active_wallet_id=wallet.wallet_id,
            active_wallet_address=normalize_address(wallet.address),
            active_wallet_unlocked=True,
            network_ok=False,
            static_ip_confirmed=config.validator_static_ip_confirmed,
            current_node_id=network_status.current_node_id,
            advertised_api_host=network_status.advertised_api_host,
            advertised_data_host=network_status.advertised_data_host,
        )

    if wallet_spendable_atomic < required_bond_atomic:
        return ValidatorModeStatus(
            can_enable=False,
            state=config.validator_state,
            reason=(
                "Active wallet chain balance is too low for validator mode. "
                f"Minimum self-bond is {active_money_policy.validator_min_bond_coins}."
            ),
            required_bond_atomic=required_bond_atomic,
            current_spendable_atomic=wallet_spendable_atomic,
            bonded_atomic=config.validator_bond_atomic,
            active_wallet_id=wallet.wallet_id,
            active_wallet_address=normalize_address(wallet.address),
            active_wallet_unlocked=True,
            network_ok=True,
            static_ip_confirmed=config.validator_static_ip_confirmed,
            current_node_id=network_status.current_node_id,
            advertised_api_host=network_status.advertised_api_host,
            advertised_data_host=network_status.advertised_data_host,
        )

    return ValidatorModeStatus(
        can_enable=True,
        state=config.validator_state,
        reason="Validator mode is ready to enable on this node.",
        required_bond_atomic=required_bond_atomic,
        current_spendable_atomic=wallet_spendable_atomic,
        bonded_atomic=config.validator_bond_atomic,
        active_wallet_id=wallet.wallet_id,
        active_wallet_address=normalize_address(wallet.address),
        active_wallet_unlocked=True,
        network_ok=True,
        static_ip_confirmed=config.validator_static_ip_confirmed,
        current_node_id=network_status.current_node_id,
        advertised_api_host=network_status.advertised_api_host,
        advertised_data_host=network_status.advertised_data_host,
    )


@dataclass(frozen=True)
class ValidatorNetworkStatus:
    can_enable: bool
    reason: str
    current_node_id: str | None = None
    advertised_api_host: str | None = None
    advertised_data_host: str | None = None


@dataclass(frozen=True)
class ValidatorAttestationStatus:
    can_attest: bool
    reason: str
    validator_id: str | None = None
    passive_replica: bool = False


def assess_validator_network_status(
    *,
    state_payload: dict | None,
    cai_url: str | None,
    CAI_url: str | None = None,
    policy: WalletPolicy | None = None,
) -> ValidatorNetworkStatus:
    config = load_or_create_node_config(policy)
    if state_payload is None:
        return ValidatorNetworkStatus(
            can_enable=False,
            reason="Validator mode requires a reachable local cai state.",
        )

    node_id, identity = _resolve_current_node_identity(
        state_payload,
        _coalesce_cai_url(cai_url, CAI_url),
    )
    if node_id is None or identity is None:
        return ValidatorNetworkStatus(
            can_enable=False,
            reason="Could not identify the current node in cai state.",
        )

    api_host = _normalize_ip_text(identity.get("apiHost"))
    data_host = _normalize_ip_text(identity.get("dataHost"))
    if not _is_public_ip(api_host) or not _is_public_ip(data_host):
        return ValidatorNetworkStatus(
            can_enable=False,
            reason="Validator node must advertise public non-NAT api/data endpoints.",
            current_node_id=node_id,
            advertised_api_host=api_host,
            advertised_data_host=data_host,
        )

    network_info = (state_payload.get("nodeNetwork") or {}).get(node_id) or {}
    interface_ips = {
        _normalize_ip_text(item.get("ipAddress"))
        for item in (network_info.get("interfaces") or [])
        if _normalize_ip_text(item.get("ipAddress"))
    }
    if api_host not in interface_ips or data_host not in interface_ips:
        return ValidatorNetworkStatus(
            can_enable=False,
            reason=(
                "Validator node appears to be behind NAT or a relay: "
                "advertised public api/data host is not assigned to a local interface."
            ),
            current_node_id=node_id,
            advertised_api_host=api_host,
            advertised_data_host=data_host,
        )

    if not config.validator_static_ip_confirmed:
        return ValidatorNetworkStatus(
            can_enable=False,
            reason="Confirm that this validator node uses a static public IP before enabling validator mode.",
            current_node_id=node_id,
            advertised_api_host=api_host,
            advertised_data_host=data_host,
        )

    return ValidatorNetworkStatus(
        can_enable=True,
        reason="Validator network eligibility checks passed.",
        current_node_id=node_id,
        advertised_api_host=api_host,
        advertised_data_host=data_host,
    )


def get_validator_attestation_status(
    *,
    policy: WalletPolicy | None = None,
    state_payload: dict | None = None,
    cai_url: str | None = None,
    CAI_url: str | None = None,
) -> ValidatorAttestationStatus:
    config = load_or_create_node_config(policy)
    if config.validator_state == ValidatorLifecycleState.JAILED:
        return ValidatorAttestationStatus(
            can_attest=False,
            reason="Validator is jailed and cannot attest settlements.",
        )
    if not config.validator_enabled or config.validator_state != ValidatorLifecycleState.BONDED:
        return ValidatorAttestationStatus(
            can_attest=False,
            reason="Validator mode is not bonded on this node.",
        )
    if (
        not config.validator_wallet_id
        or not config.validator_address
        or config.validator_bond_atomic <= 0
    ):
        return ValidatorAttestationStatus(
            can_attest=False,
            reason="Validator bond metadata is incomplete on this node.",
        )

    wallet = find_wallet_by_id(config.validator_wallet_id, policy)
    if wallet is None:
        return ValidatorAttestationStatus(
            can_attest=False,
            reason="Validator wallet could not be found locally.",
        )
    if normalize_address(wallet.address) != normalize_address(config.validator_address):
        return ValidatorAttestationStatus(
            can_attest=False,
            reason="Validator bond is bound to a different wallet address.",
        )
    money_policy = MoneyPolicy(chain_network=(policy or WalletPolicy()).chain_network)
    if not _validator_bond_is_chain_backed(
        wallet=wallet,
        config=config,
        policy=policy,
        money_policy=money_policy,
    ):
        return ValidatorAttestationStatus(
            can_attest=False,
            reason="Validator self-bond is not backed by an on-chain bond lock.",
        )

    network_status = assess_validator_network_status(
        state_payload=state_payload,
        cai_url=_coalesce_cai_url(cai_url, CAI_url),
        policy=policy,
    )
    if not network_status.can_enable:
        return ValidatorAttestationStatus(
            can_attest=False,
            reason=network_status.reason,
        )

    if config.validator_ha_enabled:
        config = refresh_validator_ha_lease(
            state_payload=state_payload,
            cai_url=_coalesce_cai_url(cai_url, CAI_url),
            policy=policy,
            allow_failover=True,
        )
        if config.validator_ha_role != VALIDATOR_HA_ROLE_ACTIVE:
            record = get_validator_record(normalize_address(config.validator_address), policy)
            active_node_id = getattr(record, "active_replica_node_id", None)
            lease_until = getattr(record, "active_replica_lease_until", None)
            lease_note = (
                f" Active replica {active_node_id} holds lease until {lease_until}."
                if active_node_id and lease_until
                else ""
            )
            return ValidatorAttestationStatus(
                can_attest=False,
                reason=(
                    "Validator HA replica is passive and must not attest settlements."
                    f"{lease_note}"
                ),
                validator_id=normalize_address(config.validator_address),
                passive_replica=True,
            )

    return ValidatorAttestationStatus(
        can_attest=True,
        reason="Validator can attest settlements on this node.",
        validator_id=normalize_address(config.validator_address),
    )


def set_validator_static_ip_confirmation(
    confirmed: bool, policy: WalletPolicy | None = None
) -> NodeRuntimeConfig:
    config = load_or_create_node_config(policy)
    sync_validator_id = config.validator_address
    config.validator_static_ip_confirmed = confirmed
    if not confirmed and config.validator_state == ValidatorLifecycleState.BONDED:
        config = _start_validator_unbonding(
            config,
            policy=policy,
            money_policy=MoneyPolicy(),
        )
    save_node_config(config, policy)
    _sync_local_validator_registry(
        config, validator_id_override=sync_validator_id, policy=policy
    )
    return config


def set_worker_mode(
    *,
    enabled: bool,
    allowed_model_ids: list[str] | None = None,
    clear_models: bool = False,
    max_parallel_jobs: int | None = None,
    max_memory_mb: int | None = None,
    policy: WalletPolicy | None = None,
) -> NodeRuntimeConfig:
    config = load_or_create_node_config(policy)
    if enabled and config.validator_state in {
        ValidatorLifecycleState.BONDED,
        ValidatorLifecycleState.UNBONDING,
    }:
        raise ValueError("Disable validator mode and finish unbonding before enabling worker mode.")
    config.worker_enabled = enabled
    if clear_models:
        config.worker_allowed_model_ids = []
    if allowed_model_ids:
        normalized_allowed_model_ids = [
            _canonical_worker_allowed_model_id(model_id.strip())
            for model_id in allowed_model_ids
            if model_id.strip()
        ]
        merged = list(
            dict.fromkeys(
                [*config.worker_allowed_model_ids, *normalized_allowed_model_ids]
            )
        )
        config.worker_allowed_model_ids = merged
    if max_parallel_jobs is not None:
        config.worker_max_parallel_jobs = max_parallel_jobs
    if max_memory_mb is not None:
        config.worker_max_memory_mb = max_memory_mb
    save_node_config(config, policy)
    return config


def set_relay_mode(
    enabled: bool,
    policy: WalletPolicy | None = None,
) -> NodeRuntimeConfig:
    config = load_or_create_node_config(policy)
    config.relay_enabled = bool(enabled)
    config.relay_mode_manually_configured = True
    save_node_config(config, policy)
    return config


def jail_validator(
    *,
    reason: str,
    money_policy: MoneyPolicy | None = None,
    policy: WalletPolicy | None = None,
    slash_bps: int | None = None,
) -> NodeRuntimeConfig:
    config = load_or_create_node_config(policy)
    if config.validator_state == ValidatorLifecycleState.JAILED:
        return config
    if config.validator_state != ValidatorLifecycleState.BONDED or not config.validator_wallet_id:
        raise ValueError("Only bonded validators can be jailed.")

    wallet = find_wallet_by_id(config.validator_wallet_id, policy)
    if wallet is None:
        raise ValueError("Validator wallet could not be found locally.")

    active_money_policy = money_policy or MoneyPolicy()
    if not _validator_bond_is_chain_backed(
        wallet=wallet,
        config=config,
        policy=policy,
        money_policy=active_money_policy,
    ):
        raise ValueError(
            "Cannot jail validator because self-bond is not backed by an on-chain bond lock."
        )
    effective_slash_bps = (
        int(slash_bps)
        if slash_bps is not None
        else int(active_money_policy.validator_jail_slash_bps)
    )
    locked_by_validator = validator_locked_bond_index(policy)
    chain_locked_atomic = max(
        0,
        int(
            locked_by_validator.get(
                normalize_address(config.validator_address or wallet.address),
                0,
            )
            or 0
        ),
    )
    effective_locked_atomic = min(
        chain_locked_atomic,
        max(0, config.validator_bond_atomic),
    )
    slash_atomic = 0
    if effective_locked_atomic > 0:
        slash_atomic = max(1, (effective_locked_atomic * effective_slash_bps) // 10_000)
        slash_atomic = min(effective_locked_atomic, slash_atomic)
    released_atomic = max(0, effective_locked_atomic - slash_atomic)

    _record_validator_bond_slash(
        wallet=wallet,
        config=config,
        slash_atomic=slash_atomic,
        released_atomic=released_atomic,
        policy=policy,
        money_policy=active_money_policy,
        reason=reason,
    )
    wallet.spendable_balance_atomic = chain_balance_atomic(wallet.address, policy)
    wallet.validator_reserved_atomic = max(
        0, wallet.validator_reserved_atomic - effective_locked_atomic
    )
    update_wallet(wallet, policy)

    ledger = load_or_create_ledger(active_money_policy, policy)
    ledger.validator_slashed_atomic += slash_atomic
    save_ledger(ledger, policy)

    jailed_at = _now_iso()
    unjail_available_at = datetime.fromisoformat(jailed_at) + _seconds_delta(
        active_money_policy.validator_unjail_cooldown_seconds
    )
    append_journal_entry(
        JournalEntry(
            entry_id=secrets.token_hex(12),
            event_type="validator_jailed",
            created_at=jailed_at,
            wallet_id=wallet.wallet_id,
            amount_atomic=slash_atomic,
            note=reason,
        ),
        policy,
    )

    config.validator_enabled = False
    config.validator_state = ValidatorLifecycleState.JAILED
    config.validator_bond_atomic = 0
    config.validator_jailed_at = jailed_at
    config.validator_unjail_available_at = unjail_available_at.isoformat()
    config.validator_jail_reason = reason.strip() or "Validator was jailed."
    config.validator_last_slash_atomic = slash_atomic
    config.validator_total_slashed_atomic += slash_atomic
    save_node_config(config, policy)
    _sync_local_validator_registry(config, policy=policy)
    return config


def clear_validator_jail(
    policy: WalletPolicy | None = None,
    *,
    now: datetime | None = None,
) -> NodeRuntimeConfig:
    config = load_or_create_node_config(policy)
    if config.validator_state != ValidatorLifecycleState.JAILED:
        raise ValueError("Validator is not jailed.")
    available_at = _parse_iso_datetime(config.validator_unjail_available_at)
    current_time = now or datetime.now(tz=UTC)
    if available_at is not None and current_time < available_at:
        remaining = int((available_at - current_time).total_seconds())
        raise ValueError(
            "Validator cooldown is still active before unjail. "
            f"Retry in {max(1, remaining)} second(s)."
        )

    sync_validator_id = config.validator_address
    config.validator_enabled = False
    config.validator_state = ValidatorLifecycleState.UNBONDED
    config.validator_wallet_id = None
    config.validator_address = None
    config.validator_bond_atomic = 0
    config.validator_jailed_at = None
    config.validator_unjail_available_at = None
    config.validator_jail_reason = None
    config.validator_last_slash_atomic = 0
    save_node_config(config, policy)
    _sync_local_validator_registry(
        config, validator_id_override=sync_validator_id, policy=policy
    )
    return config


def complete_validator_unbond(
    policy: WalletPolicy | None = None,
    *,
    now: datetime | None = None,
) -> NodeRuntimeConfig:
    config = load_or_create_node_config(policy)
    if config.validator_state != ValidatorLifecycleState.UNBONDING:
        raise ValueError("Validator is not currently unbonding.")
    available_at = _parse_iso_datetime(config.validator_unbonding_available_at)
    current_time = now or datetime.now(tz=UTC)
    if available_at is not None and current_time < available_at:
        remaining = int((available_at - current_time).total_seconds())
        raise ValueError(
            "Validator unbonding period is still active. "
            f"Retry in {max(1, remaining)} second(s)."
        )

    sync_validator_id = config.validator_address
    config = _release_validator_bond(
        config,
        policy=policy,
        note="Validator unbonding completed and self-bond released.",
    )
    save_node_config(config, policy)
    _sync_local_validator_registry(
        config, validator_id_override=sync_validator_id, policy=policy
    )
    return config


def bind_worker_reward_address(
    node_id: str,
    address: str,
    *,
    policy: WalletPolicy | None = None,
) -> NodeRuntimeConfig:
    config = load_or_create_node_config(policy)
    config.worker_reward_address_by_node_id[str(node_id)] = normalize_address(address)
    save_node_config(config, policy)
    return config


def resolve_worker_reward_address(
    node_id: str, policy: WalletPolicy | None = None
) -> str | None:
    config = load_or_create_node_config(policy)
    return config.worker_reward_address_by_node_id.get(str(node_id))


def _release_validator_bond(
    config: NodeRuntimeConfig,
    policy: WalletPolicy | None = None,
    *,
    note: str = "Validator mode disabled and self-bond released.",
) -> NodeRuntimeConfig:
    if config.validator_wallet_id and config.validator_bond_atomic > 0:
        wallet = find_wallet_by_id(config.validator_wallet_id, policy)
        if wallet is not None:
            money_policy = MoneyPolicy(
                chain_network=(policy or WalletPolicy()).chain_network
            )
            chain_backed = _validator_bond_is_chain_backed(
                wallet=wallet,
                config=config,
                policy=policy,
                money_policy=money_policy,
            )
            released_atomic = config.validator_bond_atomic if chain_backed else 0
            if chain_backed:
                _record_validator_bond_release(
                    wallet=wallet,
                    config=config,
                    released_atomic=config.validator_bond_atomic,
                    policy=policy,
                    money_policy=money_policy,
                    note=note,
                )
                wallet.spendable_balance_atomic = chain_balance_atomic(
                    wallet.address,
                    policy,
                )
                wallet.validator_reserved_atomic = max(
                    0,
                    wallet.validator_reserved_atomic - config.validator_bond_atomic,
                )
            else:
                wallet.validator_reserved_atomic = max(
                    0, wallet.validator_reserved_atomic - config.validator_bond_atomic
                )
            update_wallet(wallet, policy)
            append_journal_entry(
                JournalEntry(
                    entry_id=secrets.token_hex(12),
                    event_type="validator_bond_released",
                    created_at=_now_iso(),
                    wallet_id=wallet.wallet_id,
                    amount_atomic=released_atomic,
                    note=(
                        note
                        if chain_backed
                        else "Invalid local validator bond metadata cleared; no on-chain coins were released."
                    ),
                ),
                policy,
            )

    config.validator_enabled = False
    config.validator_state = ValidatorLifecycleState.UNBONDED
    config.validator_wallet_id = None
    config.validator_address = None
    config.validator_bond_atomic = 0
    config.validator_ha_enabled = False
    config.validator_ha_role = VALIDATOR_HA_ROLE_STANDALONE
    config.validator_ha_replica_id = None
    config.validator_unbonding_started_at = None
    config.validator_unbonding_available_at = None
    config.validator_jailed_at = None
    config.validator_unjail_available_at = None
    config.validator_jail_reason = None
    config.validator_last_slash_atomic = 0
    return config


def _start_validator_unbonding(
    config: NodeRuntimeConfig,
    *,
    policy: WalletPolicy | None = None,
    money_policy: MoneyPolicy | None = None,
) -> NodeRuntimeConfig:
    if config.validator_state == ValidatorLifecycleState.UNBONDING:
        return config
    if config.validator_state != ValidatorLifecycleState.BONDED:
        return _release_validator_bond(config, policy)

    active_money_policy = money_policy or MoneyPolicy()
    started_at = _now_iso()
    available_at = datetime.fromisoformat(started_at) + _seconds_delta(
        active_money_policy.validator_unbonding_seconds
    )

    if config.validator_wallet_id and config.validator_bond_atomic > 0:
        wallet = find_wallet_by_id(config.validator_wallet_id, policy)
        if wallet is not None:
            append_journal_entry(
                JournalEntry(
                    entry_id=secrets.token_hex(12),
                    event_type="validator_unbonding_started",
                    created_at=started_at,
                    wallet_id=wallet.wallet_id,
                    amount_atomic=config.validator_bond_atomic,
                    note=(
                        "Validator mode disabled. Self-bond remains locked until "
                        f"{available_at.isoformat()}."
                    ),
                ),
                policy,
            )

    config.validator_enabled = False
    config.validator_state = ValidatorLifecycleState.UNBONDING
    config.validator_unbonding_started_at = started_at
    config.validator_unbonding_available_at = available_at.isoformat()
    config.worker_enabled = False
    return config


def _sync_local_validator_registry(
    config: NodeRuntimeConfig,
    *,
    validator_id_override: str | None = None,
    state_payload: dict | None = None,
    cai_url: str | None = None,
    CAI_url: str | None = None,
    active_replica_lease_until: str | None = None,
    policy: WalletPolicy | None = None,
) -> None:
    validator_id = validator_id_override or config.validator_address
    if not validator_id:
        return

    synced_state = config.validator_state
    synced_bonded_atomic = config.validator_bond_atomic
    if config.validator_state == ValidatorLifecycleState.BONDED:
        wallet = (
            find_wallet_by_id(config.validator_wallet_id, policy)
            if config.validator_wallet_id
            else None
        )
        money_policy = MoneyPolicy(chain_network=(policy or WalletPolicy()).chain_network)
        if wallet is None or not _validator_bond_is_chain_backed(
            wallet=wallet,
            config=config,
            policy=policy,
            money_policy=money_policy,
        ):
            synced_state = ValidatorLifecycleState.UNBONDED
            synced_bonded_atomic = 0

    current_node_id: str | None = None
    api_host: str | None = None
    data_host: str | None = None
    if state_payload is not None:
        current_node_id, identity = _resolve_current_node_identity(
            state_payload,
            _coalesce_cai_url(cai_url, CAI_url),
        )
        if identity is not None:
            api_host = _normalize_ip_text(identity.get("apiHost"))
            data_host = _normalize_ip_text(identity.get("dataHost"))

    sync_validator_record(
        validator_id=normalize_address(validator_id),
        wallet_id=config.validator_wallet_id,
        address=normalize_address(validator_id),
        state=synced_state,
        bonded_atomic=synced_bonded_atomic,
        static_ip_confirmed=config.validator_static_ip_confirmed,
        current_node_id=current_node_id,
        advertised_api_host=api_host,
        advertised_data_host=data_host,
        unbonding_started_at=config.validator_unbonding_started_at,
        unbonding_available_at=config.validator_unbonding_available_at,
        jailed_at=config.validator_jailed_at,
        unjail_available_at=config.validator_unjail_available_at,
        last_slash_atomic=config.validator_last_slash_atomic,
        total_slashed_atomic=config.validator_total_slashed_atomic,
        ha_enabled=config.validator_ha_enabled,
        ha_role=config.validator_ha_role,
        active_replica_node_id=(
            current_node_id
            if config.validator_ha_enabled
            and config.validator_ha_role == VALIDATOR_HA_ROLE_ACTIVE
            else None
        ),
        active_replica_lease_until=(
            active_replica_lease_until
            if config.validator_ha_enabled
            and config.validator_ha_role == VALIDATOR_HA_ROLE_ACTIVE
            else None
        ),
        replica_node_ids=(
            [current_node_id]
            if config.validator_ha_enabled and current_node_id
            else None
        ),
        policy=policy,
    )


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _seconds_delta(seconds: int):
    from datetime import timedelta

    return timedelta(seconds=max(0, int(seconds)))


def _validator_ha_lease_until(config: NodeRuntimeConfig) -> str:
    return (
        datetime.now(tz=UTC)
        + _seconds_delta(
            _normalize_validator_ha_lease_seconds(config.validator_ha_lease_seconds)
        )
    ).isoformat()


def _resolve_current_node_identity(
    state_payload: dict, cai_url: str | None
) -> tuple[str | None, dict | None]:
    identities = state_payload.get("nodeIdentities") or {}
    parsed = urlparse(cai_url or "")
    target_port = parsed.port
    if target_port is not None:
        def _safe_port(value: object) -> int | None:
            try:
                return int(value)  # pyright: ignore[reportArgumentType]
            except (TypeError, ValueError):
                return None

        matches = [
            (node_id, info)
            for node_id, info in identities.items()
            if _safe_port(info.get("apiPort")) == int(target_port)
        ]
        if len(matches) == 1:
            return matches[0]
        resolved_local_node_id = _resolve_local_runtime_node_id(identities)
        if resolved_local_node_id:
            for node_id, info in matches:
                if str(node_id) == resolved_local_node_id:
                    return node_id, info
    if len(identities) == 1:
        return next(iter(identities.items()))
    return None, None


def _resolve_local_runtime_node_id(
    identities: dict[str, dict] | dict,
) -> str | None:
    for host_candidate in (
        os.environ.get("COMPUTERNAME"),
        os.environ.get("HOSTNAME"),
        socket.gethostname(),
    ):
        normalized_host = str(host_candidate or "").strip().lower()
        if not normalized_host:
            continue
        matched_ids = [
            str(node_id)
            for node_id, identity in identities.items()
            if isinstance(identity, dict)
            and str(identity.get("friendlyName", "")).strip().lower()
            == normalized_host
        ]
        if len(matched_ids) == 1:
            return matched_ids[0]

    os_hint = "windows" if os.name == "nt" else "linux"
    os_matches = [
        str(node_id)
        for node_id, identity in identities.items()
        if isinstance(identity, dict)
        and str(identity.get("osVersion", "")).strip().lower() == os_hint
    ]
    if len(os_matches) == 1:
        return os_matches[0]

    return None


def _normalize_ip_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if "%" in normalized:
        normalized = normalized.split("%", 1)[0]
    return normalized or None


def _is_public_ip(value: str | None) -> bool:
    if not value:
        return False
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )
