# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from .economics import (
    AutomaticPriceQuote,
    chain_backed_ledger_snapshot,
    FundingDecision,
    plan_funding,
    quote_automatic_compute_price,
)
from .cai_runtime_launcher import load_peer_book
from .chain import (
    chain_settlement_history,
    chain_summary,
    ensure_chain_genesis,
    wallet_balance_source,
    wallet_chain_balance_or_local_atomic,
)
from .jobs import list_execution_receipts, list_job_intents
from .model import (
    CaiNetworkConfig,
    MoneyPolicy,
    NetworkModelPolicy,
    PaymentPreference,
    WalletPolicy,
)
from .node_config import (
    get_validator_attestation_status,
    get_validator_mode_status,
    load_or_create_node_config,
)
from .settlement import (
    list_attestations,
    list_settlements,
    list_validator_evidence_cases,
    list_validator_evidence,
    list_validator_penalty_cases,
    list_worker_payouts,
)
from .validators import build_validator_committee_snapshot, list_validator_records
from .wallet import (
    LedgerState,
    WalletRecord,
    atomic_to_coins,
    coins_to_atomic,
    get_active_wallet,
    list_journal_entries,
    load_or_create_ledger,
    load_session,
)
from .wallet_signing import (
    SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65,
    SIGNING_SCHEME_ML_DSA_65,
    mldsa65_available,
)


@dataclass(frozen=True)
class WalletPanelState:
    has_active_wallet: bool
    wallet_name: str | None
    address: str | None
    balance_coins: str | None
    balance_source: str
    local_cached_balance_coins: str | None
    unlocked: bool
    history_entries: int


@dataclass(frozen=True)
class NetworkPanelState:
    state_url: str
    reachable: bool
    namespace: str
    bootstrap_peers: int
    peer_book_entries: int
    overlay_peers: int
    topology_nodes: int
    topology_connections: int
    node_system_entries: int
    error: str | None = None


@dataclass(frozen=True)
class ChainPanelState:
    network: str
    block_count: int
    transaction_count: int
    tip_height: int | None
    tip_hash: str | None
    finalized_height: int | None
    last_sync_at: str | None
    valid: bool


@dataclass(frozen=True)
class ValidatorPanelState:
    validator_enabled: bool
    validator_state: str
    validator_address: str | None
    validator_unbonding_started_at: str | None
    validator_unbonding_available_at: str | None
    validator_jailed_at: str | None
    validator_unjail_available_at: str | None
    validator_jail_reason: str | None
    validator_can_enable: bool
    validator_can_attest: bool
    validator_attestation_note: str
    validator_status_note: str
    validator_network_ok: bool
    validator_static_ip_confirmed: bool
    validator_current_node_id: str | None
    validator_advertised_api_host: str | None
    validator_advertised_data_host: str | None
    validator_bond_coins: str
    validator_last_slash_coins: str
    validator_total_slashed_coins: str
    validator_required_bond_coins: str
    active_wallet_spendable_coins: str
    validator_fee_pool_coins: str
    validator_slashed_pool_coins: str
    project_treasury_balance_coins: str
    validator_set_size: int
    validator_bonded_total_coins: str
    validator_quorum_bond_coins: str
    settlement_count: int
    attestation_count: int
    evidence_count: int
    evidence_case_count: int
    evidence_case_pending_quorum_count: int
    evidence_case_finalized_count: int
    evidence_case_applied_count: int
    penalty_case_count: int
    penalty_case_pending_count: int
    penalty_case_pending_attestation_count: int
    penalty_case_finalized_count: int
    penalty_case_applied_count: int
    latest_penalty_case_status: str | None
    latest_penalty_case_scope: str | None
    latest_penalty_case_validator_id: str | None


@dataclass(frozen=True)
class WorkerPanelState:
    worker_enabled: bool
    relay_enabled: bool
    network_default_model_id: str
    network_default_execution_model_id: str
    private_model_minimum_shards: int
    allowed_model_ids: list[str]
    max_parallel_jobs: int
    max_memory_mb: int | None
    reward_bindings: int
    reserve_balance_coins: str
    worker_paid_out_coins: str
    local_worker_earnings_coins: str
    external_payout_records: int
    unbound_payout_records: int
    settlement_records: int
    payout_records: int


@dataclass(frozen=True)
class RewardPanelState:
    payout_records: int
    settlement_records: int
    pending_count: int
    finalized_count: int
    applied_count: int
    unbound_count: int
    pending_coins: str
    finalized_coins: str
    applied_coins: str
    unbound_coins: str
    chain_recorded_count: int
    latest_status: str | None
    latest_settlement_id: str | None
    latest_payout_id: str | None


@dataclass(frozen=True)
class ComputePanelState:
    pricing_mode: str
    payment_preference: str
    quote_available: bool
    quote_reason: str
    funding_source: str | None
    compute_cost_coins: str | None
    tx_fee_coins: str | None
    settlement_fee_coins: str | None
    ai_development_fee_coins: str | None
    worker_reward_coins: str | None
    automatic_price_reason: str | None
    job_intents: int
    execution_receipts: int


@dataclass(frozen=True)
class SecurityPanelState:
    post_quantum_backend: str
    post_quantum_backend_available: bool
    wallet_signing_scheme: str | None
    wallet_address_scheme: str | None
    wallet_pq_signing_scheme: str | None
    wallet_post_quantum_ready: bool
    require_post_quantum_wallet_signatures: bool
    require_hybrid_peer_payload_signatures: bool


@dataclass(frozen=True)
class InterfaceSnapshot:
    wallet: WalletPanelState
    network: NetworkPanelState
    chain: ChainPanelState
    validator: ValidatorPanelState
    worker: WorkerPanelState
    reward: RewardPanelState
    compute: ComputePanelState
    security: SecurityPanelState

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coalesce_cai_url(cai_url: str | None = None, CAI_url: str | None = None) -> str | None:
    resolved = str(cai_url or CAI_url or "").strip()
    return resolved or None


def _chain_or_ledger_coins(
    chain_state: dict[str, Any],
    key: str,
    fallback_atomic: int,
    money_policy: MoneyPolicy,
) -> str:
    if int(chain_state.get("blockCount") or 0) > 0:
        value = chain_state.get(key)
        if isinstance(value, str) and value:
            return value
    return atomic_to_coins(fallback_atomic, money_policy)


def build_interface_snapshot(
    *,
    state_url: str,
    quote_amount_coins: str | None = None,
    quote_prompt: str | None = None,
    quote_model_id: str | None = None,
    cai_url: str | None = None,
    CAI_url: str | None = None,
    payment_preference: PaymentPreference = PaymentPreference.AUTO,
    money_policy: MoneyPolicy | None = None,
    wallet_policy: WalletPolicy | None = None,
    network_config: CaiNetworkConfig | None = None,
    network_model_policy: NetworkModelPolicy | None = None,
) -> InterfaceSnapshot:
    active_money_policy = money_policy or MoneyPolicy()
    active_wallet_policy = wallet_policy or WalletPolicy()
    active_network_config = network_config or CaiNetworkConfig()
    active_model_policy = network_model_policy or NetworkModelPolicy()
    resolved_cai_url = _coalesce_cai_url(cai_url, CAI_url)

    ensure_chain_genesis(policy=active_wallet_policy, money_policy=active_money_policy)
    ledger = chain_backed_ledger_snapshot(
        load_or_create_ledger(
            active_money_policy,
            wallet_policy=active_wallet_policy,
        ),
        money_policy=active_money_policy,
        wallet_policy=active_wallet_policy,
    )
    chain_state = chain_summary(active_wallet_policy)
    wallet = get_active_wallet(active_wallet_policy)
    wallet_panel = _build_wallet_panel(
        wallet,
        active_money_policy,
        active_wallet_policy,
    )
    node_config = load_or_create_node_config(active_wallet_policy)
    state_payload, network_error = _fetch_state_payload(state_url)
    network_panel = _build_network_panel(
        state_url=state_url,
        network_config=active_network_config,
        state_payload=state_payload,
        error=network_error,
    )
    chain_panel = _build_chain_panel(chain_state)
    validator_panel = _build_validator_panel(
        ledger,
        active_money_policy,
        node_config,
        chain_state=chain_state,
        state_payload=state_payload,
        cai_url=resolved_cai_url,
        wallet_policy=active_wallet_policy,
    )
    worker_panel = _build_worker_panel(
        ledger,
        active_money_policy,
        active_model_policy,
        node_config,
        chain_state=chain_state,
        wallet_policy=active_wallet_policy,
    )
    reward_panel = _build_reward_panel(
        active_money_policy,
        wallet_policy=active_wallet_policy,
    )
    compute_panel = _build_compute_panel(
        wallet=wallet,
        ledger=ledger,
        quote_amount_coins=quote_amount_coins,
        quote_prompt=quote_prompt,
        quote_model_id=quote_model_id or active_model_policy.network_default_model_id,
        cai_url=resolved_cai_url,
        payment_preference=payment_preference,
        money_policy=active_money_policy,
        network_model_policy=active_model_policy,
        wallet_policy=active_wallet_policy,
    )
    security_panel = _build_security_panel(wallet, active_wallet_policy)
    return InterfaceSnapshot(
        wallet=wallet_panel,
        network=network_panel,
        chain=chain_panel,
        validator=validator_panel,
        worker=worker_panel,
        reward=reward_panel,
        compute=compute_panel,
        security=security_panel,
    )


def _build_wallet_panel(
    wallet: WalletRecord | None,
    money_policy: MoneyPolicy,
    wallet_policy: WalletPolicy,
) -> WalletPanelState:
    session = load_session(wallet_policy)
    if wallet is None:
        return WalletPanelState(
            has_active_wallet=False,
            wallet_name=None,
            address=None,
            balance_coins=None,
            balance_source=wallet_balance_source(wallet_policy),
            local_cached_balance_coins=None,
            unlocked=False,
            history_entries=0,
        )

    return WalletPanelState(
        has_active_wallet=True,
        wallet_name=wallet.name,
        address=wallet.address,
        balance_coins=atomic_to_coins(
            wallet_chain_balance_or_local_atomic(wallet, wallet_policy),
            money_policy,
        ),
        balance_source=wallet_balance_source(wallet_policy),
        local_cached_balance_coins=atomic_to_coins(
            wallet.spendable_balance_atomic,
            money_policy,
        ),
        unlocked=session.unlocked_wallet_id == wallet.wallet_id,
        history_entries=len(
            list_journal_entries(
                wallet_id=wallet.wallet_id,
                limit=20,
                wallet_policy=wallet_policy,
            )
        ),
    )


def _build_security_panel(
    wallet: WalletRecord | None,
    wallet_policy: WalletPolicy,
) -> SecurityPanelState:
    wallet_pq_ready = bool(
        wallet is not None
        and wallet.signing_scheme == SIGNING_SCHEME_HYBRID_ED25519_ML_DSA_65
        and wallet.pq_signing_scheme == SIGNING_SCHEME_ML_DSA_65
        and wallet.pq_public_key_b64
    )
    return SecurityPanelState(
        post_quantum_backend="ml-dsa-65-v1",
        post_quantum_backend_available=mldsa65_available(),
        wallet_signing_scheme=getattr(wallet, "signing_scheme", None),
        wallet_address_scheme=getattr(wallet, "address_scheme", None),
        wallet_pq_signing_scheme=getattr(wallet, "pq_signing_scheme", None),
        wallet_post_quantum_ready=wallet_pq_ready,
        require_post_quantum_wallet_signatures=bool(
            wallet_policy.require_post_quantum_wallet_signatures
        ),
        require_hybrid_peer_payload_signatures=bool(
            wallet_policy.require_hybrid_peer_payload_signatures
        ),
    )


def _build_network_panel(
    *,
    state_url: str,
    network_config: CaiNetworkConfig,
    state_payload: dict[str, Any] | None,
    error: str | None,
) -> NetworkPanelState:
    if state_payload is None:
        return NetworkPanelState(
            state_url=state_url,
            reachable=False,
            namespace=network_config.namespace,
            bootstrap_peers=len(network_config.bootstrap_peers),
            peer_book_entries=len(load_peer_book()),
            overlay_peers=0,
            topology_nodes=0,
            topology_connections=0,
            node_system_entries=0,
            error=error,
        )

    topology = state_payload.get("topology") or {}
    return NetworkPanelState(
        state_url=state_url,
        reachable=True,
        namespace=network_config.namespace,
        bootstrap_peers=len(network_config.bootstrap_peers),
        peer_book_entries=len(load_peer_book()),
        overlay_peers=len(state_payload.get("overlayPeers") or {}),
        topology_nodes=len(topology.get("nodes") or []),
        topology_connections=len(topology.get("connections") or {}),
        node_system_entries=len(state_payload.get("nodeSystem") or {}),
        error=None,
    )


def _build_chain_panel(chain_state: dict[str, Any]) -> ChainPanelState:
    return ChainPanelState(
        network=str(chain_state.get("network") or ""),
        block_count=int(chain_state.get("blockCount") or 0),
        transaction_count=int(chain_state.get("transactionCount") or 0),
        tip_height=_optional_int(chain_state.get("tipHeight")),
        tip_hash=(
            str(chain_state.get("tipHash"))
            if chain_state.get("tipHash") is not None
            else None
        ),
        finalized_height=_optional_int(chain_state.get("finalizedHeight")),
        last_sync_at=(
            str(chain_state.get("lastSyncAt"))
            if chain_state.get("lastSyncAt") is not None
            else None
        ),
        valid=bool(chain_state.get("valid")),
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_validator_panel(
    ledger: LedgerState,
    money_policy: MoneyPolicy,
    node_config,
    *,
    chain_state: dict[str, Any],
    state_payload: dict[str, Any] | None = None,
    cai_url: str | None = None,
    wallet_policy: WalletPolicy,
) -> ValidatorPanelState:
    status = get_validator_mode_status(
        money_policy=money_policy,
        state_payload=state_payload,
        cai_url=cai_url,
        policy=wallet_policy,
    )
    attestation_status = get_validator_attestation_status(
        policy=wallet_policy,
        state_payload=state_payload,
        cai_url=cai_url,
    )
    committee = build_validator_committee_snapshot(wallet_policy)
    active_wallet = get_active_wallet(wallet_policy)
    active_wallet_spendable_atomic = (
        wallet_chain_balance_or_local_atomic(active_wallet, wallet_policy)
        if active_wallet is not None
        else status.current_spendable_atomic
    )
    evidence_cases = list_validator_evidence_cases(limit=100, policy=wallet_policy)
    penalty_cases = list_validator_penalty_cases(limit=100, policy=wallet_policy)
    evidence_case_pending_quorum_count = sum(
        1 for item in evidence_cases if item.status == "pending"
    )
    evidence_case_finalized_count = sum(
        1 for item in evidence_cases if item.status == "finalized"
    )
    penalty_case_pending_count = sum(
        1 for item in penalty_cases if item.status == "pending"
    )
    penalty_case_pending_attestation_count = sum(
        1
        for item in penalty_cases
        if item.evidence_quorum_reached and not item.quorum_reached
    )
    penalty_case_finalized_count = sum(
        1 for item in penalty_cases if item.status == "finalized"
    )
    latest_penalty_case = penalty_cases[0] if penalty_cases else None
    return ValidatorPanelState(
        validator_enabled=node_config.validator_enabled,
        validator_state=node_config.validator_state,
        validator_address=node_config.validator_address,
        validator_unbonding_started_at=node_config.validator_unbonding_started_at,
        validator_unbonding_available_at=node_config.validator_unbonding_available_at,
        validator_jailed_at=node_config.validator_jailed_at,
        validator_unjail_available_at=node_config.validator_unjail_available_at,
        validator_jail_reason=node_config.validator_jail_reason,
        validator_can_enable=status.can_enable,
        validator_can_attest=attestation_status.can_attest,
        validator_attestation_note=attestation_status.reason,
        validator_status_note=status.reason,
        validator_network_ok=status.network_ok,
        validator_static_ip_confirmed=status.static_ip_confirmed,
        validator_current_node_id=status.current_node_id,
        validator_advertised_api_host=status.advertised_api_host,
        validator_advertised_data_host=status.advertised_data_host,
        validator_bond_coins=atomic_to_coins(
            node_config.validator_bond_atomic, money_policy
        ),
        validator_last_slash_coins=atomic_to_coins(
            node_config.validator_last_slash_atomic, money_policy
        ),
        validator_total_slashed_coins=atomic_to_coins(
            node_config.validator_total_slashed_atomic, money_policy
        ),
        validator_required_bond_coins=atomic_to_coins(
            status.required_bond_atomic, money_policy
        ),
        active_wallet_spendable_coins=atomic_to_coins(
            active_wallet_spendable_atomic, money_policy
        ),
        validator_fee_pool_coins=_chain_or_ledger_coins(
            chain_state,
            "validatorSettlementFeePoolBalanceCoins",
            ledger.validator_fee_pool_atomic,
            money_policy,
        ),
        validator_slashed_pool_coins=_chain_or_ledger_coins(
            chain_state,
            "validatorSlashedCoins",
            ledger.validator_slashed_atomic,
            money_policy,
        ),
        project_treasury_balance_coins=_chain_or_ledger_coins(
            chain_state,
            "developerTreasuryBalanceCoins",
            ledger.project_treasury_balance_atomic,
            money_policy,
        ),
        validator_set_size=len(list_validator_records(wallet_policy)),
        validator_bonded_total_coins=atomic_to_coins(
            committee.total_bonded_atomic, money_policy
        ),
        validator_quorum_bond_coins=atomic_to_coins(
            committee.quorum_bond_atomic, money_policy
        ),
        settlement_count=ledger.settlements_applied,
        attestation_count=len(list_attestations(limit=100, policy=wallet_policy)),
        evidence_count=len(list_validator_evidence(limit=100, policy=wallet_policy)),
        evidence_case_count=len(evidence_cases),
        evidence_case_pending_quorum_count=evidence_case_pending_quorum_count,
        evidence_case_finalized_count=evidence_case_finalized_count,
        evidence_case_applied_count=sum(
            1 for item in evidence_cases if item.status == "applied"
        ),
        penalty_case_count=len(penalty_cases),
        penalty_case_pending_count=penalty_case_pending_count,
        penalty_case_pending_attestation_count=penalty_case_pending_attestation_count,
        penalty_case_finalized_count=penalty_case_finalized_count,
        penalty_case_applied_count=sum(
            1 for item in penalty_cases if item.status == "applied"
        ),
        latest_penalty_case_status=(
            latest_penalty_case.status if latest_penalty_case is not None else None
        ),
        latest_penalty_case_scope=(
            latest_penalty_case.support_scope if latest_penalty_case is not None else None
        ),
        latest_penalty_case_validator_id=(
            latest_penalty_case.validator_id if latest_penalty_case is not None else None
        ),
    )


def _build_worker_panel(
    ledger: LedgerState,
    money_policy: MoneyPolicy,
    network_model_policy: NetworkModelPolicy,
    node_config,
    *,
    chain_state: dict[str, Any],
    wallet_policy: WalletPolicy,
) -> WorkerPanelState:
    payout_records = list_worker_payouts(policy=wallet_policy)
    local_earnings_atomic = sum(
        item.reward_atomic for item in payout_records if item.status == "credited_local_wallet"
    )
    external_payout_records = sum(
        1 for item in payout_records if item.status == "recorded_external_address"
    )
    unbound_payout_records = sum(1 for item in payout_records if item.status == "unbound")
    return WorkerPanelState(
        worker_enabled=node_config.worker_enabled,
        relay_enabled=bool(getattr(node_config, "relay_enabled", False)),
        network_default_model_id=network_model_policy.network_default_model_id,
        network_default_execution_model_id=network_model_policy.network_default_execution_model_id,
        private_model_minimum_shards=network_model_policy.minimum_worker_shards,
        allowed_model_ids=list(node_config.worker_allowed_model_ids),
        max_parallel_jobs=node_config.worker_max_parallel_jobs,
        max_memory_mb=node_config.worker_max_memory_mb,
        reward_bindings=len(node_config.worker_reward_address_by_node_id),
        reserve_balance_coins=_chain_or_ledger_coins(
            chain_state,
            "computeReserveBalanceCoins",
            ledger.compute_reserve_balance_atomic,
            money_policy,
        ),
        worker_paid_out_coins=atomic_to_coins(
            ledger.worker_distributed_atomic, money_policy
        ),
        local_worker_earnings_coins=atomic_to_coins(local_earnings_atomic, money_policy),
        external_payout_records=external_payout_records,
        unbound_payout_records=unbound_payout_records,
        settlement_records=len(list_settlements(policy=wallet_policy)),
        payout_records=len(payout_records),
    )


def _build_reward_panel(
    money_policy: MoneyPolicy,
    *,
    wallet_policy: WalletPolicy,
) -> RewardPanelState:
    payouts = list_worker_payouts(policy=wallet_policy)
    settlements = list_settlements(policy=wallet_policy)
    settlement_by_id = {item.settlement_id: item for item in settlements}
    settlement_ids = sorted({item.settlement_id for item in payouts if item.settlement_id})
    chain_reward_payout_ids: set[str] = set()
    chain_reward_settlement_ids: set[str] = set()
    chain_recorded_count = 0
    reward_tx_types = {"worker_reward_credit", "settlement_worker_reward"}

    for settlement_id in settlement_ids:
        for entry in chain_settlement_history(settlement_id, wallet_policy):
            if entry.get("tx_type") not in reward_tx_types:
                continue
            chain_recorded_count += 1
            payout_id = str(entry.get("payout_id") or "").strip()
            if payout_id:
                chain_reward_payout_ids.add(payout_id)
            else:
                chain_reward_settlement_ids.add(settlement_id)

    count_by_state = {
        "pending": 0,
        "finalized": 0,
        "applied": 0,
        "unbound": 0,
    }
    atomic_by_state = {
        "pending": 0,
        "finalized": 0,
        "applied": 0,
        "unbound": 0,
    }
    latest_status: str | None = None
    latest_settlement_id: str | None = None
    latest_payout_id: str | None = None

    for payout in payouts:
        settlement = settlement_by_id.get(payout.settlement_id)
        state = _reward_state_for_payout(
            payout,
            settlement,
            chain_reward_payout_ids=chain_reward_payout_ids,
            chain_reward_settlement_ids=chain_reward_settlement_ids,
        )
        if latest_status is None:
            latest_status = state
            latest_settlement_id = payout.settlement_id
            latest_payout_id = payout.payout_id
        count_by_state[state] += 1
        atomic_by_state[state] += max(0, int(payout.reward_atomic or 0))

    return RewardPanelState(
        payout_records=len(payouts),
        settlement_records=len(settlements),
        pending_count=count_by_state["pending"],
        finalized_count=count_by_state["finalized"],
        applied_count=count_by_state["applied"],
        unbound_count=count_by_state["unbound"],
        pending_coins=atomic_to_coins(atomic_by_state["pending"], money_policy),
        finalized_coins=atomic_to_coins(atomic_by_state["finalized"], money_policy),
        applied_coins=atomic_to_coins(atomic_by_state["applied"], money_policy),
        unbound_coins=atomic_to_coins(atomic_by_state["unbound"], money_policy),
        chain_recorded_count=chain_recorded_count,
        latest_status=latest_status,
        latest_settlement_id=latest_settlement_id,
        latest_payout_id=latest_payout_id,
    )


def _reward_state_for_payout(
    payout,
    settlement,
    *,
    chain_reward_payout_ids: set[str],
    chain_reward_settlement_ids: set[str],
) -> str:
    payout_id = str(getattr(payout, "payout_id", "") or "").strip()
    settlement_id = str(getattr(payout, "settlement_id", "") or "").strip()
    if payout_id and payout_id in chain_reward_payout_ids:
        return "applied"
    if settlement_id and settlement_id in chain_reward_settlement_ids:
        return "applied"
    if not getattr(payout, "recipient_address", None):
        return "unbound"

    settlement_status = str(getattr(settlement, "status", "") or "").strip()
    if settlement_status in {"finalized", "applied"} or getattr(
        settlement,
        "applied_at",
        None,
    ):
        return "finalized"
    return "pending"


def _build_compute_panel(
    *,
    wallet: WalletRecord | None,
    ledger: LedgerState,
    quote_amount_coins: str | None,
    quote_prompt: str | None,
    quote_model_id: str,
    cai_url: str | None,
    payment_preference: PaymentPreference,
    money_policy: MoneyPolicy,
    network_model_policy: NetworkModelPolicy,
    wallet_policy: WalletPolicy,
) -> ComputePanelState:
    if wallet is None or (quote_amount_coins is None and not quote_prompt):
        return ComputePanelState(
            pricing_mode="none",
            payment_preference=payment_preference.value,
            quote_available=False,
            quote_reason="Provide an active wallet plus either a quote amount or a quote prompt to build a compute preview.",
            funding_source=None,
            compute_cost_coins=None,
            tx_fee_coins=None,
            settlement_fee_coins=None,
            ai_development_fee_coins=None,
            worker_reward_coins=None,
            automatic_price_reason=None,
            job_intents=len(list_job_intents(wallet_policy)),
            execution_receipts=len(list_execution_receipts(wallet_policy)),
        )

    automatic_quote: AutomaticPriceQuote | None = None
    resolved_quote_amount_coins = quote_amount_coins
    pricing_mode = "manual"
    pricing_reason: str | None = None
    if resolved_quote_amount_coins is None and quote_prompt:
        automatic_quote = quote_automatic_compute_price(
            prompt=quote_prompt,
            model_id=quote_model_id,
            ledger=ledger,
            cai_url=cai_url,
            money_policy=money_policy,
            network_model_policy=network_model_policy,
        )
        resolved_quote_amount_coins = atomic_to_coins(
            automatic_quote.compute_cost_atomic, money_policy
        )
        pricing_mode = automatic_quote.pricing_mode
        pricing_reason = automatic_quote.reason

    wallet_for_decision = replace(
        wallet,
        spendable_balance_atomic=wallet_chain_balance_or_local_atomic(
            wallet,
            wallet_policy,
        ),
    )
    decision = plan_funding(
        ledger=ledger,
        wallet=wallet_for_decision,
        compute_cost_atomic=coins_to_atomic(resolved_quote_amount_coins, money_policy),
        payment_preference=payment_preference,
        money_policy=money_policy,
    )
    return _decision_to_compute_panel(
        decision,
        money_policy,
        payment_preference,
        pricing_mode=pricing_mode,
        pricing_reason=pricing_reason,
        wallet_policy=wallet_policy,
    )


def _decision_to_compute_panel(
    decision: FundingDecision,
    money_policy: MoneyPolicy,
    payment_preference: PaymentPreference,
    *,
    pricing_mode: str,
    pricing_reason: str | None,
    wallet_policy: WalletPolicy,
) -> ComputePanelState:
    return ComputePanelState(
        pricing_mode=pricing_mode,
        payment_preference=payment_preference.value,
        quote_available=decision.can_fund,
        quote_reason=decision.reason,
        funding_source=decision.funding_source.value if decision.funding_source else None,
        compute_cost_coins=atomic_to_coins(
            decision.fee_quote.compute_cost_atomic, money_policy
        ),
        tx_fee_coins=atomic_to_coins(decision.fee_quote.tx_fee_atomic, money_policy),
        settlement_fee_coins=atomic_to_coins(
            decision.fee_quote.settlement_fee_atomic, money_policy
        ),
        ai_development_fee_coins=atomic_to_coins(
            decision.fee_quote.ai_development_fee_atomic, money_policy
        ),
        worker_reward_coins=atomic_to_coins(
            decision.fee_quote.worker_reward_atomic, money_policy
        ),
        automatic_price_reason=pricing_reason,
        job_intents=len(list_job_intents(wallet_policy)),
        execution_receipts=len(list_execution_receipts(wallet_policy)),
    )


def _fetch_state_payload(state_url: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with urlopen(state_url, timeout=5) as response:
            return json.loads(response.read().decode("utf-8")), None
    except (TimeoutError, URLError, OSError, json.JSONDecodeError) as exc:
        return None, str(exc)


