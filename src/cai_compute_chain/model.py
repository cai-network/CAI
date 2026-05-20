# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from .gguf_shard_policy import (
    GGUF_SHARD_MODE_UNSUPPORTED_FOR_SHARDING,
    gguf_shard_compatibility,
)


class NodeRole(StrEnum):
    SEED = "seed"
    PEER = "peer"
    VALIDATOR = "validator"
    WORKER = "worker"
    CLIENT = "client"


class ValidatorLifecycleState(StrEnum):
    UNBONDED = "unbonded"
    BONDED = "bonded"
    UNBONDING = "unbonding"
    JAILED = "jailed"


class VerificationMode(StrEnum):
    DETERMINISTIC = "deterministic"
    REDUNDANT = "redundant"
    CHALLENGE = "challenge"
    RECEIPT_ONLY = "receipt_only"


class PaymentPreference(StrEnum):
    AUTO = "auto"
    RESERVE_ONLY = "reserve_only"
    WALLET_ONLY = "wallet_only"


class FundingSource(StrEnum):
    RESERVE = "reserve"
    WALLET = "wallet"


class NetworkModelMode(StrEnum):
    PUBLIC_SHARED = "public_shared"
    PRIVATE_NETWORK_CURATED = "private_network_curated"


class ChainNetwork(StrEnum):
    MAINNET = "mainnet"
    TESTNET = "testnet"


LEGACY_PRIVATE_NETWORK_MODEL_ID = "cai-network/qwen3-0.6b-4bit"
DEFAULT_PRIVATE_NETWORK_MODEL_ID = "cai-network/Qwen3-0.6B-GGUF"
DEFAULT_PRIVATE_EXECUTION_MODEL_ID = "Qwen/Qwen3-0.6B-GGUF"


_CHAIN_NETWORK_ALIASES: dict[str, ChainNetwork] = {
    "mainnet": ChainNetwork.MAINNET,
    "main": ChainNetwork.MAINNET,
    "prod": ChainNetwork.MAINNET,
    "production": ChainNetwork.MAINNET,
    "testnet": ChainNetwork.TESTNET,
    "test": ChainNetwork.TESTNET,
    "staging": ChainNetwork.TESTNET,
}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_flag_or_none(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on", "strict", "required"}


def default_require_post_quantum_wallet_signatures() -> bool:
    configured = _env_flag_or_none("CAI_REQUIRE_POST_QUANTUM_SIGNATURES")
    if configured is not None:
        return configured
    return resolve_active_chain_network() == ChainNetwork.MAINNET


def default_require_hybrid_peer_payload_signatures() -> bool:
    configured = _env_flag_or_none("CAI_REQUIRE_HYBRID_PEER_PAYLOAD_SIGNATURES")
    if configured is not None:
        return configured
    return resolve_active_chain_network() == ChainNetwork.MAINNET


@dataclass(frozen=True)
class ChainNetworkPreset:
    namespace: str
    bootstrap_peers: tuple[str, ...]
    trusted_validator_addresses: tuple[str, ...]
    default_api_port: int
    default_libp2p_port: int
    default_cai_home_dirname: str
    wallet_data_dirname: str
    developer_treasury_wallet_id: str
    developer_treasury_address: str
    ai_development_wallet_id: str
    ai_development_address: str


MAINNET_BOOTSTRAP_PEERS: tuple[str, ...] = (
    # Stable public bootstrap/relay/validator endpoints can be added here by PR.
    "/ip4/192.145.29.212/tcp/52416",
)

MAINNET_TRUSTED_VALIDATOR_ADDRESSES: tuple[str, ...] = (
    # Initial validator root used only to authenticate mainnet chain sync before
    # a node has a local chain-backed validator set.
    "2dabc2bfc0e6182aade1e5c17633d191fc290e3bd94b6153ab0992c79078a8ec",
)


CHAIN_NETWORK_PRESETS: dict[ChainNetwork, ChainNetworkPreset] = {
    ChainNetwork.MAINNET: ChainNetworkPreset(
        namespace="cai-ai-net",
        bootstrap_peers=MAINNET_BOOTSTRAP_PEERS,
        trusted_validator_addresses=MAINNET_TRUSTED_VALIDATOR_ADDRESSES,
        default_api_port=52415,
        default_libp2p_port=52418,
        default_cai_home_dirname=".cai",
        wallet_data_dirname=".cai-local",
        developer_treasury_wallet_id="f566089781403edca18c2d06c9c0af8a",
        developer_treasury_address=(
            "6d9c5fc0ab4f5ad786881d1848800f778dc8f21473ebcff514181dfb50023881"
        ),
        ai_development_wallet_id="d614b61241fa972c34cf7c256848dc3d",
        ai_development_address="d7fbe14450f1a91363fb9585393ccd38",
    ),
    ChainNetwork.TESTNET: ChainNetworkPreset(
        namespace="cai-ai-testnet",
        bootstrap_peers=(),
        trusted_validator_addresses=(),
        default_api_port=52515,
        default_libp2p_port=52518,
        default_cai_home_dirname=".cai-testnet",
        wallet_data_dirname=".cai-local-testnet",
        developer_treasury_wallet_id="855ecd31ec15064d9787560bc0e44ccc",
        developer_treasury_address="c45112f1eb695a79732dc622e16e0423",
        ai_development_wallet_id="4b41845b6ba0422fdee9a2dc435101d1",
        ai_development_address="d96d56a8fd63fb352009bf17d05c7d96",
    ),
}


@dataclass(frozen=True)
class CuratedNetworkModel:
    model_id: str
    execution_model_id: str
    display_name: str
    source_repo_id: str
    preferred_filename: str
    runtime_model_ids: tuple[str, ...]
    total_layers: int | None = None
    hidden_size: int | None = None
    private_network: bool = True
    minimum_worker_shards: int = 1
    allow_single_node_fallback: bool = False
    model_format: str = "gguf"
    gguf_architecture: str = "unknown"
    shard_compatibility: str = GGUF_SHARD_MODE_UNSUPPORTED_FOR_SHARDING
    layer_range_supported: bool = False
    layer_range_probe_abi: str | None = None
    layer_range_probe_report: str | None = None
    layer_range_equivalence_probe_report: str | None = None
    state_format: str | None = None
    activation_state_format: str | None = None
    decode_state_format: str | None = None


def _curated_gguf_policy_fields(
    *,
    model_id: str,
    gguf_architecture: str,
    filename: str,
) -> dict[str, object]:
    compatibility = gguf_shard_compatibility(
        model_id=model_id,
        gguf_architecture=gguf_architecture,
        family=gguf_architecture,
        filename=filename,
    )
    metadata = compatibility.to_metadata()
    return {
        key: value
        for key, value in metadata.items()
        if key
        in {
            "model_format",
            "gguf_architecture",
            "shard_compatibility",
            "layer_range_supported",
            "layer_range_probe_abi",
            "layer_range_probe_report",
            "layer_range_equivalence_probe_report",
            "state_format",
            "activation_state_format",
            "decode_state_format",
        }
    }


CURATED_NETWORK_MODELS: tuple[CuratedNetworkModel, ...] = (
    CuratedNetworkModel(
        model_id=DEFAULT_PRIVATE_NETWORK_MODEL_ID,
        execution_model_id=DEFAULT_PRIVATE_EXECUTION_MODEL_ID,
        display_name="Qwen3 0.6B GGUF",
        source_repo_id=DEFAULT_PRIVATE_NETWORK_MODEL_ID,
        preferred_filename="Qwen3-0.6B-Q8_0.gguf",
        total_layers=28,
        hidden_size=1024,
        runtime_model_ids=(
            DEFAULT_PRIVATE_NETWORK_MODEL_ID,
        ),
        private_network=True,
        minimum_worker_shards=2,
        allow_single_node_fallback=False,
        **_curated_gguf_policy_fields(
            model_id=DEFAULT_PRIVATE_NETWORK_MODEL_ID,
            gguf_architecture="qwen3",
            filename="Qwen3-0.6B-Q8_0.gguf",
        ),
    ),
    CuratedNetworkModel(
        model_id="Qwen/Qwen3-0.6B-GGUF",
        execution_model_id="Qwen/Qwen3-0.6B-GGUF",
        display_name="Qwen3 0.6B GGUF",
        source_repo_id="Qwen/Qwen3-0.6B-GGUF",
        preferred_filename="Qwen3-0.6B-Q8_0.gguf",
        total_layers=28,
        hidden_size=1024,
        runtime_model_ids=("Qwen/Qwen3-0.6B-GGUF",),
        private_network=False,
        minimum_worker_shards=1,
        allow_single_node_fallback=False,
        **_curated_gguf_policy_fields(
            model_id="Qwen/Qwen3-0.6B-GGUF",
            gguf_architecture="qwen3",
            filename="Qwen3-0.6B-Q8_0.gguf",
        ),
    ),
    CuratedNetworkModel(
        model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        execution_model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        display_name="Qwen2.5 0.5B Instruct GGUF",
        source_repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        preferred_filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        total_layers=24,
        hidden_size=896,
        runtime_model_ids=("Qwen/Qwen2.5-0.5B-Instruct-GGUF",),
        private_network=False,
        minimum_worker_shards=1,
        allow_single_node_fallback=False,
        **_curated_gguf_policy_fields(
            model_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            gguf_architecture="qwen2",
            filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        ),
    ),
)

DEFAULT_PRIVATE_CURATED_MODEL_IDS = tuple(
    model.model_id for model in CURATED_NETWORK_MODELS if model.private_network
)
DEFAULT_PRIVATE_RUNTIME_MODEL_IDS = tuple(
    dict.fromkeys(
        runtime_model_id
        for model in CURATED_NETWORK_MODELS
        if model.private_network
        for runtime_model_id in model.runtime_model_ids
    )
)
DEFAULT_NETWORK_EXECUTION_MODEL_IDS = tuple(
    dict.fromkeys(model.execution_model_id for model in CURATED_NETWORK_MODELS)
)


def resolve_active_chain_network(value: str | ChainNetwork | None = None) -> ChainNetwork:
    if isinstance(value, ChainNetwork):
        return value
    candidates = (
        str(value).strip().lower()
        if value is not None
        else str(os.getenv("CAI_CHAIN_NETWORK") or os.getenv("CAI_NETWORK") or "").strip().lower()
    )
    return _CHAIN_NETWORK_ALIASES.get(candidates, ChainNetwork.MAINNET)


def active_chain_preset(network: str | ChainNetwork | None = None) -> ChainNetworkPreset:
    return CHAIN_NETWORK_PRESETS[resolve_active_chain_network(network)]


def _env_or_preset(name: str, value: str) -> str:
    configured = str(os.getenv(name) or "").strip()
    return configured or value


def default_wallet_data_dirname() -> str:
    return active_chain_preset().wallet_data_dirname


def default_cai_namespace() -> str:
    return active_chain_preset().namespace


def default_bootstrap_peers() -> tuple[str, ...]:
    configured_peers = _split_peer_env(
        os.getenv("CAI_BOOTSTRAP_PEERS") or os.getenv("EXO_BOOTSTRAP_PEERS")
    )
    extra_peers = _split_peer_env(os.getenv("CAI_EXTRA_BOOTSTRAP_PEERS"))
    base_peers = configured_peers or list(active_chain_preset().bootstrap_peers)
    return tuple(dict.fromkeys([*base_peers, *extra_peers]))


def default_trusted_validator_addresses() -> tuple[str, ...]:
    configured = _split_peer_env(os.getenv("CAI_TRUSTED_VALIDATOR_ADDRESSES"))
    extra = _split_peer_env(os.getenv("CAI_EXTRA_TRUSTED_VALIDATOR_ADDRESSES"))
    base = configured or list(active_chain_preset().trusted_validator_addresses)
    return tuple(dict.fromkeys([*base, *extra]))


def _split_peer_env(raw: str | None) -> list[str]:
    if raw is None:
        return []
    peers: list[str] = []
    for chunk in raw.replace("\n", ",").split(","):
        peer = chunk.strip()
        if peer:
            peers.append(peer)
    return peers


def default_api_port() -> int:
    return active_chain_preset().default_api_port


def default_libp2p_port() -> int:
    return active_chain_preset().default_libp2p_port


def default_cai_home_dirname() -> str:
    return active_chain_preset().default_cai_home_dirname


def default_developer_treasury_wallet_id() -> str:
    return _env_or_preset(
        "CAI_DEVELOPER_TREASURY_WALLET_ID",
        active_chain_preset().developer_treasury_wallet_id,
    )


def default_developer_treasury_address() -> str:
    return _env_or_preset(
        "CAI_DEVELOPER_TREASURY_ADDRESS",
        active_chain_preset().developer_treasury_address,
    )


def default_ai_development_wallet_id() -> str:
    return active_chain_preset().ai_development_wallet_id


def default_ai_development_address() -> str:
    return active_chain_preset().ai_development_address


@dataclass(frozen=True)
class NetworkPrinciples:
    uses_cai_runtime_base: bool = True
    minimize_extra_protocol_logic: bool = True
    has_validator_role: bool = True
    has_dedicated_centers: bool = False
    has_claim_protocol: bool = False
    worker_payments_settled_on_chain: bool = True
    reserve_can_fund_compute: bool = True
    tx_fee_paid_by_initiator: bool = True


@dataclass(frozen=True)
class RoadmapStatus:
    execution_base: str
    currency_layer: str
    validator_consensus: str
    settlement_layer: str


@dataclass(frozen=True)
class MoneyPolicy:
    chain_network: ChainNetwork = field(default_factory=resolve_active_chain_network)
    currency_code: str = "CAICN"
    currency_name: str = "CAI Network Credit"
    reward_token_code: str = "CAICN"
    decimals: int = 8
    total_supply_coins: int = 1_000_000_000
    compute_reserve_share: float = 0.85
    compute_reserve_coins: int = 850_000_000
    project_treasury_share: float = 0.05
    project_treasury_coins: int = 50_000_000
    developer_treasury_share: float = 0.05
    developer_treasury_coins: int = 50_000_000
    developer_contribution_fund_share: float = 0.10
    developer_contribution_fund_coins: int = 100_000_000
    developer_treasury_wallet_id: str = field(
        default_factory=default_developer_treasury_wallet_id
    )
    developer_treasury_address: str = field(
        default_factory=default_developer_treasury_address
    )
    ai_development_wallet_id: str = field(
        default_factory=default_ai_development_wallet_id
    )
    ai_development_address: str = field(
        default_factory=default_ai_development_address
    )
    ai_development_fee_bps: int = 0
    daily_user_reserve_limit_enabled: bool = True
    daily_user_reserve_limit_coins: str = "1.00000000"
    daily_ip_reserve_limit_enabled: bool = True
    daily_ip_reserve_limit_coins: str = "1.00000000"
    reserve_funds_compute_first: bool = True
    user_can_fund_jobs_from_wallet: bool = True
    tx_fee_paid_by_initiator: bool = True
    default_tx_fee_coins: str = "0.00010000"
    validator_settlement_fee_bps: int = 200
    validator_min_bond_coins: str = "10000.00000000"
    validator_committee_target_size: int = 3
    validator_committee_selection_mode: str = "stake_weighted_lottery"
    validator_jail_slash_bps: int = 500
    validator_conflicting_attestation_slash_bps: int = 2000
    validator_unbonding_seconds: int = 86400
    validator_unjail_cooldown_seconds: int = 21600
    automatic_pricing_enabled: bool = True
    automatic_token_pricing_enabled: bool = True
    automatic_price_floor_coins: str = "0.00100000"
    automatic_price_cap_coins: str = "0.00500000"
    automatic_price_per_input_token_coins: str = "0.00000300"
    automatic_price_per_output_token_coins: str = "0.00000600"
    automatic_price_default_reserved_output_tokens: int = 128
    automatic_price_prompt_unit_chars: int = 256
    automatic_price_per_prompt_unit_coins: str = "0.00025000"
    automatic_price_target_connections: int = 2
    automatic_price_low_load_threshold: float = 0.35
    automatic_price_high_load_threshold: float = 0.85
    automatic_price_low_load_discount_bps: int = 500
    automatic_price_high_load_surcharge_bps: int = 1500
    automatic_price_connection_scarcity_surcharge_bps: int = 1200
    automatic_price_healthy_network_discount_bps: int = 500
    automatic_price_unreachable_safety_bps: int = 500
    automatic_price_non_default_model_premium_bps: int = 2500
    automatic_price_reserve_guard_bps: int = 500
    automatic_price_reserve_critical_bps: int = 1000


@dataclass(frozen=True)
class WalletPolicy:
    chain_network: ChainNetwork = field(default_factory=resolve_active_chain_network)
    requires_password: bool = True
    require_post_quantum_wallet_signatures: bool | None = None
    require_hybrid_peer_payload_signatures: bool | None = None
    trusted_validator_addresses: tuple[str, ...] = field(
        default_factory=default_trusted_validator_addresses
    )
    wallet_data_dirname: str = field(default_factory=default_wallet_data_dirname)
    wallet_file_name: str = "wallets.json"
    session_file_name: str = "session.json"
    ledger_file_name: str = "ledger.json"
    journal_file_name: str = "journal.jsonl"
    node_config_file_name: str = "node-config.json"
    settlement_file_name: str = "settlements.json"
    attestation_file_name: str = "attestations.jsonl"
    validator_evidence_file_name: str = "validator-evidence.jsonl"
    validator_penalty_case_file_name: str = "validator-penalty-cases.json"
    validator_penalty_attestation_file_name: str = "validator-penalty-attestations.jsonl"
    worker_capability_attestation_file_name: str = "worker-capability-attestations.jsonl"
    validator_set_file_name: str = "validators.json"
    chain_file_name: str = "chain.json"
    worker_payout_file_name: str = "worker-payouts.json"
    job_intent_file_name: str = "job-intents.json"
    execution_receipt_file_name: str = "execution-receipts.json"
    secret_dir_name: str = "secrets"
    developer_treasury_seed_file_name: str = "developer-treasury-seed.txt"
    developer_treasury_password_file_name: str = "developer-treasury-password.txt"
    ai_development_seed_file_name: str = "ai-development-seed.txt"
    ai_development_password_file_name: str = "ai-development-password.txt"
    password_kdf_rounds: int = 200_000

    def __post_init__(self) -> None:
        if self.require_post_quantum_wallet_signatures is None:
            configured = _env_flag_or_none("CAI_REQUIRE_POST_QUANTUM_SIGNATURES")
            object.__setattr__(
                self,
                "require_post_quantum_wallet_signatures",
                (
                    configured
                    if configured is not None
                    else self.chain_network == ChainNetwork.MAINNET
                ),
            )
        if self.require_hybrid_peer_payload_signatures is None:
            configured = _env_flag_or_none("CAI_REQUIRE_HYBRID_PEER_PAYLOAD_SIGNATURES")
            object.__setattr__(
                self,
                "require_hybrid_peer_payload_signatures",
                configured
                if configured is not None
                else default_require_hybrid_peer_payload_signatures(),
            )


@dataclass(frozen=True)
class ChunkCachePolicy:
    max_store_bytes: int = 8 * 1024 * 1024 * 1024
    target_store_bytes: int = 6 * 1024 * 1024 * 1024
    evict_expired_leases_first: bool = True
    protect_active_leases: bool = True
    protect_pinned_chunks: bool = True
    protect_hot_chunks_when_possible: bool = True
    assignment_lease_seconds: int = 3600
    pin_assignment_chunks: bool = False


@dataclass(frozen=True)
class ChunkFetchPolicy:
    prefer_healthy_sources: bool = True
    max_inventory_age_seconds: int = 600
    warm_prefetch_weight_chunk_count_per_manifest: int = 1
    warm_prefetch_max_weight_bytes_per_manifest: int = 268_435_456
    hint_prefetch_weight_chunk_count_per_manifest: int = 2
    hint_prefetch_max_weight_bytes_per_manifest: int = 536_870_912
    recent_shard_hint_ttl_seconds: int = 3600
    recent_shard_hint_capacity_per_node: int = 16
    recent_hint_prefetch_max_hints: int = 4
    source_failure_cooldown_seconds: int = 60
    source_failure_backoff_multiplier: float = 2.0
    max_source_failure_cooldown_seconds: int = 900
    reset_failures_on_success: bool = True
    skip_cooldowned_sources_when_alternatives_exist: bool = True


@dataclass(frozen=True)
class ChunkStorageAccountingPolicy:
    max_accounting_interval_seconds: int = 3600
    min_accounting_seconds: int = 1


@dataclass(frozen=True)
class CaiNetworkConfig:
    chain_network: ChainNetwork = field(default_factory=resolve_active_chain_network)
    namespace: str = field(default_factory=default_cai_namespace)
    overlay_first: bool = True
    direct_inbound_required: bool = False
    bootstrap_peers: tuple[str, ...] = field(default_factory=default_bootstrap_peers)
    default_api_port: int = field(default_factory=default_api_port)
    default_libp2p_port: int = field(default_factory=default_libp2p_port)
    default_cai_home_dirname: str = field(default_factory=default_cai_home_dirname)
    advertise_env_var_name: str = "CAI_ADVERTISE_PEERS"


@dataclass(frozen=True)
class CaiLaunchPlan:
    executable_candidates: tuple[Path, ...]
    env_var_name: str = "CAI_RUNTIME_EXECUTABLE"


@dataclass(frozen=True)
class NetworkModelPolicy:
    mode: NetworkModelMode = NetworkModelMode.PRIVATE_NETWORK_CURATED
    network_default_model_id: str = DEFAULT_PRIVATE_NETWORK_MODEL_ID
    network_default_execution_model_id: str = DEFAULT_PRIVATE_EXECUTION_MODEL_ID
    private_curated_model_ids: tuple[str, ...] = DEFAULT_PRIVATE_CURATED_MODEL_IDS
    private_runtime_model_ids: tuple[str, ...] = DEFAULT_PRIVATE_RUNTIME_MODEL_IDS
    network_execution_model_ids: tuple[str, ...] = DEFAULT_NETWORK_EXECUTION_MODEL_IDS
    user_must_download_model: bool = False
    client_receives_weights: bool = False
    private_model_allows_full_single_worker_copy: bool = False
    private_model_requires_sharded_distribution: bool = True
    minimum_worker_shards: int = 2
    minimum_worker_ram_headroom_mb: int = 256
    minimum_worker_vram_headroom_mb: int = 0
    minimum_worker_pipeline_layers_per_node: int = 2
    allow_single_node_private_inference: bool = field(
        default_factory=lambda: _env_flag(
            "CAI_PRIVATE_NETWORK_MODEL_ALLOW_BOOTSTRAP_SINGLE_NODE", True
        )
    )
    shard_cache_encrypted_at_rest: bool = True
    shard_keys_released_per_assignment: bool = True
    model_owner_can_publish_public_model: bool = True
    absolute_leak_protection_on_untrusted_hardware: bool = False
    v1_delivery_strategy: str = "client-zero-download + worker-shard-cache"


def effective_private_worker_shard_minimum(
    policy: NetworkModelPolicy | None = None,
    *,
    available_worker_count: int | None = None,
) -> int:
    active_policy = policy or NetworkModelPolicy()
    minimum_worker_shards = max(1, int(active_policy.minimum_worker_shards))
    if (
        active_policy.allow_single_node_private_inference
        and available_worker_count is not None
        and int(available_worker_count) > 0
    ):
        return 1
    return minimum_worker_shards


@dataclass(frozen=True)
class ModelPrivacyPrinciples:
    no_client_weight_distribution: bool = True
    no_full_model_distribution_to_untrusted_workers: bool = True
    overlay_network_model_access: bool = True
    validator_gated_shard_assignment: bool = True
    per_shard_integrity_verification: bool = True
    per_shard_decryption_keys: bool = True
    confidential_compute_required_for_strongest_protection: bool = True


def resolve_execution_model_id(
    model_id: str, policy: NetworkModelPolicy | None = None
) -> str:
    active_policy = policy or NetworkModelPolicy()
    normalized_model_id = normalize_network_model_id(model_id, active_policy)
    curated_model = curated_model_for_id(normalized_model_id)
    if curated_model is not None and normalized_model_id == curated_model.model_id:
        return curated_model.execution_model_id
    if normalized_model_id == active_policy.network_default_model_id:
        return active_policy.network_default_execution_model_id
    return normalized_model_id


def normalize_network_model_id(
    model_id: str, policy: NetworkModelPolicy | None = None
) -> str:
    active_policy = policy or NetworkModelPolicy()
    if model_id == LEGACY_PRIVATE_NETWORK_MODEL_ID:
        return active_policy.network_default_model_id
    return model_id


def is_private_curated_model_id(
    model_id: str, policy: NetworkModelPolicy | None = None
) -> bool:
    active_policy = policy or NetworkModelPolicy()
    normalized_model_id = normalize_network_model_id(model_id, active_policy)
    return normalized_model_id in active_policy.private_curated_model_ids


def curated_model_registry() -> tuple[CuratedNetworkModel, ...]:
    return CURATED_NETWORK_MODELS


def curated_network_model_ids() -> tuple[str, ...]:
    return tuple(model.model_id for model in CURATED_NETWORK_MODELS)


def curated_worker_default_model_ids() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(model.model_id for model in CURATED_NETWORK_MODELS)
    )


def curated_private_model_ids() -> tuple[str, ...]:
    return DEFAULT_PRIVATE_CURATED_MODEL_IDS


def curated_model_for_id(model_id: str) -> CuratedNetworkModel | None:
    candidate = str(model_id or "").strip()
    if not candidate:
        return None
    for model in CURATED_NETWORK_MODELS:
        if candidate == model.model_id:
            return model
    for model in CURATED_NETWORK_MODELS:
        if candidate == model.execution_model_id:
            return model
        if candidate in model.runtime_model_ids:
            return model
    return None


def is_registered_curated_model_id(model_id: str) -> bool:
    return curated_model_for_id(model_id) is not None
