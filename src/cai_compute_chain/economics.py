# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import ipaddress
import json
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from math import ceil
from urllib.error import URLError
from urllib.request import urlopen

from .model import (
    FundingSource,
    MoneyPolicy,
    NetworkModelPolicy,
    normalize_network_model_id,
    PaymentPreference,
    WalletPolicy,
)
from .chain import (
    chain_balance_atomic,
    compute_reserve_chain_address,
    ensure_chain_genesis,
    tx_fee_pool_chain_address,
    validator_settlement_fee_pool_chain_address,
    validator_slash_pool_chain_address,
)
from .wallet import (
    LedgerState,
    WalletRecord,
    atomic_to_coins,
    coins_to_atomic,
    update_wallet,
)


@dataclass(frozen=True)
class FeeQuote:
    compute_cost_atomic: int
    tx_fee_atomic: int
    settlement_fee_atomic: int
    ai_development_fee_atomic: int
    worker_reward_atomic: int


@dataclass(frozen=True)
class FundingDecision:
    can_fund: bool
    reason: str
    payment_preference: PaymentPreference
    funding_source: FundingSource | None
    fee_quote: FeeQuote
    reserve_before_atomic: int
    reserve_after_atomic: int
    wallet_before_atomic: int
    wallet_after_atomic: int
    daily_reserve_limit_atomic: int | None = None
    daily_reserve_spent_today_atomic: int = 0
    daily_reserve_remaining_atomic: int | None = None
    reserve_limit_identity_keys: tuple[str, ...] = ()
    reserve_client_ip_hash: str | None = None

    def pretty_lines(self, *, money_policy: MoneyPolicy | None = None) -> list[str]:
        active_money_policy = money_policy or MoneyPolicy()
        return [
            f"- can_fund={self.can_fund}",
            f"- reason={self.reason}",
            f"- payment_preference={self.payment_preference.value}",
            f"- funding_source={self.funding_source.value if self.funding_source else 'none'}",
            f"- compute_cost={atomic_to_coins(self.fee_quote.compute_cost_atomic, active_money_policy)}",
            f"- tx_fee={atomic_to_coins(self.fee_quote.tx_fee_atomic, active_money_policy)}",
            f"- settlement_fee={atomic_to_coins(self.fee_quote.settlement_fee_atomic, active_money_policy)}",
            f"- ai_development_fee={atomic_to_coins(self.fee_quote.ai_development_fee_atomic, active_money_policy)}",
            f"- worker_reward={atomic_to_coins(self.fee_quote.worker_reward_atomic, active_money_policy)}",
            f"- reserve_before={atomic_to_coins(self.reserve_before_atomic, active_money_policy)}",
            f"- reserve_after={atomic_to_coins(self.reserve_after_atomic, active_money_policy)}",
            f"- wallet_before={atomic_to_coins(self.wallet_before_atomic, active_money_policy)}",
            f"- wallet_after={atomic_to_coins(self.wallet_after_atomic, active_money_policy)}",
            (
                f"- daily_reserve_limit="
                f"{atomic_to_coins(self.daily_reserve_limit_atomic, active_money_policy)}"
                if self.daily_reserve_limit_atomic is not None
                else "- daily_reserve_limit=<disabled>"
            ),
            f"- daily_reserve_spent_today={atomic_to_coins(self.daily_reserve_spent_today_atomic, active_money_policy)}",
            (
                f"- daily_reserve_remaining="
                f"{atomic_to_coins(self.daily_reserve_remaining_atomic, active_money_policy)}"
                if self.daily_reserve_remaining_atomic is not None
                else "- daily_reserve_remaining=<disabled>"
            ),
        ]


@dataclass(frozen=True)
class NetworkStatePricingSnapshot:
    state_url: str | None
    reachable: bool
    topology_nodes: int
    topology_connections: int
    node_system_entries: int
    average_cpu_usage: float | None
    error: str | None = None


def chain_backed_ledger_snapshot(
    ledger: LedgerState,
    *,
    money_policy: MoneyPolicy | None = None,
    wallet_policy: WalletPolicy | None = None,
) -> LedgerState:
    active_money_policy = money_policy or MoneyPolicy()
    ensure_chain_genesis(policy=wallet_policy, money_policy=active_money_policy)
    return replace(
        ledger,
        compute_reserve_balance_atomic=chain_balance_atomic(
            compute_reserve_chain_address(active_money_policy),
            wallet_policy,
        ),
        project_treasury_balance_atomic=chain_balance_atomic(
            active_money_policy.developer_treasury_address,
            wallet_policy,
        ),
        ai_development_fee_pool_atomic=chain_balance_atomic(
            active_money_policy.ai_development_address,
            wallet_policy,
        ),
        validator_fee_pool_atomic=chain_balance_atomic(
            validator_settlement_fee_pool_chain_address(active_money_policy),
            wallet_policy,
        ),
        validator_slashed_atomic=chain_balance_atomic(
            validator_slash_pool_chain_address(active_money_policy),
            wallet_policy,
        ),
        tx_fee_pool_atomic=chain_balance_atomic(
            tx_fee_pool_chain_address(active_money_policy),
            wallet_policy,
        ),
    )


@dataclass(frozen=True)
class AutomaticPriceQuote:
    pricing_mode: str
    pricing_basis: str
    compute_cost_atomic: int
    base_price_atomic: int
    prompt_units: int
    prompt_tokens_estimate: int | None
    reserved_output_tokens: int | None
    input_token_price_atomic: int | None
    output_token_price_atomic: int | None
    model_multiplier_bps: int
    load_multiplier_bps: int
    reserve_multiplier_bps: int
    final_multiplier_bps: int
    was_capped: bool
    snapshot: NetworkStatePricingSnapshot
    reason: str

    def pretty_lines(self, *, money_policy: MoneyPolicy | None = None) -> list[str]:
        active_money_policy = money_policy or MoneyPolicy()
        avg_cpu = (
            f"{self.snapshot.average_cpu_usage:.3f}"
            if self.snapshot.average_cpu_usage is not None
            else "<none>"
        )
        return [
            f"- pricing_mode={self.pricing_mode}",
            f"- pricing_basis={self.pricing_basis}",
            f"- reason={self.reason}",
            f"- compute_cost={atomic_to_coins(self.compute_cost_atomic, active_money_policy)}",
            f"- base_price={atomic_to_coins(self.base_price_atomic, active_money_policy)}",
            (
                f"- prompt_tokens_estimate={self.prompt_tokens_estimate}"
                if self.prompt_tokens_estimate is not None
                else f"- prompt_units={self.prompt_units}"
            ),
            (
                f"- reserved_output_tokens={self.reserved_output_tokens}"
                if self.reserved_output_tokens is not None
                else "- reserved_output_tokens=<none>"
            ),
            (
                f"- input_token_price={atomic_to_coins(self.input_token_price_atomic, active_money_policy)}"
                if self.input_token_price_atomic is not None
                else "- input_token_price=<none>"
            ),
            (
                f"- output_token_price={atomic_to_coins(self.output_token_price_atomic, active_money_policy)}"
                if self.output_token_price_atomic is not None
                else "- output_token_price=<none>"
            ),
            f"- model_multiplier_bps={self.model_multiplier_bps}",
            f"- load_multiplier_bps={self.load_multiplier_bps}",
            f"- reserve_multiplier_bps={self.reserve_multiplier_bps}",
            f"- final_multiplier_bps={self.final_multiplier_bps}",
            f"- was_capped={self.was_capped}",
            f"- state_url={self.snapshot.state_url or '<none>'}",
            f"- state_reachable={self.snapshot.reachable}",
            f"- topology_nodes={self.snapshot.topology_nodes}",
            f"- topology_connections={self.snapshot.topology_connections}",
            f"- node_system_entries={self.snapshot.node_system_entries}",
            f"- average_cpu_usage={avg_cpu}",
            f"- state_error={self.snapshot.error or '<none>'}",
        ]


@dataclass(frozen=True)
class ResolvedPriceQuote:
    compute_cost_atomic: int
    pricing_mode: str
    pricing_basis: str
    pricing_reason: str
    automatic_quote: AutomaticPriceQuote | None = None


@dataclass(frozen=True)
class TokenPricedCost:
    compute_cost_atomic: int
    base_price_atomic: int
    was_capped: bool


def _coalesce_cai_url(cai_url: str | None = None, CAI_url: str | None = None) -> str | None:
    resolved = str(cai_url or CAI_url or "").strip()
    return resolved or None


def build_fee_quote(
    compute_cost_atomic: int,
    *,
    money_policy: MoneyPolicy | None = None,
    tx_fee_atomic: int | None = None,
) -> FeeQuote:
    active_money_policy = money_policy or MoneyPolicy()
    resolved_tx_fee = (
        tx_fee_atomic
        if tx_fee_atomic is not None
        else _default_tx_fee_atomic(active_money_policy)
    )
    settlement_fee = (
        compute_cost_atomic * active_money_policy.validator_settlement_fee_bps
    ) // 10_000
    ai_development_fee = (
        compute_cost_atomic * active_money_policy.ai_development_fee_bps
    ) // 10_000
    worker_reward = max(compute_cost_atomic - settlement_fee - ai_development_fee, 0)
    return FeeQuote(
        compute_cost_atomic=compute_cost_atomic,
        tx_fee_atomic=resolved_tx_fee,
        settlement_fee_atomic=settlement_fee,
        ai_development_fee_atomic=ai_development_fee,
        worker_reward_atomic=worker_reward,
    )


def resolve_compute_price(
    *,
    compute_amount_coins: str | None,
    prompt: str | None,
    model_id: str,
    ledger: LedgerState,
    max_output_tokens: int | None = None,
    cai_url: str | None = None,
    CAI_url: str | None = None,
    money_policy: MoneyPolicy | None = None,
    network_model_policy: NetworkModelPolicy | None = None,
) -> ResolvedPriceQuote:
    active_money_policy = money_policy or MoneyPolicy()
    if compute_amount_coins is not None:
        return ResolvedPriceQuote(
            compute_cost_atomic=coins_to_atomic(compute_amount_coins, active_money_policy),
            pricing_mode="manual",
            pricing_basis="manual",
            pricing_reason="Manual compute amount was provided by the user.",
            automatic_quote=None,
        )

    if not prompt or not prompt.strip():
        raise ValueError(
            "Prompt is required when automatic network pricing is used without --amount."
        )

    auto_quote = quote_automatic_compute_price(
        prompt=prompt,
        model_id=model_id,
        max_output_tokens=max_output_tokens,
        cai_url=_coalesce_cai_url(cai_url, CAI_url),
        ledger=ledger,
        money_policy=active_money_policy,
        network_model_policy=network_model_policy,
    )
    return ResolvedPriceQuote(
        compute_cost_atomic=auto_quote.compute_cost_atomic,
        pricing_mode=auto_quote.pricing_mode,
        pricing_basis=auto_quote.pricing_basis,
        pricing_reason=auto_quote.reason,
        automatic_quote=auto_quote,
    )


def quote_automatic_compute_price(
    *,
    prompt: str,
    model_id: str,
    ledger: LedgerState,
    max_output_tokens: int | None = None,
    cai_url: str | None = None,
    CAI_url: str | None = None,
    money_policy: MoneyPolicy | None = None,
    network_model_policy: NetworkModelPolicy | None = None,
    snapshot: NetworkStatePricingSnapshot | None = None,
) -> AutomaticPriceQuote:
    active_money_policy = money_policy or MoneyPolicy()
    active_network_model_policy = network_model_policy or NetworkModelPolicy()
    active_snapshot = snapshot or fetch_network_state_pricing_snapshot(
        _coalesce_cai_url(cai_url, CAI_url)
    )

    floor_atomic = coins_to_atomic(
        active_money_policy.automatic_price_floor_coins, active_money_policy
    )
    cap_atomic = coins_to_atomic(
        active_money_policy.automatic_price_cap_coins, active_money_policy
    )
    per_unit_atomic = coins_to_atomic(
        active_money_policy.automatic_price_per_prompt_unit_coins, active_money_policy
    )
    pricing_basis = "prompt_chars"
    prompt_units = 0
    prompt_tokens_estimate: int | None = None
    reserved_output_tokens: int | None = None
    input_token_price_atomic: int | None = None
    output_token_price_atomic: int | None = None
    if active_money_policy.automatic_token_pricing_enabled:
        pricing_basis = "llm_tokens"
        prompt_tokens_estimate = estimate_prompt_token_upper_bound(prompt)
        reserved_output_tokens = resolve_reserved_output_tokens(
            max_output_tokens=max_output_tokens,
            money_policy=active_money_policy,
        )
        input_token_price_atomic = coins_to_atomic(
            active_money_policy.automatic_price_per_input_token_coins,
            active_money_policy,
        )
        output_token_price_atomic = coins_to_atomic(
            active_money_policy.automatic_price_per_output_token_coins,
            active_money_policy,
        )
        base_price_atomic = (
            prompt_tokens_estimate * input_token_price_atomic
            + reserved_output_tokens * output_token_price_atomic
        )
    else:
        prompt_units = max(
            1,
            ceil(
                max(len(prompt.strip()), 1)
                / max(active_money_policy.automatic_price_prompt_unit_chars, 1)
            ),
        )
        base_price_atomic = floor_atomic + (max(prompt_units - 1, 0) * per_unit_atomic)

    model_multiplier_bps = _model_multiplier_bps(
        model_id, active_money_policy, active_network_model_policy
    )
    load_multiplier_bps, load_reason = _load_multiplier_bps(
        active_snapshot, active_money_policy, active_network_model_policy
    )
    reserve_multiplier_bps, reserve_reason = _reserve_multiplier_bps(
        ledger, active_money_policy
    )

    final_multiplier_bps = (
        model_multiplier_bps * load_multiplier_bps * reserve_multiplier_bps
    ) // (10_000 * 10_000)
    final_multiplier_bps = max(final_multiplier_bps, 1_000)
    token_cost = calculate_token_priced_cost(
        prompt_tokens=prompt_tokens_estimate,
        completion_tokens=reserved_output_tokens,
        final_multiplier_bps=final_multiplier_bps,
        money_policy=active_money_policy,
        input_token_price_atomic=input_token_price_atomic,
        output_token_price_atomic=output_token_price_atomic,
        floor_atomic=floor_atomic,
        cap_atomic=cap_atomic,
    )
    if pricing_basis == "llm_tokens":
        bounded_compute_cost_atomic = token_cost.compute_cost_atomic
        was_capped = token_cost.was_capped
    else:
        raw_compute_cost_atomic = (base_price_atomic * final_multiplier_bps) // 10_000
        bounded_compute_cost_atomic = min(max(raw_compute_cost_atomic, floor_atomic), cap_atomic)
        was_capped = raw_compute_cost_atomic > cap_atomic

    reason_parts = [
        "Automatic network pricing keeps compute within a bounded protocol range.",
        (
            "Prompt and reserved output are priced by LLM token allowance."
            if pricing_basis == "llm_tokens"
            else f"Prompt size contributes {prompt_units} unit(s)."
        ),
        load_reason,
        reserve_reason,
    ]
    if pricing_basis == "llm_tokens" and prompt_tokens_estimate is not None:
        reason_parts.append(
            f"Reserved prompt/output budget: {prompt_tokens_estimate} input token(s) and {reserved_output_tokens} output token(s)."
        )
    if was_capped:
        reason_parts.append("Protocol price cap prevented the quote from becoming too high.")

    return AutomaticPriceQuote(
        pricing_mode="network_auto",
        pricing_basis=pricing_basis,
        compute_cost_atomic=bounded_compute_cost_atomic,
        base_price_atomic=base_price_atomic,
        prompt_units=prompt_units,
        prompt_tokens_estimate=prompt_tokens_estimate,
        reserved_output_tokens=reserved_output_tokens,
        input_token_price_atomic=input_token_price_atomic,
        output_token_price_atomic=output_token_price_atomic,
        model_multiplier_bps=model_multiplier_bps,
        load_multiplier_bps=load_multiplier_bps,
        reserve_multiplier_bps=reserve_multiplier_bps,
        final_multiplier_bps=final_multiplier_bps,
        was_capped=was_capped,
        snapshot=active_snapshot,
        reason=" ".join(part.strip() for part in reason_parts if part.strip()),
    )


def estimate_prompt_token_upper_bound(prompt: str | None) -> int:
    normalized = str(prompt or "").strip()
    return max(len(normalized), 1)


def resolve_reserved_output_tokens(
    *,
    max_output_tokens: int | None,
    money_policy: MoneyPolicy | None = None,
) -> int:
    active_money_policy = money_policy or MoneyPolicy()
    if max_output_tokens is not None and int(max_output_tokens) > 0:
        return int(max_output_tokens)
    return max(int(active_money_policy.automatic_price_default_reserved_output_tokens), 1)


def calculate_token_priced_cost(
    *,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    final_multiplier_bps: int,
    money_policy: MoneyPolicy | None = None,
    input_token_price_atomic: int | None = None,
    output_token_price_atomic: int | None = None,
    floor_atomic: int | None = None,
    cap_atomic: int | None = None,
) -> TokenPricedCost:
    active_money_policy = money_policy or MoneyPolicy()
    resolved_input_price_atomic = (
        input_token_price_atomic
        if input_token_price_atomic is not None
        else coins_to_atomic(
            active_money_policy.automatic_price_per_input_token_coins,
            active_money_policy,
        )
    )
    resolved_output_price_atomic = (
        output_token_price_atomic
        if output_token_price_atomic is not None
        else coins_to_atomic(
            active_money_policy.automatic_price_per_output_token_coins,
            active_money_policy,
        )
    )
    resolved_floor_atomic = (
        floor_atomic
        if floor_atomic is not None
        else coins_to_atomic(
            active_money_policy.automatic_price_floor_coins,
            active_money_policy,
        )
    )
    resolved_cap_atomic = (
        cap_atomic
        if cap_atomic is not None
        else coins_to_atomic(
            active_money_policy.automatic_price_cap_coins,
            active_money_policy,
        )
    )
    base_price_atomic = max(int(prompt_tokens or 0), 0) * resolved_input_price_atomic
    base_price_atomic += max(int(completion_tokens or 0), 0) * resolved_output_price_atomic
    raw_compute_cost_atomic = (base_price_atomic * max(int(final_multiplier_bps), 1_000)) // 10_000
    bounded_compute_cost_atomic = min(
        max(raw_compute_cost_atomic, resolved_floor_atomic),
        resolved_cap_atomic,
    )
    return TokenPricedCost(
        compute_cost_atomic=bounded_compute_cost_atomic,
        base_price_atomic=base_price_atomic,
        was_capped=raw_compute_cost_atomic > resolved_cap_atomic,
    )


def fetch_network_state_pricing_snapshot(
    cai_url: str | None = None,
    CAI_url: str | None = None,
) -> NetworkStatePricingSnapshot:
    cai_url = _coalesce_cai_url(cai_url, CAI_url)
    if not cai_url:
        return NetworkStatePricingSnapshot(
            state_url=None,
            reachable=False,
            topology_nodes=0,
            topology_connections=0,
            node_system_entries=0,
            average_cpu_usage=None,
            error="cai_url is not set",
        )

    state_url = f"{cai_url.rstrip('/')}/state"
    try:
        with urlopen(state_url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (TimeoutError, URLError, OSError, json.JSONDecodeError) as exc:
        return NetworkStatePricingSnapshot(
            state_url=state_url,
            reachable=False,
            topology_nodes=0,
            topology_connections=0,
            node_system_entries=0,
            average_cpu_usage=None,
            error=str(exc),
        )

    topology = payload.get("topology") or {}
    node_system = payload.get("nodeSystem") or {}
    return NetworkStatePricingSnapshot(
        state_url=state_url,
        reachable=True,
        topology_nodes=len(topology.get("nodes") or []),
        topology_connections=len(topology.get("connections") or {}),
        node_system_entries=len(node_system),
        average_cpu_usage=_extract_average_cpu_usage(node_system),
        error=None,
    )


def plan_funding(
    *,
    ledger: LedgerState,
    wallet: WalletRecord,
    compute_cost_atomic: int,
    payment_preference: PaymentPreference,
    money_policy: MoneyPolicy | None = None,
    wallet_policy: WalletPolicy | None = None,
    tx_fee_atomic: int | None = None,
    reserve_client_ip: str | None = None,
    reserve_limit_identity_keys: tuple[str, ...] | None = None,
    reserve_client_hash: str | None = None,
) -> FundingDecision:
    active_money_policy = money_policy or MoneyPolicy()
    fee_quote = build_fee_quote(
        compute_cost_atomic,
        money_policy=active_money_policy,
        tx_fee_atomic=tx_fee_atomic,
    )
    reserve_before = ledger.compute_reserve_balance_atomic
    wallet_before = wallet.spendable_balance_atomic
    fallback_daily_limit_atomic = _daily_user_reserve_limit_atomic(active_money_policy)
    provided_reserve_limit_identity_keys = (
        tuple(str(item) for item in reserve_limit_identity_keys if str(item).strip())
        if reserve_limit_identity_keys is not None
        else ()
    )
    resolved_reserve_limit_identity_keys = (
        provided_reserve_limit_identity_keys
        or build_reserve_limit_identity_keys(
            wallet.wallet_id,
            reserve_client_ip=reserve_client_ip,
            money_policy=active_money_policy,
        )
    )
    resolved_reserve_client_hash = (
        str(reserve_client_hash).strip()
        if reserve_client_hash is not None and str(reserve_client_hash).strip()
        else reserve_client_ip_hash(reserve_client_ip)
    )
    daily_spent_today_atomic = _daily_user_reserve_spent_today_atomic(
        wallet.wallet_id,
        money_policy=active_money_policy,
        wallet_policy=wallet_policy,
        reserve_limit_identity_keys=resolved_reserve_limit_identity_keys,
    )
    reserve_identity_spent_atomic = _daily_reserve_spent_today_by_identity(
        wallet.wallet_id,
        reserve_limit_identity_keys=resolved_reserve_limit_identity_keys,
        wallet_policy=wallet_policy,
    )
    reserve_identity_limits_atomic = {
        key: limit
        for key in resolved_reserve_limit_identity_keys
        if (
            limit := _daily_reserve_limit_for_identity_key(key, active_money_policy)
        )
        is not None
    }
    daily_limit_atomic = (
        min(reserve_identity_limits_atomic.values())
        if reserve_identity_limits_atomic
        else fallback_daily_limit_atomic
    )
    reserve_metered_compute_atomic = fee_quote.compute_cost_atomic
    daily_remaining_atomic = (
        min(
            max(
                limit - int(reserve_identity_spent_atomic.get(identity_key, 0)),
                0,
            )
            for identity_key, limit in reserve_identity_limits_atomic.items()
        )
        if reserve_identity_limits_atomic
        else None
    )
    reserve_limit_exceeded = any(
        int(reserve_identity_spent_atomic.get(identity_key, 0))
        + reserve_metered_compute_atomic
        > limit
        for identity_key, limit in reserve_identity_limits_atomic.items()
    )

    if payment_preference is PaymentPreference.RESERVE_ONLY:
        if reserve_limit_exceeded:
            return _reject(
                (
                    f"Daily reserve allowance of "
                    f"{atomic_to_coins(daily_limit_atomic, active_money_policy)} "
                    f"{active_money_policy.currency_code} would be exceeded. "
                    f"Already reserve-funded today: "
                    f"{atomic_to_coins(daily_spent_today_atomic, active_money_policy)}."
                ),
                payment_preference,
                fee_quote,
                reserve_before,
                wallet_before,
                daily_reserve_limit_atomic=daily_limit_atomic,
                daily_reserve_spent_today_atomic=daily_spent_today_atomic,
                daily_reserve_remaining_atomic=daily_remaining_atomic,
                reserve_limit_identity_keys=resolved_reserve_limit_identity_keys,
                reserve_client_ip_hash=resolved_reserve_client_hash,
            )
        if reserve_before < fee_quote.compute_cost_atomic:
            return _reject(
                "Reserve balance is insufficient for the requested compute cost.",
                payment_preference,
                fee_quote,
                reserve_before,
                wallet_before,
                daily_reserve_limit_atomic=daily_limit_atomic,
                daily_reserve_spent_today_atomic=daily_spent_today_atomic,
                daily_reserve_remaining_atomic=daily_remaining_atomic,
                reserve_limit_identity_keys=resolved_reserve_limit_identity_keys,
                reserve_client_ip_hash=resolved_reserve_client_hash,
            )
        if wallet_before < fee_quote.tx_fee_atomic:
            return _reject(
                "Wallet balance is insufficient to pay the transaction fee.",
                payment_preference,
                fee_quote,
                reserve_before,
                wallet_before,
                daily_reserve_limit_atomic=daily_limit_atomic,
                daily_reserve_spent_today_atomic=daily_spent_today_atomic,
                daily_reserve_remaining_atomic=daily_remaining_atomic,
                reserve_limit_identity_keys=resolved_reserve_limit_identity_keys,
                reserve_client_ip_hash=resolved_reserve_client_hash,
            )
        return FundingDecision(
            can_fund=True,
            reason="Compute is funded by reserve, tx fee by wallet.",
            payment_preference=payment_preference,
            funding_source=FundingSource.RESERVE,
            fee_quote=fee_quote,
            reserve_before_atomic=reserve_before,
            reserve_after_atomic=reserve_before - fee_quote.compute_cost_atomic,
            wallet_before_atomic=wallet_before,
            wallet_after_atomic=wallet_before - fee_quote.tx_fee_atomic,
            daily_reserve_limit_atomic=daily_limit_atomic,
            daily_reserve_spent_today_atomic=daily_spent_today_atomic,
            daily_reserve_remaining_atomic=(
                None
                if daily_remaining_atomic is None
                else max(daily_remaining_atomic - reserve_metered_compute_atomic, 0)
            ),
            reserve_limit_identity_keys=resolved_reserve_limit_identity_keys,
            reserve_client_ip_hash=resolved_reserve_client_hash,
        )

    wallet_total_required = fee_quote.compute_cost_atomic + fee_quote.tx_fee_atomic
    if payment_preference is PaymentPreference.WALLET_ONLY:
        if wallet_before < wallet_total_required:
            return _reject(
                "Wallet balance is insufficient to fund compute cost and tx fee.",
                payment_preference,
                fee_quote,
                reserve_before,
                wallet_before,
                daily_reserve_limit_atomic=daily_limit_atomic,
                daily_reserve_spent_today_atomic=daily_spent_today_atomic,
                daily_reserve_remaining_atomic=daily_remaining_atomic,
                reserve_limit_identity_keys=resolved_reserve_limit_identity_keys,
                reserve_client_ip_hash=resolved_reserve_client_hash,
            )
        return FundingDecision(
            can_fund=True,
            reason="Compute and tx fee are funded by wallet.",
            payment_preference=payment_preference,
            funding_source=FundingSource.WALLET,
            fee_quote=fee_quote,
            reserve_before_atomic=reserve_before,
            reserve_after_atomic=reserve_before,
            wallet_before_atomic=wallet_before,
            wallet_after_atomic=wallet_before - wallet_total_required,
            daily_reserve_limit_atomic=daily_limit_atomic,
            daily_reserve_spent_today_atomic=daily_spent_today_atomic,
            daily_reserve_remaining_atomic=daily_remaining_atomic,
            reserve_limit_identity_keys=resolved_reserve_limit_identity_keys,
            reserve_client_ip_hash=resolved_reserve_client_hash,
        )

    # AUTO mode: reserve first while allowance remains, then wallet fallback.
    reserve_allowed = (
        not reserve_limit_exceeded
    )
    if (
        reserve_allowed
        and reserve_before >= fee_quote.compute_cost_atomic
        and wallet_before >= fee_quote.tx_fee_atomic
    ):
        return FundingDecision(
            can_fund=True,
            reason="Auto mode selected reserve for compute and wallet for tx fee.",
            payment_preference=payment_preference,
            funding_source=FundingSource.RESERVE,
            fee_quote=fee_quote,
            reserve_before_atomic=reserve_before,
            reserve_after_atomic=reserve_before - fee_quote.compute_cost_atomic,
            wallet_before_atomic=wallet_before,
            wallet_after_atomic=wallet_before - fee_quote.tx_fee_atomic,
            daily_reserve_limit_atomic=daily_limit_atomic,
            daily_reserve_spent_today_atomic=daily_spent_today_atomic,
            daily_reserve_remaining_atomic=(
                None
                if daily_remaining_atomic is None
                else max(daily_remaining_atomic - reserve_metered_compute_atomic, 0)
            ),
            reserve_limit_identity_keys=resolved_reserve_limit_identity_keys,
            reserve_client_ip_hash=resolved_reserve_client_hash,
        )

    reserve_total_required = fee_quote.compute_cost_atomic + fee_quote.tx_fee_atomic
    if reserve_allowed and reserve_before >= reserve_total_required:
        return FundingDecision(
            can_fund=True,
            reason="Auto mode selected reserve for compute and tx fee.",
            payment_preference=payment_preference,
            funding_source=FundingSource.RESERVE,
            fee_quote=fee_quote,
            reserve_before_atomic=reserve_before,
            reserve_after_atomic=reserve_before - reserve_total_required,
            wallet_before_atomic=wallet_before,
            wallet_after_atomic=wallet_before,
            daily_reserve_limit_atomic=daily_limit_atomic,
            daily_reserve_spent_today_atomic=daily_spent_today_atomic,
            daily_reserve_remaining_atomic=(
                None
                if daily_remaining_atomic is None
                else max(daily_remaining_atomic - reserve_metered_compute_atomic, 0)
            ),
            reserve_limit_identity_keys=resolved_reserve_limit_identity_keys,
            reserve_client_ip_hash=resolved_reserve_client_hash,
        )

    if wallet_before >= wallet_total_required:
        return FundingDecision(
            can_fund=True,
            reason=(
                "Auto mode fell back to wallet funding after daily reserve allowance was exhausted."
                if not reserve_allowed
                else "Auto mode fell back to wallet funding."
            ),
            payment_preference=payment_preference,
            funding_source=FundingSource.WALLET,
            fee_quote=fee_quote,
            reserve_before_atomic=reserve_before,
            reserve_after_atomic=reserve_before,
            wallet_before_atomic=wallet_before,
            wallet_after_atomic=wallet_before - wallet_total_required,
            daily_reserve_limit_atomic=daily_limit_atomic,
            daily_reserve_spent_today_atomic=daily_spent_today_atomic,
            daily_reserve_remaining_atomic=daily_remaining_atomic,
            reserve_limit_identity_keys=resolved_reserve_limit_identity_keys,
            reserve_client_ip_hash=resolved_reserve_client_hash,
        )

    return _reject(
        (
            "Daily reserve allowance is exhausted and wallet balance is insufficient to fund this compute job."
            if not reserve_allowed
            else "Neither reserve+wallet nor wallet-only path can fund this compute job."
        ),
        payment_preference,
        fee_quote,
        reserve_before,
        wallet_before,
        daily_reserve_limit_atomic=daily_limit_atomic,
        daily_reserve_spent_today_atomic=daily_spent_today_atomic,
        daily_reserve_remaining_atomic=daily_remaining_atomic,
        reserve_limit_identity_keys=resolved_reserve_limit_identity_keys,
        reserve_client_ip_hash=resolved_reserve_client_hash,
    )


def apply_funding_decision(
    *,
    ledger: LedgerState,
    wallet: WalletRecord,
    decision: FundingDecision,
    money_policy: MoneyPolicy | None = None,
) -> tuple[LedgerState, WalletRecord]:
    _ = money_policy or MoneyPolicy()
    if not decision.can_fund or decision.funding_source is None:
        raise ValueError("Cannot apply an unfunded decision.")

    wallet.spendable_balance_atomic = decision.wallet_after_atomic

    if decision.funding_source is FundingSource.RESERVE:
        ledger.compute_reserve_balance_atomic = decision.reserve_after_atomic
    else:
        ledger.compute_reserve_balance_atomic = decision.reserve_after_atomic

    ledger.validator_fee_pool_atomic += decision.fee_quote.settlement_fee_atomic
    ledger.ai_development_fee_pool_atomic += decision.fee_quote.ai_development_fee_atomic
    ledger.tx_fee_pool_atomic += decision.fee_quote.tx_fee_atomic
    ledger.worker_distributed_atomic += decision.fee_quote.worker_reward_atomic
    ledger.settlements_applied += 1

    return ledger, wallet


def save_applied_funding(
    *,
    ledger: LedgerState,
    wallet: WalletRecord,
    decision: FundingDecision,
) -> tuple[LedgerState, WalletRecord]:
    updated_ledger, updated_wallet = apply_funding_decision(
        ledger=ledger,
        wallet=wallet,
        decision=decision,
    )
    update_wallet(updated_wallet)
    return updated_ledger, updated_wallet


def _default_tx_fee_atomic(money_policy: MoneyPolicy) -> int:
    whole, _, fraction = money_policy.default_tx_fee_coins.partition(".")
    if not whole:
        whole = "0"
    fraction = (fraction + ("0" * money_policy.decimals))[: money_policy.decimals]
    return int(whole) * (10**money_policy.decimals) + int(fraction or "0")


def _model_multiplier_bps(
    model_id: str,
    money_policy: MoneyPolicy,
    network_model_policy: NetworkModelPolicy,
) -> int:
    normalized_model_id = normalize_network_model_id(model_id, network_model_policy)
    known_network_model_ids = {
        normalize_network_model_id(candidate, network_model_policy)
        for candidate in network_model_policy.network_execution_model_ids
    }
    if normalized_model_id in known_network_model_ids:
        return 10_000
    return 10_000 + money_policy.automatic_price_non_default_model_premium_bps


def _load_multiplier_bps(
    snapshot: NetworkStatePricingSnapshot,
    money_policy: MoneyPolicy,
    network_model_policy: NetworkModelPolicy,
) -> tuple[int, str]:
    factor_bps = 10_000
    reasons: list[str] = []
    if not snapshot.reachable:
        factor_bps += money_policy.automatic_price_unreachable_safety_bps
        reasons.append("Network state was unreachable, so a conservative but bounded price was used.")
        return factor_bps, " ".join(reasons)

    min_connections = max(
        money_policy.automatic_price_target_connections,
        network_model_policy.minimum_worker_shards,
    )
    if snapshot.topology_connections < min_connections:
        factor_bps += money_policy.automatic_price_connection_scarcity_surcharge_bps
        reasons.append(
            f"Only {snapshot.topology_connections} routable connection(s) are available, so the network applied a small scarcity surcharge."
        )
    elif snapshot.topology_connections > min_connections:
        factor_bps = max(
            factor_bps - money_policy.automatic_price_healthy_network_discount_bps,
            1_000,
        )
        reasons.append("Healthy multi-node connectivity enabled a small protocol discount.")

    if snapshot.average_cpu_usage is not None:
        if snapshot.average_cpu_usage >= money_policy.automatic_price_high_load_threshold:
            factor_bps += money_policy.automatic_price_high_load_surcharge_bps
            reasons.append("Current node load is high, so the quote added a bounded load surcharge.")
        elif (
            snapshot.average_cpu_usage <= money_policy.automatic_price_low_load_threshold
            and snapshot.topology_connections >= min_connections
        ):
            factor_bps = max(
                factor_bps - money_policy.automatic_price_low_load_discount_bps,
                1_000,
            )
            reasons.append("Current node load is low, so the quote applied a small discount.")

    if not reasons:
        reasons.append("Network load stayed within the normal target range.")
    return factor_bps, " ".join(reasons)


def _reserve_multiplier_bps(
    ledger: LedgerState, money_policy: MoneyPolicy
) -> tuple[int, str]:
    initial_reserve_atomic = money_policy.compute_reserve_coins * (10**money_policy.decimals)
    if initial_reserve_atomic <= 0:
        return 10_000, "Reserve baseline is unavailable, so no reserve adjustment was applied."

    reserve_ratio = ledger.compute_reserve_balance_atomic / initial_reserve_atomic
    if reserve_ratio <= 0.10:
        return (
            10_000 + money_policy.automatic_price_reserve_critical_bps,
            "Compute reserve is critically low, so the quote added a bounded protection premium.",
        )
    if reserve_ratio <= 0.25:
        return (
            10_000 + money_policy.automatic_price_reserve_guard_bps,
            "Compute reserve is getting low, so the quote added a small reserve-protection premium.",
        )
    return 10_000, "Compute reserve is healthy, so no reserve premium was added."


def _extract_average_cpu_usage(node_system: dict) -> float | None:
    usages: list[float] = []
    for value in node_system.values():
        if not isinstance(value, dict):
            continue
        sample: list[float] = []
        for key in ("pcpuUsage", "ecpuUsage"):
            metric = value.get(key)
            if isinstance(metric, (int, float)):
                sample.append(float(metric))
        if sample:
            usages.append(sum(sample))
    if not usages:
        return None
    return sum(usages) / len(usages)


def _reject(
    reason: str,
    payment_preference: PaymentPreference,
    fee_quote: FeeQuote,
    reserve_before: int,
    wallet_before: int,
    *,
    daily_reserve_limit_atomic: int | None = None,
    daily_reserve_spent_today_atomic: int = 0,
    daily_reserve_remaining_atomic: int | None = None,
    reserve_limit_identity_keys: tuple[str, ...] = (),
    reserve_client_ip_hash: str | None = None,
) -> FundingDecision:
    return FundingDecision(
        can_fund=False,
        reason=reason,
        payment_preference=payment_preference,
        funding_source=None,
        fee_quote=fee_quote,
        reserve_before_atomic=reserve_before,
        reserve_after_atomic=reserve_before,
        wallet_before_atomic=wallet_before,
        wallet_after_atomic=wallet_before,
        daily_reserve_limit_atomic=daily_reserve_limit_atomic,
        daily_reserve_spent_today_atomic=daily_reserve_spent_today_atomic,
        daily_reserve_remaining_atomic=daily_reserve_remaining_atomic,
        reserve_limit_identity_keys=reserve_limit_identity_keys,
        reserve_client_ip_hash=reserve_client_ip_hash,
    )


def _daily_user_reserve_limit_atomic(money_policy: MoneyPolicy) -> int | None:
    if not money_policy.daily_user_reserve_limit_enabled:
        return None
    limit = str(money_policy.daily_user_reserve_limit_coins).strip()
    if not limit:
        return None
    return max(coins_to_atomic(limit, money_policy), 0)


def _daily_ip_reserve_limit_atomic(money_policy: MoneyPolicy) -> int | None:
    if not money_policy.daily_ip_reserve_limit_enabled:
        return None
    limit = str(money_policy.daily_ip_reserve_limit_coins).strip()
    if not limit:
        return None
    return max(coins_to_atomic(limit, money_policy), 0)


def normalize_reserve_client_ip(value: str | None) -> str | None:
    raw = str(value or "").strip().lower()
    if not raw or raw == "unknown":
        return None
    try:
        return str(ipaddress.ip_address(raw))
    except ValueError:
        return raw


def reserve_client_ip_hash(value: str | None) -> str | None:
    normalized = normalize_reserve_client_ip(value)
    if normalized is None:
        return None
    return hashlib.sha256(f"ip:{normalized}".encode("utf-8")).hexdigest()


def reserve_client_ip_prefix_hash(value: str | None) -> str | None:
    normalized = normalize_reserve_client_ip(value)
    if normalized is None:
        return None
    try:
        ip = ipaddress.ip_address(normalized)
    except ValueError:
        return None
    if ip.version == 4:
        network = ipaddress.ip_network(f"{ip}/24", strict=False)
    else:
        network = ipaddress.ip_network(f"{ip}/64", strict=False)
    return hashlib.sha256(f"ip-prefix:{network}".encode("utf-8")).hexdigest()


def build_reserve_limit_identity_keys(
    wallet_id: str,
    *,
    reserve_client_ip: str | None = None,
    money_policy: MoneyPolicy | None = None,
) -> tuple[str, ...]:
    active_money_policy = money_policy or MoneyPolicy()
    keys = [f"wallet:{str(wallet_id or '').strip()}"]
    client_hash = reserve_client_ip_hash(reserve_client_ip)
    if (
        client_hash
        and _daily_ip_reserve_limit_atomic(active_money_policy) is not None
    ):
        keys.append(f"ip:{client_hash}")
        prefix_hash = reserve_client_ip_prefix_hash(reserve_client_ip)
        if prefix_hash:
            keys.append(f"ip-prefix:{prefix_hash}")
    return tuple(item for item in keys if item and item != "wallet:")


def _daily_reserve_limit_for_identity_key(
    identity_key: str,
    money_policy: MoneyPolicy,
) -> int | None:
    normalized = str(identity_key or "").strip()
    if normalized.startswith(("ip:", "ip-prefix:")):
        return _daily_ip_reserve_limit_atomic(money_policy)
    return _daily_user_reserve_limit_atomic(money_policy)


def _daily_reserve_spent_today_by_identity(
    wallet_id: str,
    *,
    reserve_limit_identity_keys: tuple[str, ...],
    wallet_policy: WalletPolicy | None = None,
) -> dict[str, int]:
    today = datetime.now(tz=UTC).date()
    identity_keys = tuple(reserve_limit_identity_keys) or (
        f"wallet:{str(wallet_id or '').strip()}",
    )
    return {
        identity_key: max(
            _daily_user_reserve_spent_today_from_chain(
                wallet_id,
                today=today,
                wallet_policy=wallet_policy,
                reserve_limit_identity_key=identity_key,
            ),
            _daily_user_reserve_spent_today_from_settlements(
                wallet_id,
                today=today,
                wallet_policy=wallet_policy,
                reserve_limit_identity_key=identity_key,
            ),
        )
        for identity_key in identity_keys
    }


def _daily_user_reserve_spent_today_atomic(
    wallet_id: str,
    *,
    money_policy: MoneyPolicy | None = None,
    wallet_policy: WalletPolicy | None = None,
    reserve_limit_identity_keys: tuple[str, ...] = (),
) -> int:
    spent_by_identity = _daily_reserve_spent_today_by_identity(
        wallet_id,
        reserve_limit_identity_keys=tuple(reserve_limit_identity_keys),
        wallet_policy=wallet_policy,
    )
    return max(spent_by_identity.values() or [0])


def _daily_user_reserve_spent_today_from_chain(
    wallet_id: str,
    *,
    today: date,
    wallet_policy: WalletPolicy | None,
    reserve_limit_identity_key: str,
) -> int:
    from .chain import list_chain_blocks

    spent_atomic = 0
    try:
        blocks = list_chain_blocks(wallet_policy)
    except (OSError, ValueError, json.JSONDecodeError):
        return 0

    for block in blocks:
        for tx in block.transactions:
            if tx.tx_type != "settlement_compute_reserve_debit":
                continue
            metadata = tx.metadata or {}
            identity_keys = _reserve_limit_identity_keys_from_metadata(
                metadata,
                fallback_wallet_id=str(metadata.get("source_wallet_id") or ""),
            )
            if reserve_limit_identity_key not in identity_keys:
                continue
            if not _is_same_utc_day(tx.created_at, today):
                continue
            spent_atomic += _reserve_compute_cost_from_metadata(
                metadata,
                fallback_atomic=abs(int(tx.delta_atomic or 0)),
            )
    return spent_atomic


def _daily_user_reserve_spent_today_from_settlements(
    wallet_id: str,
    *,
    today: date,
    wallet_policy: WalletPolicy | None,
    reserve_limit_identity_key: str,
) -> int:
    from .settlement import list_settlements

    spent_atomic = 0
    for item in list_settlements(wallet_policy):
        identity_keys = tuple(
            str(value)
            for value in getattr(item, "reserve_limit_identity_keys", []) or []
            if str(value).strip()
        ) or (f"wallet:{str(item.source_wallet_id or '').strip()}",)
        if reserve_limit_identity_key not in identity_keys:
            continue
        if str(item.funding_source).strip().lower() != FundingSource.RESERVE.value:
            continue
        if not _is_same_utc_day(item.created_at, today):
            continue
        spent_atomic += int(item.compute_cost_atomic)
    return spent_atomic


def _reserve_limit_identity_keys_from_metadata(
    metadata: dict,
    *,
    fallback_wallet_id: str,
) -> tuple[str, ...]:
    raw_keys = metadata.get("reserve_limit_identity_keys")
    if isinstance(raw_keys, list):
        keys = tuple(str(item) for item in raw_keys if str(item).strip())
        if keys:
            return keys
    wallet_id = str(fallback_wallet_id or "").strip()
    return (f"wallet:{wallet_id}",) if wallet_id else ()


def _reserve_compute_cost_from_metadata(
    metadata: dict,
    *,
    fallback_atomic: int,
) -> int:
    raw_value = metadata.get("compute_cost_atomic")
    try:
        return max(int(raw_value), 0)
    except (TypeError, ValueError):
        return max(int(fallback_atomic), 0)


def _is_same_utc_day(value: str | None, today: date) -> bool:
    try:
        created_at = datetime.fromisoformat(str(value or ""))
    except ValueError:
        return False
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return created_at.astimezone(UTC).date() == today
