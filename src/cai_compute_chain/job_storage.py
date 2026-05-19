# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .local_json_store import atomic_write_json_array_file, read_json_array_file
from .model import WalletPolicy
from .wallet import data_root


@dataclass
class JobIntent:
    job_id: str
    created_at: str
    source_wallet_id: str
    source_wallet_address: str
    model_id: str
    cai_url: str
    prompt: str
    payment_preference: str
    requested_compute_cost_atomic: int
    execution_cai_url: str | None = None
    requester_node_id: str | None = None
    pricing_mode: str = "manual"
    pricing_basis: str = "manual"
    pricing_reason: str | None = None
    reserved_prompt_tokens: int | None = None
    reserved_completion_tokens: int | None = None
    pricing_final_multiplier_bps: int | None = None
    pricing_floor_atomic: int | None = None
    pricing_cap_atomic: int | None = None
    pricing_input_token_price_atomic: int | None = None
    pricing_output_token_price_atomic: int | None = None
    reserve_limit_identity_keys: list[str] = field(default_factory=list)
    reserve_client_ip_hash: str | None = None
    status: str = "created"
    receipt_id: str | None = None
    settlement_id: str | None = None
    last_error: str | None = None
    execution_attempts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def CAI_url(self) -> str:
        return self.cai_url


@dataclass
class ExecutionReceipt:
    receipt_id: str
    created_at: str
    job_id: str
    cai_url: str
    model_id: str
    execution_model_id: str
    response_id: str | None
    finish_reason: str | None
    output_text: str
    instance_id: str | None
    worker_payouts: list[dict[str, Any]] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)
    pricing_mode: str = "manual"
    pricing_basis: str = "manual"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reserved_prompt_tokens: int | None = None
    reserved_completion_tokens: int | None = None
    reserved_compute_cost_atomic: int | None = None
    actual_compute_cost_atomic: int | None = None
    reservation_surplus_atomic: int = 0
    usage_priced: bool = False
    token_usage_source: str | None = None
    token_usage_audit: dict[str, Any] | None = None
    network_audit: dict[str, Any] | None = None

    @property
    def CAI_url(self) -> str:
        return self.cai_url


def job_intent_file_path(policy: WalletPolicy | None = None) -> Path:
    active_policy = policy or WalletPolicy()
    return data_root(active_policy) / active_policy.job_intent_file_name


def execution_receipt_file_path(policy: WalletPolicy | None = None) -> Path:
    active_policy = policy or WalletPolicy()
    return data_root(active_policy) / active_policy.execution_receipt_file_name


def _normalize_job_intent_payload(raw: dict[str, Any]) -> dict[str, Any]:
    payload = dict(raw)
    if "cai_url" not in payload and payload.get("CAI_url") is not None:
        payload["cai_url"] = payload.get("CAI_url")
    payload.pop("CAI_url", None)
    payload.setdefault("pricing_basis", "manual")
    payload.setdefault("reserved_prompt_tokens", None)
    payload.setdefault("reserved_completion_tokens", None)
    payload.setdefault("pricing_final_multiplier_bps", None)
    payload.setdefault("pricing_floor_atomic", None)
    payload.setdefault("pricing_cap_atomic", None)
    payload.setdefault("pricing_input_token_price_atomic", None)
    payload.setdefault("pricing_output_token_price_atomic", None)
    payload.setdefault("requester_node_id", None)
    payload.setdefault("reserve_limit_identity_keys", [])
    payload.setdefault("reserve_client_ip_hash", None)
    attempts = payload.get("execution_attempts")
    payload["execution_attempts"] = attempts if isinstance(attempts, list) else []
    return payload


def _normalize_execution_receipt_payload(raw: dict[str, Any]) -> dict[str, Any]:
    payload = dict(raw)
    if "cai_url" not in payload and payload.get("CAI_url") is not None:
        payload["cai_url"] = payload.get("CAI_url")
    payload.pop("CAI_url", None)
    payload.pop("execution_cai_url", None)
    payload.setdefault("pricing_mode", "manual")
    payload.setdefault("pricing_basis", "manual")
    payload.setdefault("prompt_tokens", None)
    payload.setdefault("completion_tokens", None)
    payload.setdefault("total_tokens", None)
    payload.setdefault("reserved_prompt_tokens", None)
    payload.setdefault("reserved_completion_tokens", None)
    payload.setdefault("reserved_compute_cost_atomic", None)
    payload.setdefault("actual_compute_cost_atomic", None)
    payload.setdefault("reservation_surplus_atomic", 0)
    payload.setdefault("usage_priced", False)
    payload.setdefault("token_usage_source", None)
    payload.setdefault("token_usage_audit", None)
    return payload


def list_job_intents(policy: WalletPolicy | None = None) -> list[JobIntent]:
    path = job_intent_file_path(policy)
    if not path.exists():
        return []
    raw = read_json_array_file(path, heal_corrupt=True)
    items = [JobIntent(**_normalize_job_intent_payload(item)) for item in raw]
    items.sort(key=lambda item: item.created_at, reverse=True)
    return items


def save_job_intents(items: list[JobIntent], policy: WalletPolicy | None = None) -> None:
    path = job_intent_file_path(policy)
    atomic_write_json_array_file(
        path,
        [asdict(item) for item in items],
    )


def list_execution_receipts(
    policy: WalletPolicy | None = None,
) -> list[ExecutionReceipt]:
    path = execution_receipt_file_path(policy)
    if not path.exists():
        return []
    raw = read_json_array_file(path, heal_corrupt=True)
    items = [
        ExecutionReceipt(
            **{
                **_normalize_execution_receipt_payload(item),
                "execution_model_id": item.get("execution_model_id", item["model_id"]),
                "instance_id": item.get("instance_id"),
                "worker_payouts": item.get("worker_payouts") or [],
                "network_audit": item.get("network_audit"),
            }
        )
        for item in raw
    ]
    items.sort(key=lambda item: item.created_at, reverse=True)
    return items


def save_execution_receipts(
    items: list[ExecutionReceipt], policy: WalletPolicy | None = None
) -> None:
    path = execution_receipt_file_path(policy)
    atomic_write_json_array_file(
        path,
        [asdict(item) for item in items],
    )


def update_execution_receipt(
    receipt: ExecutionReceipt,
    policy: WalletPolicy | None = None,
) -> None:
    items = list_execution_receipts(policy)
    for index, existing in enumerate(items):
        if existing.receipt_id == receipt.receipt_id:
            items[index] = receipt
            save_execution_receipts(items, policy)
            return
    items.append(receipt)
    save_execution_receipts(items, policy)


def resolve_job_intent(job_id: str, policy: WalletPolicy | None = None) -> JobIntent | None:
    for item in list_job_intents(policy):
        if item.job_id == job_id:
            return item
    return None


def update_job_intent(job: JobIntent, policy: WalletPolicy | None = None) -> None:
    items = list_job_intents(policy)
    for index, item in enumerate(items):
        if item.job_id == job.job_id:
            items[index] = job
            save_job_intents(items, policy)
            return
    raise ValueError(f"Job intent '{job.job_id}' not found.")
