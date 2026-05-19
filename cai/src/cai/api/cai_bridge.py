# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
import importlib
import json
import os
import secrets
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


class _CaiModules:
    def __init__(self) -> None:
        _ensure_cai_on_path()
        self.chain = importlib.import_module("cai_compute_chain.chain")
        self.cai_owned_diagnostics = importlib.import_module(
            "cai_compute_chain.cai_owned_diagnostics"
        )
        self.decentralized_compute = importlib.import_module(
            "cai_compute_chain.decentralized_compute"
        )
        self.ui_state = importlib.import_module("cai_compute_chain.ui_state")
        self.jobs = importlib.import_module("cai_compute_chain.jobs")
        self.model = importlib.import_module("cai_compute_chain.model")
        self.model_distribution = importlib.import_module(
            "cai_compute_chain.model_distribution"
        )
        self.node_capabilities = importlib.import_module(
            "cai_compute_chain.node_capabilities"
        )
        self.node_config = importlib.import_module("cai_compute_chain.node_config")
        self.peer_payload = importlib.import_module("cai_compute_chain.peer_payload")
        self.route_health = importlib.import_module("cai_compute_chain.route_health")
        self.settlement = importlib.import_module("cai_compute_chain.settlement")
        self.update_channel = importlib.import_module("cai_compute_chain.update_channel")
        self.validators = importlib.import_module("cai_compute_chain.validators")
        self.wallet = importlib.import_module("cai_compute_chain.wallet")
        self.wallet_signing = importlib.import_module(
            "cai_compute_chain.wallet_signing"
        )
        self.worker_capability_attestations = importlib.import_module(
            "cai_compute_chain.worker_capability_attestations"
        )


class CaiBridgeService:
    def __init__(
        self,
        *,
        state_url: str,
        cai_url: str | None = None,
        execution_cai_url: str | None = None,
        local_node_id: str | None = None,
        CAI_url: str | None = None,
    ) -> None:
        resolved_cai_url = str(cai_url or CAI_url or "").rstrip("/")
        if not resolved_cai_url:
            raise ValueError("CAI service requires a CAI base URL.")
        self.state_url = state_url
        self.cai_url = resolved_cai_url
        self.execution_cai_url = str(execution_cai_url or resolved_cai_url).rstrip("/")
        self.local_node_id = local_node_id
        self.modules = _CaiModules()
        self.money_policy = self.modules.model.MoneyPolicy()
        self.network_config = self.modules.model.CaiNetworkConfig()
        self.network_model_policy = self.modules.model.NetworkModelPolicy()
        self.wallet_policy = self.modules.model.WalletPolicy()

    @property
    def CAI_url(self) -> str:
        return self.cai_url

    def summary(self) -> dict[str, Any]:
        maintenance_errors: list[dict[str, str]] = []
        maintenance_results: dict[str, Any] = {}
        try:
            state_payload = _load_state_payload(self.state_url)
        except Exception as exc:
            state_payload = None
            maintenance_errors.append(_operation_error_payload("load_state_payload", exc))

        try:
            repair_result = self.modules.jobs.repair_local_worker_reward_state(
                cai_url=self.cai_url,
                money_policy=self.money_policy,
                wallet_policy=self.wallet_policy,
                state_payload=state_payload,
                timeout_sec=3,
            )
            if repair_result is not None:
                maintenance_results["repairLocalWorkerRewardState"] = repair_result
        except Exception as exc:
            maintenance_errors.append(
                _operation_error_payload("repair_local_worker_reward_state", exc)
            )

        try:
            self.modules.chain.ensure_chain_genesis(
                policy=self.wallet_policy,
                money_policy=self.money_policy,
            )
        except Exception as exc:
            maintenance_errors.append(_operation_error_payload("ensure_chain_genesis", exc))

        active_wallet = self.modules.wallet.get_active_wallet(self.wallet_policy)
        snapshot = self.modules.ui_state.build_interface_snapshot(
            state_url=self.state_url,
            cai_url=self.cai_url,
            money_policy=self.money_policy,
            wallet_policy=self.wallet_policy,
            network_config=self.network_config,
            network_model_policy=self.network_model_policy,
        ).to_dict()
        try:
            self.modules.jobs.reconcile_stale_running_job_intents(self.wallet_policy)
        except Exception as exc:
            maintenance_errors.append(
                _operation_error_payload("reconcile_stale_running_job_intents", exc)
            )
        to_coins = self.modules.wallet.atomic_to_coins
        active_wallet_id = getattr(active_wallet, "wallet_id", None)
        job_items = self.modules.jobs.list_job_intents(self.wallet_policy)
        if active_wallet_id:
            job_items = [item for item in job_items if item.source_wallet_id == active_wallet_id]
        receipt_items = self.modules.jobs.list_execution_receipts(self.wallet_policy)
        receipt_by_id = {
            getattr(item, "receipt_id", None): item for item in receipt_items if getattr(item, "receipt_id", None)
        }
        journal_summaries = _wallet_activity_history_summaries(
            self.modules,
            active_wallet=active_wallet,
            active_wallet_id=active_wallet_id,
            wallet_policy=self.wallet_policy,
            money_policy=self.money_policy,
            to_coins=to_coins,
            limit=8,
        )
        payout_items = self.modules.settlement.list_worker_payouts(limit=6, policy=self.wallet_policy)
        settlement_items = self.modules.settlement.list_settlements(policy=self.wallet_policy)
        latest_job = job_items[0] if job_items else None
        latest_receipt = receipt_items[0] if receipt_items else None
        latest_payout = payout_items[0] if payout_items else None
        latest_settlement = _resolve_latest_settlement_for_job(
            settlement_items,
            latest_job,
        )

        local_runtime_node_id = (
            _resolve_local_runtime_node_id(
                state_payload=state_payload,
                local_node_id=self.local_node_id,
            )
            if isinstance(state_payload, dict)
            else self.local_node_id
        )
        worker_summary = dict(snapshot["worker"])
        resource_summary = _local_node_resource_summary(
            state_payload,
            local_runtime_node_id,
        )
        if resource_summary:
            worker_summary["resources"] = resource_summary
            worker_summary["resourceSummary"] = resource_summary
        model_shard_inventory = _build_local_model_shard_inventory(
            state_payload,
            local_runtime_node_id,
            model_distribution=self.modules.model_distribution,
            wallet_policy=self.wallet_policy,
        )
        if model_shard_inventory:
            worker_summary["model_shard_inventory"] = model_shard_inventory
            worker_summary["modelShardInventory"] = model_shard_inventory
        readiness_summary = _local_node_readiness_summary(
            state_payload,
            local_runtime_node_id,
        )
        if readiness_summary:
            worker_summary["readiness"] = readiness_summary
            cai_owned_transport = readiness_summary.get("caiOwnedTransport")
            if isinstance(cai_owned_transport, dict):
                worker_summary["caiOwnedTransport"] = dict(cai_owned_transport)
                worker_summary["cai_owned_transport"] = dict(cai_owned_transport)
        try:
            diagnostics = self.modules.cai_owned_diagnostics
            runtime_queue = diagnostics.build_cai_owned_worker_runtime_queue_snapshot(
                local_node_id=local_runtime_node_id,
                max_records=20,
                policy=self.wallet_policy,
            )
        except Exception as exc:  # noqa: BLE001
            runtime_queue = {
                "localNodeId": local_runtime_node_id,
                "ready": False,
                "reason": str(exc),
                "statusCounts": {},
                "recordCount": 0,
                "receivedCount": 0,
                "processingCount": 0,
                "processedCount": 0,
                "failedCount": 0,
                "timedOutCount": 0,
                "deliveredCount": 0,
                "currentBatch": None,
                "lastError": str(exc),
                "records": [],
            }
        worker_summary["runtime_queue"] = runtime_queue
        worker_summary["runtimeQueue"] = runtime_queue
        worker_reward_address = self._local_worker_reward_address()
        if worker_reward_address:
            worker_summary["worker_reward_address"] = worker_reward_address

        try:
            update_summary = self.modules.update_channel.build_local_update_summary()
        except Exception as exc:  # noqa: BLE001
            update_summary = {
                "runtime": {
                    "version": getattr(self.modules.update_channel, "__version__", None),
                    "gitCommit": None,
                    "gitBranch": None,
                    "gitDirty": False,
                },
                "updates": {
                    "autoUpdateEnabled": self.modules.update_channel.auto_update_enabled(),
                    "channel": None,
                    "provider": None,
                    "repository": None,
                    "targetBranch": None,
                    "sourceUrl": None,
                    "baseUrl": None,
                    "checked": False,
                    "checkedAt": None,
                    "lastUpdatedAt": None,
                    "updated": False,
                    "updateAvailable": False,
                    "remoteGitCommit": None,
                    "remoteGitBranch": None,
                    "remoteVersion": None,
                    "status": "error",
                    "phase": "error",
                    "progress": 0,
                    "message": str(exc),
                    "canApply": False,
                    "applyReason": str(exc),
                    "canCancel": False,
                    "cancelRequested": False,
                    "restartScheduled": False,
                    "restartRequired": False,
                    "dashboardBuildStatus": None,
                    "dashboardBuildMessage": None,
                },
            }
        chain_state = self.modules.chain.chain_summary(self.wallet_policy)
        chain_status = snapshot.get("chain") or {
            "network": chain_state.get("network"),
            "block_count": chain_state.get("blockCount"),
            "transaction_count": chain_state.get("transactionCount"),
            "tip_height": chain_state.get("tipHeight"),
            "tip_hash": chain_state.get("tipHash"),
            "finalized_height": chain_state.get("finalizedHeight"),
            "last_sync_at": chain_state.get("lastSyncAt"),
            "valid": chain_state.get("valid"),
        }
        safety_summary = _build_safety_summary(
            money_policy=self.money_policy,
            chain_status=chain_status,
            validator_summary=snapshot.get("validator") or {},
        )

        return {
            "available": True,
            "currency": {
                "code": self.money_policy.currency_code,
                "name": self.money_policy.currency_name,
                "decimals": self.money_policy.decimals,
            },
            "networkConfig": {
                "chainNetwork": self.money_policy.chain_network.value,
                "namespace": self.network_config.namespace,
                "bootstrapPeers": list(self.network_config.bootstrap_peers),
                "defaultApiPort": self.network_config.default_api_port,
                "defaultLibp2pPort": self.network_config.default_libp2p_port,
            },
            "economics": {
                "rewardTokenCode": self.money_policy.reward_token_code,
                "dailyUserReserveLimitEnabled": self.money_policy.daily_user_reserve_limit_enabled,
                "dailyUserReserveLimitCoins": self.money_policy.daily_user_reserve_limit_coins,
                "dailyIpReserveLimitEnabled": self.money_policy.daily_ip_reserve_limit_enabled,
                "dailyIpReserveLimitCoins": self.money_policy.daily_ip_reserve_limit_coins,
                "automaticTokenPricingEnabled": self.money_policy.automatic_token_pricing_enabled,
                "inputTokenPriceCoins": self.money_policy.automatic_price_per_input_token_coins,
                "outputTokenPriceCoins": self.money_policy.automatic_price_per_output_token_coins,
                "defaultReservedOutputTokens": self.money_policy.automatic_price_default_reserved_output_tokens,
                "developerTreasuryWalletId": self.money_policy.developer_treasury_wallet_id,
                "developerTreasuryAddress": self.money_policy.developer_treasury_address,
                "developerTreasuryCoins": str(self.money_policy.developer_treasury_coins),
                "aiDevelopmentWalletId": self.money_policy.ai_development_wallet_id,
                "aiDevelopmentAddress": self.money_policy.ai_development_address,
                "aiDevelopmentFeeBps": self.money_policy.ai_development_fee_bps,
            },
            "runtime": update_summary.get("runtime") or {},
            "updates": update_summary.get("updates") or {},
            "wallet": snapshot["wallet"],
            "wallets": self.list_wallet_rows(),
            "chainStatus": chain_status,
            "safety": safety_summary,
            "validator": snapshot["validator"],
            "validatorSet": self.validator_set(),
            "chain": chain_state,
            "worker": worker_summary,
            "reward": snapshot.get("reward", {}),
            "compute": snapshot["compute"],
            "diagnostics": {
                "maintenanceStatus": "degraded" if maintenance_errors else "ok",
                "statePayloadAvailable": isinstance(state_payload, dict),
                "maintenanceErrors": maintenance_errors,
                "maintenanceResults": maintenance_results,
            },
            "history": {
                "journal": journal_summaries,
                "jobs": [
                    _job_to_history_summary(item, receipt_by_id.get(getattr(item, "receipt_id", None)))
                    for item in job_items[:6]
                ],
                "payouts": [_payout_to_summary(item, self.money_policy, to_coins) for item in payout_items],
                "settlements": [
                    _settlement_to_summary(
                        item,
                        self.money_policy,
                        to_coins,
                        chain_transactions=_settlement_chain_history_from_modules(
                            self.modules,
                            getattr(item, "settlement_id", None),
                            self.wallet_policy,
                        ),
                    )
                    for item in settlement_items[:6]
                ],
            },
            "latestJob": _job_to_summary(latest_job),
            "latestReceipt": _receipt_to_summary(latest_receipt),
            "latestSettlement": _settlement_to_summary(
                latest_settlement,
                self.money_policy,
                to_coins,
                chain_transactions=_settlement_chain_history_from_modules(
                    self.modules,
                    getattr(latest_settlement, "settlement_id", None),
                    self.wallet_policy,
                ),
            ),
            "latestPayout": _payout_to_summary(latest_payout, self.money_policy, to_coins),
        }

    def cancel_update(self) -> dict[str, Any]:
        return self.modules.update_channel.cancel_pending_portable_update()

    def validator_set(self) -> dict[str, Any]:
        try:
            self.modules.node_config.refresh_validator_ha_lease(
                state_payload=_load_state_payload(self.state_url),
                cai_url=self.cai_url,
                policy=self.wallet_policy,
                allow_failover=True,
            )
        except Exception:
            pass
        committee = self.modules.validators.build_validator_committee_snapshot(
            self.wallet_policy
        )
        records = self.modules.validators.list_validator_records(self.wallet_policy)
        payload = {
            "validators": [
                {
                    "validatorId": item.validator_id,
                    "address": item.address,
                    "state": item.state,
                    "bondedAtomic": item.bonded_atomic,
                    "bondedCoins": self.modules.wallet.atomic_to_coins(
                        item.bonded_atomic, self.money_policy
                    ),
                    "staticIpConfirmed": item.static_ip_confirmed,
                    "nodeId": item.current_node_id,
                    "apiHost": item.advertised_api_host,
                    "dataHost": item.advertised_data_host,
                    "haEnabled": getattr(item, "ha_enabled", False),
                    "activeReplicaNodeId": getattr(item, "active_replica_node_id", None),
                    "activeReplicaLeaseUntil": getattr(
                        item, "active_replica_lease_until", None
                    ),
                    "replicaNodeIds": list(getattr(item, "replica_node_ids", []) or []),
                    "unbondingStartedAt": getattr(item, "unbonding_started_at", None),
                    "unbondingAvailableAt": getattr(item, "unbonding_available_at", None),
                    "jailedAt": getattr(item, "jailed_at", None),
                    "unjailAvailableAt": getattr(item, "unjail_available_at", None),
                    "source": getattr(item, "source", "local"),
                    "sourceUrl": getattr(item, "source_url", None),
                    "lastSlashCoins": self.modules.wallet.atomic_to_coins(
                        getattr(item, "last_slash_atomic", 0), self.money_policy
                    ),
                    "totalSlashedCoins": self.modules.wallet.atomic_to_coins(
                        getattr(item, "total_slashed_atomic", 0), self.money_policy
                    ),
                    "updatedAt": getattr(item, "updated_at", None),
                }
                for item in records
            ],
            "committee": {
                "validatorIds": list(committee.validator_ids),
                "totalBondedAtomic": committee.total_bonded_atomic,
                "totalBondedCoins": self.modules.wallet.atomic_to_coins(
                    committee.total_bonded_atomic, self.money_policy
                ),
                "quorumBondAtomic": committee.quorum_bond_atomic,
                "quorumBondCoins": self.modules.wallet.atomic_to_coins(
                    committee.quorum_bond_atomic, self.money_policy
                ),
            },
        }
        payload = self.modules.peer_payload.add_peer_payload_metadata(
            payload,
            policy=self.wallet_policy,
        )
        return self._sign_peer_payload(payload)

    def chain(self) -> dict[str, Any]:
        payload = self.modules.chain.export_chain_payload(self.wallet_policy)
        payload = self.modules.peer_payload.add_peer_payload_metadata(
            payload,
            policy=self.wallet_policy,
        )
        return self._sign_peer_payload(payload)

    def sync_chain(self, payload: dict[str, Any]) -> dict[str, Any]:
        if isinstance(payload, dict):
            self.modules.peer_payload.validate_peer_payload_network(
                payload,
                policy=self.wallet_policy,
                payload_name="chain sync",
            )
            signature_ok, signature_error = (
                self.modules.peer_payload.verify_peer_payload_signature(
                    payload,
                    payload_name="chain sync",
                    require_signature=(
                        self.modules.peer_payload.peer_payload_signatures_required()
                    ),
                )
            )
            if not signature_ok:
                raise ValueError(
                    signature_error or "Invalid chain sync payload signature."
                )
        raw_chain = (
            payload.get("chain")
            if isinstance(payload, dict) and isinstance(payload.get("chain"), dict)
            else payload
        )
        if isinstance(raw_chain, dict):
            self.modules.peer_payload.validate_peer_payload_network(
                raw_chain,
                policy=self.wallet_policy,
                payload_name="chain sync",
            )
        imported_blocks, imported_transactions = (
            self.modules.chain.merge_remote_chain_payload(
                payload,
                policy=self.wallet_policy,
            )
        )
        return {
            "message": "Chain sync completed.",
            "importedBlocks": imported_blocks,
            "importedTransactions": imported_transactions,
            "chain": self.modules.chain.chain_summary(self.wallet_policy),
        }

    def _sign_peer_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            wallet = self.modules.wallet.get_active_wallet(self.wallet_policy)
        except Exception as exc:
            wallet = None
            wallet_error = exc
        else:
            wallet_error = None
        if wallet_error is not None:
            return _unsigned_peer_payload(
                payload,
                reason="active_wallet_unavailable",
                exc=wallet_error,
            )
        if wallet is None:
            return _unsigned_peer_payload(payload, reason="no_active_wallet")
        try:
            signer = self.modules.wallet.load_unlocked_wallet_signing_material(
                wallet,
                self.wallet_policy,
            )
        except Exception as exc:
            signer = None
            signer_error = exc
        else:
            signer_error = None
        if signer_error is not None:
            return _unsigned_peer_payload(
                payload,
                reason="signing_material_unavailable",
                exc=signer_error,
            )
        if not signer:
            return _unsigned_peer_payload(payload, reason="wallet_locked")
        try:
            return self.modules.peer_payload.sign_peer_payload(
                payload,
                public_key_b64=str(signer.get("public_key_b64") or ""),
                signing_seed_b64=str(signer.get("signing_seed_b64") or ""),
                pq_public_key_b64=str(signer.get("pq_public_key_b64") or ""),
                pq_private_key_b64=str(signer.get("pq_private_key_b64") or ""),
                signer_wallet_id=getattr(wallet, "wallet_id", None),
                signer_address=getattr(wallet, "address", None),
            )
        except Exception as exc:
            return _unsigned_peer_payload(
                payload,
                reason="signing_failed",
                exc=exc,
            )

    def sync_validator_set(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if isinstance(payload, dict) and (
            isinstance(payload.get("validatorSet"), dict)
            or isinstance(payload.get("records"), list)
            or isinstance(payload.get("validators"), list)
        ):
            validator_payload = (
                payload.get("validatorSet")
                if isinstance(payload.get("validatorSet"), dict)
                else payload
            )
            source_url = str(
                payload.get("sourceUrl")
                or payload.get("source_url")
                or payload.get("caiUrl")
                or payload.get("cai_url")
                or self.cai_url
            ).strip()
            imported_records = self.modules.validators.merge_remote_validator_set_payload(
                validator_payload,
                source_url=source_url,
                policy=self.wallet_policy,
            )
            return {
                "message": "Validator set payload imported.",
                "attemptedPeers": 1,
                "successfulPeers": 1,
                "importedRecords": imported_records,
                "peerUrls": [source_url],
                "failedPeers": 0,
                "failedPeerUrls": [],
                "peerErrors": [],
                "validatorSet": self.validator_set(),
            }

        state_payload = _load_state_payload(self.state_url)
        result = self.modules.validators.sync_validator_set_from_cai_peers(
            state_payload=state_payload,
            cai_url=self.cai_url,
            policy=self.wallet_policy,
            local_node_id=getattr(self, "local_node_id", None),
        )
        return {
            "message": "Validator set sync completed.",
            **_peer_sync_result_summary(result),
            "importedRecords": result.imported_records,
            "validatorSet": self.validator_set(),
        }

    def validator_evidence(self) -> dict[str, Any]:
        payload = {
            "exportedAt": datetime.now(tz=UTC).isoformat(),
            "evidence": [
                {
                    "evidenceId": item.evidence_id,
                    "createdAt": item.created_at,
                    "validatorId": item.validator_id,
                    "reporterValidatorId": getattr(
                        item, "reporter_validator_id", None
                    ),
                    "evidenceType": item.evidence_type,
                    "settlementId": item.settlement_id,
                    "attestationId": item.attestation_id,
                    "conflictingAttestationId": item.conflicting_attestation_id,
                    "slashAtomic": item.slash_atomic,
                    "slashCoins": self.modules.wallet.atomic_to_coins(
                        item.slash_atomic, self.money_policy
                    ),
                    "jailed": item.jailed,
                    "note": item.note,
                    "source": getattr(item, "source", "local"),
                    "sourceUrl": getattr(item, "source_url", None),
                    "lastSeenAt": getattr(item, "last_seen_at", None),
                    "updatedAt": getattr(item, "updated_at", None),
                    "appliedToRegistry": getattr(item, "applied_to_registry", False),
                }
                for item in self.modules.settlement.list_validator_evidence(
                    policy=self.wallet_policy
                )
            ],
        }
        payload = self.modules.peer_payload.add_peer_payload_metadata(
            payload,
            policy=self.wallet_policy,
        )
        return self._sign_peer_payload(payload)

    def sync_validator_evidence(self) -> dict[str, Any]:
        state_payload = _load_state_payload(self.state_url)
        result = self.modules.settlement.sync_validator_evidence_from_cai_peers(
            state_payload=state_payload,
            cai_url=self.cai_url,
            policy=self.wallet_policy,
            local_node_id=getattr(self, "local_node_id", None),
        )
        return {
            "message": "Validator evidence sync completed.",
            **_peer_sync_result_summary(result),
            "importedRecords": result.imported_records,
            "appliedRecords": result.applied_records,
            "validatorSetSyncError": getattr(
                result, "validator_set_sync_error", None
            ),
            "penaltyAttestationSyncError": getattr(
                result, "penalty_attestation_sync_error", None
            ),
            "validatorSet": self.validator_set(),
            "evidence": self.validator_evidence(),
        }

    def chunk_inventory(
        self,
        *,
        source_kind: str = "peer_cache",
        endpoint_base_url: str | None = None,
    ) -> dict[str, Any]:
        source_id = str(
            getattr(self, "local_node_id", None) or socket.gethostname()
        ).strip()
        payload = self.modules.model_distribution.export_chunk_inventory_payload(
            source_id=source_id,
            source_kind=source_kind,
            endpoint_base_url=endpoint_base_url or self.cai_url,
            policy=self.wallet_policy,
        )
        return payload.to_dict()

    def sync_chunk_inventory(self, *, source_kind: str = "peer_cache") -> dict[str, Any]:
        state_payload = _load_state_payload(self.state_url)
        result = self.modules.model_distribution.sync_chunk_inventory_from_cai_peers(
            state_payload=state_payload,
            cai_url=self.cai_url,
            source_kind=source_kind,
            policy=self.wallet_policy,
            local_node_id=getattr(self, "local_node_id", None),
        )
        return {
            "message": "Chunk inventory sync completed.",
            "sourceKind": source_kind,
            **_peer_sync_result_summary(result),
            "importedPayloads": result.imported_payloads,
            "prunedPayloads": result.pruned_payloads,
        }

    def node_capabilities(self) -> dict[str, Any]:
        state_payload = _load_state_payload(self.state_url)
        payload = self.modules.node_capabilities.export_node_capabilities_payload(
            state_payload=state_payload,
            cai_url=self.cai_url,
            local_node_id=getattr(self, "local_node_id", None),
            policy=self.wallet_policy,
        )
        payload = self.modules.peer_payload.add_peer_payload_metadata(
            payload,
            policy=self.wallet_policy,
        )
        self.modules.node_capabilities.refresh_local_node_capabilities(
            state_payload=state_payload,
            cai_url=self.cai_url,
            local_node_id=getattr(self, "local_node_id", None),
            policy=self.wallet_policy,
        )
        return self._sign_peer_payload(payload)

    def sync_node_capabilities(self) -> dict[str, Any]:
        state_payload = _load_state_payload(self.state_url)
        self.modules.node_capabilities.refresh_local_node_capabilities(
            state_payload=state_payload,
            cai_url=self.cai_url,
            local_node_id=getattr(self, "local_node_id", None),
            policy=self.wallet_policy,
        )
        result = self.modules.node_capabilities.sync_node_capabilities_from_cai_peers(
            state_payload=state_payload,
            cai_url=self.cai_url,
            policy=self.wallet_policy,
            local_node_id=getattr(self, "local_node_id", None),
        )
        records = self.modules.node_capabilities.list_node_capabilities(
            self.wallet_policy
        )
        return {
            "message": "Node capability sync completed.",
            **_peer_sync_result_summary(result),
            "importedRecords": result.imported_records,
            "prunedRecords": result.pruned_records,
            "convergenceStatus": getattr(result, "convergence_status", "unknown"),
            "convergenceRepairRecommended": bool(
                getattr(result, "convergence_repair_recommended", False)
            ),
            "convergenceRepairActions": list(
                getattr(result, "convergence_repair_actions", [])
            ),
            "convergenceAudit": dict(getattr(result, "convergence_audit", {}) or {}),
            "recordCount": len(records),
            "records": [getattr(item, "__dict__", item) for item in records],
        }

    def worker_capability_attestations(self) -> dict[str, Any]:
        payload = (
            self.modules.worker_capability_attestations.export_worker_capability_attestations_payload(
                self.wallet_policy
            )
        )
        return self._sign_peer_payload(payload)

    def sync_worker_capability_attestations(self, payload: dict[str, Any]) -> dict[str, Any]:
        imported = (
            self.modules.worker_capability_attestations.merge_remote_worker_capability_attestations_payload(
                payload,
                source_url=str(payload.get("sourceUrl") or payload.get("source_url") or ""),
                policy=self.wallet_policy,
            )
        )
        records = (
            self.modules.worker_capability_attestations.list_worker_capability_attestations(
                policy=self.wallet_policy
            )
        )
        return {
            "message": "Worker capability attestation sync completed.",
            "importedRecords": imported,
            "recordCount": len(records),
            "records": [getattr(item, "__dict__", item) for item in records],
        }

    def worker_capability_challenge(self, payload: dict[str, Any]) -> dict[str, Any]:
        challenge = payload.get("challenge") if isinstance(payload, dict) else None
        if not isinstance(challenge, dict):
            challenge = payload
        if not isinstance(challenge, dict):
            raise ValueError("Worker capability challenge payload is required.")

        ok, error = (
            self.modules.worker_capability_attestations.verify_worker_capability_challenge(
                challenge,
                policy=self.wallet_policy,
                require_bonded_validator=False,
            )
        )
        if not ok:
            raise ValueError(error or "Worker capability challenge is invalid.")

        state_payload = _load_state_payload(self.state_url)
        records = self.modules.node_capabilities.refresh_local_node_capabilities(
            state_payload=state_payload,
            cai_url=self.cai_url,
            local_node_id=getattr(self, "local_node_id", None),
            policy=self.wallet_policy,
        )
        requested_node_id = str(
            challenge.get("worker_node_id") or challenge.get("workerNodeId") or ""
        ).strip()
        record = next(
            (
                item
                for item in records
                if str(getattr(item, "node_id", "") or "").strip()
                == requested_node_id
            ),
            None,
        )
        if record is None:
            raise ValueError("Local worker capability record was not found.")
        if not bool(getattr(record, "worker_enabled", False)):
            raise ValueError("Local worker mode is not enabled.")

        active_wallet = self.modules.wallet.get_active_wallet(self.wallet_policy)
        signer = (
            self.modules.wallet.load_unlocked_wallet_signing_material(
                active_wallet,
                self.wallet_policy,
            )
            if active_wallet is not None
            else None
        )
        if not signer:
            raise ValueError("Worker wallet must be unlocked to answer challenge.")

        _prepare_record_for_worker_challenge(
            record,
            signer=signer,
            active_wallet=active_wallet,
            modules=self.modules,
        )
        capability_fingerprint = (
            self.modules.worker_capability_attestations.worker_capability_fingerprint_from_record(
                record
            )
        )
        expected_fingerprint = str(
            challenge.get("capability_fingerprint")
            or challenge.get("capabilityFingerprint")
            or ""
        ).strip()
        if capability_fingerprint != expected_fingerprint:
            raise ValueError(
                "Local worker capability no longer matches validator challenge."
            )

        receipt = (
            self.modules.worker_capability_attestations.create_worker_capability_challenge_receipt(
                record,
                challenge=challenge,
                worker_public_key_b64=str(signer.get("public_key_b64") or ""),
                worker_signing_seed_b64=str(signer.get("signing_seed_b64") or ""),
                worker_pq_public_key_b64=str(signer.get("pq_public_key_b64") or ""),
                worker_pq_private_key_b64=str(signer.get("pq_private_key_b64") or ""),
            )
        )
        return {
            "accepted": True,
            "challengeId": challenge.get("challenge_id")
            or challenge.get("challengeId"),
            "workerNodeId": getattr(record, "node_id", None),
            "receipt": receipt,
        }

    def route_health(self) -> dict[str, Any]:
        records = self.modules.route_health.list_route_health_records(
            self.wallet_policy
        )
        return {
            "recordCount": len(records),
            "records": [getattr(item, "__dict__", item) for item in records],
        }

    def compute_cells(self) -> dict[str, Any]:
        state_payload = _load_state_payload(self.state_url)
        records = self.modules.route_health.list_route_health_records(
            self.wallet_policy
        )
        worker_node_ids = _compute_cell_worker_node_ids(state_payload)
        cells: list[dict[str, Any]] = []
        profile_fn = getattr(
            self.modules.route_health,
            "llama_cpp_compute_cell_profile_for_path",
            None,
        )
        planner_fn = getattr(
            getattr(self.modules, "decentralized_compute", None),
            "plan_llama_cpp_distributed_execution",
            None,
        )
        if callable(profile_fn):
            for source_node_id in worker_node_ids:
                sink_node_ids = [
                    node_id for node_id in worker_node_ids if node_id != source_node_id
                ]
                if not sink_node_ids:
                    continue
                execution_strategy: dict[str, Any] | None = None
                if callable(planner_fn):
                    execution_strategy = planner_fn(
                        source_node_id,
                        sink_node_ids,
                        records,
                    )
                    profile = (
                        execution_strategy.get("computeCellProfile")
                        if isinstance(execution_strategy, dict)
                        else None
                    )
                else:
                    profile = profile_fn(source_node_id, sink_node_ids, records)
                if not isinstance(profile, dict):
                    continue
                cell = {
                    "sourceNodeId": source_node_id,
                    "sinkNodeIds": sink_node_ids,
                    **profile,
                }
                if isinstance(execution_strategy, dict):
                    cell.update(
                        {
                            "executionMode": execution_strategy.get("executionMode"),
                            "standardLlamaCppRpcReady": execution_strategy.get(
                                "standardLlamaCppRpcReady"
                            ),
                            "requiresCaiOwnedTransport": execution_strategy.get(
                                "requiresCaiOwnedTransport"
                            ),
                            "caiOwnedTransport": execution_strategy.get(
                                "caiOwnedTransport"
                            ),
                            "executionStrategy": execution_strategy,
                        }
                    )
                cells.append(cell)
        ready_cells = [
            item for item in cells if bool(item.get("readyForLlamaCppRpc"))
        ]
        cai_owned_transport_required_cells = [
            item for item in cells if bool(item.get("requiresCaiOwnedTransport"))
        ]
        return {
            "workerNodeCount": len(worker_node_ids),
            "workerNodeIds": worker_node_ids,
            "cellCount": len(cells),
            "readyCellCount": len(ready_cells),
            "caiOwnedTransportRequiredCellCount": len(
                cai_owned_transport_required_cells
            ),
            "cells": cells,
        }

    def cai_owned_transport_sessions(self) -> dict[str, Any]:
        records = self.modules.decentralized_compute.list_cai_owned_transport_sessions(
            self.wallet_policy
        )
        to_dict = self.modules.decentralized_compute.cai_owned_transport_session_to_dict
        return {
            "sessionCount": len(records),
            "sessions": [to_dict(item) for item in records],
        }

    def cai_owned_transport_batch_inbox(
        self,
        *,
        node_id: str | None = None,
        status: str | None = "received",
    ) -> dict[str, Any]:
        local_node_id = str(node_id or self.local_node_id or "").strip()
        batches = self.modules.decentralized_compute.list_cai_owned_transport_batch_inbox(
            local_node_id,
            status=status,
            policy=self.wallet_policy,
        )
        return {
            "nodeId": local_node_id,
            "status": status,
            "batchCount": len(batches),
            "batches": batches,
        }

    def claim_next_cai_owned_transport_batch(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            raise ValueError(
                "CAI-owned transport claim-next payload must be an object."
            )
        return self.modules.decentralized_compute.claim_next_cai_owned_transport_batch(
            str(payload.get("nodeId") or payload.get("node_id") or "")
            or self.local_node_id,
            status=payload.get("status", "received"),
            session_id=payload.get("sessionId") or payload.get("session_id"),
            runtime_id=payload.get("runtimeId") or payload.get("runtime_id"),
            lease_seconds=_optional_float(
                payload.get("leaseSeconds")
                if "leaseSeconds" in payload
                else payload.get("lease_seconds")
            ),
            policy=self.wallet_policy,
        )

    def create_cai_owned_transport_session(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("CAI-owned transport session payload must be an object.")
        record = self.modules.decentralized_compute.create_cai_owned_transport_session(
            instance_id=str(payload.get("instanceId") or payload.get("instance_id") or ""),
            session_id=payload.get("sessionId") or payload.get("session_id"),
            chain_id=payload.get("chainId") or payload.get("chain_id") or payload.get("network"),
            model_id=payload.get("modelId") or payload.get("model_id"),
            task_id=payload.get("taskId") or payload.get("task_id"),
            source_node_id=payload.get("sourceNodeId") or payload.get("source_node_id"),
            participant_node_ids=payload.get("participantNodeIds")
            or payload.get("participant_node_ids")
            or [],
            executor_node_ids=payload.get("executorNodeIds")
            or payload.get("executor_node_ids")
            or None,
            execution_mode=payload.get("executionMode") or payload.get("execution_mode"),
            route_policy=payload.get("routePolicy")
            if isinstance(payload.get("routePolicy"), dict)
            else payload.get("route_policy")
            if isinstance(payload.get("route_policy"), dict)
            else None,
            policy=self.wallet_policy,
        )
        return self.modules.decentralized_compute.cai_owned_transport_session_to_dict(
            record
        )

    def accept_cai_owned_transport_session_offer(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError(
                "CAI-owned transport session offer payload must be an object."
            )
        record = (
            self.modules.decentralized_compute.create_cai_owned_transport_session_from_offer(
                payload,
                session_id=session_id,
                local_node_id=self.local_node_id,
                policy=self.wallet_policy,
            )
        )
        return self.modules.decentralized_compute.cai_owned_transport_session_to_dict(
            record
        )

    def complete_cai_owned_transport_session(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("CAI-owned transport completion payload must be an object.")
        record = self.modules.decentralized_compute.complete_cai_owned_transport_session(
            session_id,
            activation_batch_count=int(payload.get("activationBatchCount") or 0),
            decode_batch_count=int(payload.get("decodeBatchCount") or 0),
            shard_receipts=payload.get("shardReceipts")
            if isinstance(payload.get("shardReceipts"), list)
            else None,
            proof=payload.get("proof") if isinstance(payload.get("proof"), dict) else None,
            policy=self.wallet_policy,
        )
        return self.modules.decentralized_compute.cai_owned_transport_session_to_dict(
            record
        )

    def accept_cai_owned_transport_completion_notice(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError(
                "CAI-owned transport completion notice payload must be an object."
            )
        proof = payload.get("proof")
        if not isinstance(proof, dict):
            raise ValueError(
                "CAI-owned transport completion notice proof must be an object."
            )
        record = (
            self.modules.decentralized_compute.accept_cai_owned_transport_completion_notice(
                session_id,
                proof,
                policy=self.wallet_policy,
            )
        )
        return self.modules.decentralized_compute.cai_owned_transport_session_to_dict(
            record
        )

    def latest_cai_owned_transport_final_output(
        self,
        session_id: str,
        *,
        requester_node_id: str | None = None,
    ) -> dict[str, Any]:
        output = self.modules.decentralized_compute.latest_cai_owned_transport_final_output(
            session_id,
            requester_node_id=requester_node_id,
            policy=self.wallet_policy,
        )
        if output is None:
            return {
                "status": "pending",
                "sessionId": session_id,
                "finalOutput": None,
            }
        return {
            "status": "delivered",
            "sessionId": session_id,
            "finalOutput": _jsonable_transport_payload(output),
        }

    def await_cai_owned_transport_final_result(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError(
                "CAI-owned transport await-final-result payload must be an object."
            )
        result = (
            self.modules.decentralized_compute.await_cai_owned_transport_session_final_result(
                session_id,
                requester_node_id=payload.get("requesterNodeId")
                or payload.get("requester_node_id")
                or self.local_node_id,
                timeout_sec=_optional_float(
                    payload.get("timeoutSec")
                    if "timeoutSec" in payload
                    else payload.get("timeout_sec")
                )
                or 30.0,
                poll_interval_sec=_optional_float(
                    payload.get("pollIntervalSec")
                    if "pollIntervalSec" in payload
                    else payload.get("poll_interval_sec")
                )
                or 0.25,
                policy=self.wallet_policy,
            )
        )
        return _jsonable_transport_payload(result)

    def reconcile_cai_owned_transport_timeouts(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError(
                "CAI-owned transport reconcile-timeouts payload must be an object."
            )
        return self.modules.decentralized_compute.reconcile_cai_owned_transport_session_timeouts(
            session_id,
            received_timeout_sec=_optional_float(
                payload.get("receivedTimeoutSec")
                if "receivedTimeoutSec" in payload
                else payload.get("received_timeout_sec")
            ),
            max_attempts=_optional_int(
                payload.get("maxAttempts")
                if "maxAttempts" in payload
                else payload.get("max_attempts")
            ),
            policy=self.wallet_policy,
        )

    def record_cai_owned_transport_batch(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("CAI-owned transport batch payload must be an object.")
        record = self.modules.decentralized_compute.record_cai_owned_transport_batch(
            session_id,
            phase=str(payload.get("phase") or ""),
            source_node_id=str(
                payload.get("sourceNodeId") or payload.get("source_node_id") or ""
            ),
            sink_node_id=str(
                payload.get("sinkNodeId") or payload.get("sink_node_id") or ""
            ),
            payload_size_bytes=int(payload.get("payloadSizeBytes") or 0),
            payload_sha256_hex=payload.get("payloadSha256Hex")
            or payload.get("payload_sha256_hex"),
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None,
            route_audit=_optional_dict(payload, "routeAudit", "route_audit"),
            policy=self.wallet_policy,
        )
        return self.modules.decentralized_compute.cai_owned_transport_session_to_dict(
            record
        )

    def record_cai_owned_transport_batch_envelope(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("CAI-owned transport batch envelope must be an object.")
        record = (
            self.modules.decentralized_compute.record_cai_owned_transport_batch_envelope(
                session_id,
                payload,
                local_node_id=self.local_node_id,
                policy=self.wallet_policy,
            )
        )
        return self.modules.decentralized_compute.cai_owned_transport_session_to_dict(
            record
        )

    def mark_cai_owned_transport_batch_status(
        self,
        session_id: str,
        batch_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("CAI-owned transport batch status payload must be an object.")
        record = self.modules.decentralized_compute.mark_cai_owned_transport_batch_status(
            session_id,
            batch_id,
            status=str(payload.get("status") or ""),
            node_id=str(payload.get("nodeId") or payload.get("node_id") or "")
            or self.local_node_id,
            metrics=payload.get("metrics")
            if isinstance(payload.get("metrics"), dict)
            else None,
            error=payload.get("error"),
            input_payload_sha256_hex=payload.get("inputPayloadSha256Hex")
            or payload.get("input_payload_sha256_hex"),
            output_payload_sha256_hex=payload.get("outputPayloadSha256Hex")
            or payload.get("output_payload_sha256_hex"),
            output_payload_size_bytes=_optional_int(
                payload.get("outputPayloadSizeBytes")
                if "outputPayloadSizeBytes" in payload
                else payload.get("output_payload_size_bytes")
            ),
            output_payload_storage_key=payload.get("outputPayloadStorageKey")
            or payload.get("output_payload_storage_key"),
            previous_batch_id=payload.get("previousBatchId")
            or payload.get("previous_batch_id"),
            hash_chain_sha256_hex=payload.get("hashChainSha256Hex")
            or payload.get("hash_chain_sha256_hex"),
            route_audit=_optional_dict(payload, "routeAudit", "route_audit"),
            runtime_audit=_optional_dict(payload, "runtimeAudit", "runtime_audit"),
            policy=self.wallet_policy,
        )
        return self.modules.decentralized_compute.cai_owned_transport_session_to_dict(
            record
        )

    def claim_cai_owned_transport_batch(
        self,
        session_id: str,
        batch_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError(
                "CAI-owned transport batch claim payload must be an object."
            )
        record = self.modules.decentralized_compute.claim_cai_owned_transport_batch(
            session_id,
            batch_id,
            node_id=str(payload.get("nodeId") or payload.get("node_id") or "")
            or self.local_node_id,
            runtime_id=payload.get("runtimeId") or payload.get("runtime_id"),
            lease_seconds=_optional_float(
                payload.get("leaseSeconds")
                if "leaseSeconds" in payload
                else payload.get("lease_seconds")
            ),
            policy=self.wallet_policy,
        )
        return self.modules.decentralized_compute.cai_owned_transport_session_to_dict(
            record
        )

    def heartbeat_cai_owned_transport_batch(
        self,
        session_id: str,
        batch_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError(
                "CAI-owned transport batch heartbeat payload must be an object."
            )
        record = self.modules.decentralized_compute.heartbeat_cai_owned_transport_batch(
            session_id,
            batch_id,
            node_id=str(payload.get("nodeId") or payload.get("node_id") or "")
            or self.local_node_id,
            runtime_id=payload.get("runtimeId") or payload.get("runtime_id"),
            lease_seconds=_optional_float(
                payload.get("leaseSeconds")
                if "leaseSeconds" in payload
                else payload.get("lease_seconds")
            ),
            policy=self.wallet_policy,
        )
        return self.modules.decentralized_compute.cai_owned_transport_session_to_dict(
            record
        )

    def complete_cai_owned_transport_work_item(
        self,
        session_id: str,
        batch_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError(
                "CAI-owned transport complete-work-item payload must be an object."
            )
        return self.modules.decentralized_compute.complete_cai_owned_transport_work_item(
            session_id,
            batch_id,
            node_id=str(payload.get("nodeId") or payload.get("node_id") or "")
            or self.local_node_id,
            runtime_id=payload.get("runtimeId") or payload.get("runtime_id"),
            coordinator_cai_url=payload.get("coordinatorCaiUrl")
            or payload.get("coordinator_cai_url"),
            metrics=payload.get("metrics")
            if isinstance(payload.get("metrics"), dict)
            else None,
            output_payload=_optional_base64_bytes(
                payload.get("outputPayloadBase64")
                if "outputPayloadBase64" in payload
                else payload.get("output_payload_base64"),
                "outputPayloadBase64",
            ),
            output_payload_sha256_hex=payload.get("outputPayloadSha256Hex")
            or payload.get("output_payload_sha256_hex"),
            route_audit=_optional_dict(payload, "routeAudit", "route_audit"),
            runtime_audit=_optional_dict(payload, "runtimeAudit", "runtime_audit"),
            timeout_sec=_optional_float(
                payload.get("timeoutSec")
                if "timeoutSec" in payload
                else payload.get("timeout_sec")
            )
            or 5.0,
            policy=self.wallet_policy,
        )

    def fail_cai_owned_transport_work_item(
        self,
        session_id: str,
        batch_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError(
                "CAI-owned transport fail-work-item payload must be an object."
            )
        return self.modules.decentralized_compute.fail_cai_owned_transport_work_item(
            session_id,
            batch_id,
            node_id=str(payload.get("nodeId") or payload.get("node_id") or "")
            or self.local_node_id,
            runtime_id=payload.get("runtimeId") or payload.get("runtime_id"),
            error=payload.get("error") or payload.get("lastError"),
            retryable=_optional_bool(payload.get("retryable"), True),
            max_attempts=_optional_int(
                payload.get("maxAttempts")
                if "maxAttempts" in payload
                else payload.get("max_attempts")
            ),
            metrics=payload.get("metrics")
            if isinstance(payload.get("metrics"), dict)
            else None,
            policy=self.wallet_policy,
        )

    def cai_owned_transport_batch_payload_path(
        self,
        session_id: str,
        batch_id: str,
    ) -> Path:
        return (
            self.modules.decentralized_compute.verified_cai_owned_transport_batch_payload_path(
                session_id,
                batch_id,
                self.wallet_policy,
            )
        )

    def record_cai_owned_transport_shard_receipt(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("CAI-owned transport shard receipt must be an object.")
        record = (
            self.modules.decentralized_compute.record_cai_owned_transport_shard_receipt(
                session_id,
                node_id=str(payload.get("nodeId") or payload.get("node_id") or ""),
                chain_id=payload.get("chainId") or payload.get("chain_id") or payload.get("network"),
                status=str(payload.get("status") or "completed"),
                activation_batch_count=int(payload.get("activationBatchCount") or 0),
                decode_batch_count=int(payload.get("decodeBatchCount") or 0),
                layer_start=_optional_int(payload.get("layerStart")),
                layer_end=_optional_int(payload.get("layerEnd")),
                metrics=payload.get("metrics")
                if isinstance(payload.get("metrics"), dict)
                else None,
                batch_ids=payload.get("batchIds") or payload.get("batch_ids"),
                stage_ids=payload.get("stageIds") or payload.get("stage_ids"),
                sequences=payload.get("sequences"),
                input_payload_sha256_hexes=payload.get("inputPayloadSha256Hexes")
                or payload.get("input_payload_sha256_hexes"),
                output_payload_sha256_hexes=payload.get("outputPayloadSha256Hexes")
                or payload.get("output_payload_sha256_hexes"),
                hash_chain_sha256_hexes=payload.get("hashChainSha256Hexes")
                or payload.get("hash_chain_sha256_hexes"),
                route_audits=payload.get("routeAudits") or payload.get("route_audits"),
                runtime_audits=payload.get("runtimeAudits")
                or payload.get("runtime_audits"),
                signature=payload.get("signature")
                if isinstance(payload.get("signature"), dict)
                else None,
                signer_node_id=payload.get("signerNodeId")
                or payload.get("signer_node_id"),
                recorded_at=payload.get("recordedAt") or payload.get("recorded_at"),
                policy=self.wallet_policy,
            )
        )
        return self.modules.decentralized_compute.cai_owned_transport_session_to_dict(
            record
        )

    def probe_route_health(self) -> dict[str, Any]:
        state_payload = _load_state_payload(self.state_url)
        direct_records = self.modules.route_health.probe_direct_api_routes(
            state_payload=state_payload,
            local_node_id=getattr(self, "local_node_id", None),
            policy=self.wallet_policy,
        )
        data_records = self.modules.route_health.probe_direct_data_routes(
            state_payload=state_payload,
            local_node_id=getattr(self, "local_node_id", None),
            policy=self.wallet_policy,
        )
        rpc_probe = getattr(
            self.modules.route_health,
            "probe_llama_cpp_rpc_routes",
            None,
        )
        rpc_records = (
            rpc_probe(
                state_payload=state_payload,
                local_node_id=getattr(self, "local_node_id", None),
                policy=self.wallet_policy,
            )
            if callable(rpc_probe)
            else []
        )
        overlay_records = self.modules.route_health.record_overlay_routes_from_state(
            state_payload=state_payload,
            policy=self.wallet_policy,
        )
        pruned_records = self.modules.route_health.prune_stale_route_health_records(
            policy=self.wallet_policy,
        )
        all_records = self.modules.route_health.list_route_health_records(
            self.wallet_policy
        )
        relay_score = self.modules.route_health.score_relay_route_candidates(
            state_payload=state_payload,
            route_health_records=all_records,
        )
        records = [*direct_records, *data_records, *rpc_records, *overlay_records]
        return {
            "message": "Route health probe completed.",
            "probedRecords": len(records),
            "directProbedRecords": len(direct_records),
            "directDataProbedRecords": len(data_records),
            "llamaCppRpcProbedRecords": len(rpc_records),
            "overlayObservedRecords": len(overlay_records),
            "prunedRecords": pruned_records,
            "relayScore": relay_score,
            "records": [getattr(item, "__dict__", item) for item in records],
        }

    def chunk_payload(self, *, catalog_id: str, version: str, chunk_id: str) -> bytes:
        manifest = self.modules.model_distribution.load_model_package_manifest(
            catalog_id,
            version,
            self.wallet_policy,
        )
        if (
            str(getattr(manifest, "package_kind", "")).strip()
            != self.modules.model_distribution.ModelPackageKind.PUBLIC_SHARED
        ):
            raise ValueError(
                f"Chunk transport is disabled for non-public package {catalog_id}@{version}."
            )
        return self.modules.model_distribution._read_cached_chunk_payload(
            chunk_id,
            catalog_id=catalog_id,
            version=version,
            policy=self.wallet_policy,
        )

    def attest_settlement(self, payload: dict[str, Any]) -> dict[str, Any]:
        committee_validator_ids = [
            str(item).strip().lower()
            for item in (payload.get("committee_validator_ids") or [])
            if str(item).strip()
        ]
        accepted_note = str(
            payload.get("accepted_note")
            or "Remote committee validator accepted settlement."
        )
        state_payload = _load_state_payload(self.state_url)
        attestation_status = self.modules.node_config.get_validator_attestation_status(
            policy=self.wallet_policy,
            state_payload=state_payload,
            cai_url=self.cai_url,
        )
        config = self.modules.node_config.load_or_create_node_config(self.wallet_policy)
        validator_id = (
            attestation_status.validator_id
            or getattr(config, "validator_address", None)
        )
        if validator_id is None:
            return {
                "validatorId": None,
                "attested": False,
                "ignored": True,
                "accepted": None,
                "note": "No eligible local validator is configured on this node.",
            }
        validator_id = str(validator_id).strip().lower()
        if committee_validator_ids and validator_id not in committee_validator_ids:
            return {
                "validatorId": validator_id,
                "attested": False,
                "ignored": True,
                "accepted": None,
                "note": "Local validator is not a member of this settlement committee.",
            }
        if attestation_status.can_attest:
            proposal = payload.get("settlement_proposal")
            imported_settlement_id = str(payload.get("settlement_id") or "").strip()
            if isinstance(proposal, dict):
                try:
                    imported_settlement = self.modules.settlement.import_settlement_proposal_payload(
                        proposal,
                        policy=self.wallet_policy,
                        money_policy=self.money_policy,
                    )
                    imported_settlement_id = str(
                        getattr(imported_settlement, "settlement_id", imported_settlement_id)
                    ).strip()
                    self.modules.settlement.record_validator_attestation(
                        settlement_id=imported_settlement_id,
                        validator_id=validator_id,
                        accepted=True,
                        note=accepted_note,
                        policy=self.wallet_policy,
                    )
                except ValueError as exc:
                    return {
                        "validatorId": validator_id,
                        "attested": True,
                        "ignored": False,
                        "accepted": False,
                        "note": str(exc),
                    }
            return {
                "validatorId": validator_id,
                "attested": True,
                "ignored": False,
                "accepted": True,
                "note": accepted_note,
                "chain": self.chain() if isinstance(proposal, dict) else None,
            }
        return {
            "validatorId": validator_id,
            "attested": True,
            "ignored": False,
            "accepted": False,
            "note": str(attestation_status.reason),
        }

    def attest_worker_capability(self, payload: dict[str, Any]) -> dict[str, Any]:
        state_payload = _load_state_payload(self.state_url)
        attestation_status = self.modules.node_config.get_validator_attestation_status(
            policy=self.wallet_policy,
            state_payload=state_payload,
            cai_url=self.cai_url,
        )
        config = self.modules.node_config.load_or_create_node_config(self.wallet_policy)
        validator_id = (
            attestation_status.validator_id
            or getattr(config, "validator_address", None)
        )
        if validator_id is None:
            return {
                "validatorId": None,
                "attested": False,
                "ignored": True,
                "accepted": None,
                "note": "No eligible local validator is configured on this node.",
            }
        validator_id = str(validator_id).strip().lower()
        if not attestation_status.can_attest:
            return {
                "validatorId": validator_id,
                "attested": True,
                "ignored": False,
                "accepted": False,
                "note": str(attestation_status.reason),
            }

        source_url = str(payload.get("sourceUrl") or payload.get("source_url") or "").strip()
        submitted_capability_payload = payload.get("capabilityPayload") or payload.get(
            "capability_payload"
        )
        if isinstance(submitted_capability_payload, dict):
            return self._attest_worker_submitted_capability(
                payload,
                config=config,
                validator_id=validator_id,
                source_url=source_url,
            )
        if not source_url:
            return {
                "validatorId": validator_id,
                "attested": True,
                "ignored": False,
                "accepted": False,
                "note": "Worker capability sourceUrl is required for validator probe.",
            }
        endpoint_url = _worker_capability_source_endpoint(source_url)
        try:
            with urlopen(endpoint_url, timeout=5) as response:
                capability_payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            return {
                "validatorId": validator_id,
                "attested": True,
                "ignored": False,
                "accepted": False,
                "note": f"Worker capability probe failed: {exc}",
                "probe": {"sourceUrl": endpoint_url, "reachable": False},
            }

        requested_node_id = str(
            payload.get("nodeId") or payload.get("node_id") or ""
        ).strip()
        try:
            self.modules.node_capabilities.merge_remote_node_capabilities_payload(
                capability_payload,
                source_url=endpoint_url,
                policy=self.wallet_policy,
                only_node_id=requested_node_id or None,
            )
        except ValueError as exc:
            return {
                "validatorId": validator_id,
                "attested": True,
                "ignored": False,
                "accepted": False,
                "note": str(exc),
                "probe": {"sourceUrl": endpoint_url, "reachable": True},
            }

        records = self.modules.node_capabilities.list_node_capabilities(
            self.wallet_policy
        )
        candidates = [
            record
            for record in records
            if not requested_node_id
            or str(getattr(record, "node_id", "") or "").strip() == requested_node_id
        ]
        record = next((item for item in candidates if getattr(item, "worker_verified", False)), None)
        if record is None:
            return {
                "validatorId": validator_id,
                "attested": True,
                "ignored": False,
                "accepted": False,
                "note": "No signed verified worker capability was found at sourceUrl.",
                "probe": {"sourceUrl": endpoint_url, "reachable": True},
            }

        validator_wallet = self.modules.wallet.find_wallet_by_id(
            getattr(config, "validator_wallet_id", None),
            self.wallet_policy,
        )
        signer = (
            self.modules.wallet.load_unlocked_wallet_signing_material(
                validator_wallet,
                self.wallet_policy,
            )
            if validator_wallet is not None
            else None
        )
        if not signer:
            return {
                "validatorId": validator_id,
                "attested": True,
                "ignored": False,
                "accepted": False,
                "note": "Validator wallet must be unlocked to sign worker capability attestation.",
            }

        challenge_url = _worker_capability_challenge_endpoint(source_url)
        try:
            challenge = (
                self.modules.worker_capability_attestations.create_worker_capability_challenge(
                    record,
                    validator_id=validator_id,
                    validator_public_key_b64=str(signer.get("public_key_b64") or ""),
                    validator_signing_seed_b64=str(
                        signer.get("signing_seed_b64") or ""
                    ),
                    validator_pq_public_key_b64=str(
                        signer.get("pq_public_key_b64") or ""
                    ),
                    validator_pq_private_key_b64=str(
                        signer.get("pq_private_key_b64") or ""
                    ),
                    ttl_seconds=int(
                        payload.get("challengeTtlSeconds")
                        or payload.get("challenge_ttl_seconds")
                        or 60
                    ),
                    difficulty=int(
                        payload.get("challengeDifficulty")
                        or payload.get("challenge_difficulty")
                        or 2
                    ),
                )
            )
            request = Request(
                challenge_url,
                data=json.dumps(challenge).encode("utf-8"),
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urlopen(request, timeout=5) as response:
                challenge_response = json.loads(response.read().decode("utf-8"))
            challenge_receipt = (
                challenge_response.get("receipt")
                if isinstance(challenge_response, dict)
                else None
            )
            if not isinstance(challenge_receipt, dict):
                challenge_receipt = challenge_response
            if not isinstance(challenge_receipt, dict):
                return {
                    "validatorId": validator_id,
                    "attested": True,
                    "ignored": False,
                    "accepted": False,
                    "note": "Worker capability challenge receipt is missing.",
                    "probe": {
                        "sourceUrl": endpoint_url,
                        "challengeUrl": challenge_url,
                        "reachable": True,
                        "challenge": challenge,
                    },
                }
            ok, error = (
                self.modules.worker_capability_attestations.verify_worker_capability_challenge_receipt(
                    challenge_receipt,
                    challenge=challenge,
                    record=record,
                )
            )
            if not ok:
                return {
                    "validatorId": validator_id,
                    "attested": True,
                    "ignored": False,
                    "accepted": False,
                    "note": error
                    or "Worker capability challenge receipt is invalid.",
                    "probe": {
                        "sourceUrl": endpoint_url,
                        "challengeUrl": challenge_url,
                        "reachable": True,
                        "challenge": challenge,
                        "challengeReceipt": challenge_receipt,
                        "challengeVerified": False,
                    },
                }
        except Exception as exc:  # noqa: BLE001
            return {
                "validatorId": validator_id,
                "attested": True,
                "ignored": False,
                "accepted": False,
                "note": f"Worker capability challenge failed: {exc}",
                "probe": {
                    "sourceUrl": endpoint_url,
                    "challengeUrl": challenge_url,
                    "reachable": True,
                    "challengeVerified": False,
                },
            }

        attestation = (
            self.modules.worker_capability_attestations.record_worker_capability_attestation(
                record,
                validator_id=validator_id,
                validator_public_key_b64=str(signer.get("public_key_b64") or ""),
                validator_signing_seed_b64=str(signer.get("signing_seed_b64") or ""),
                validator_pq_public_key_b64=str(signer.get("pq_public_key_b64") or ""),
                validator_pq_private_key_b64=str(signer.get("pq_private_key_b64") or ""),
                ttl_seconds=int(payload.get("ttlSeconds") or payload.get("ttl_seconds") or 600),
                accepted=True,
                note="Validator accepted worker capability probe.",
                probe_result={
                    "sourceUrl": endpoint_url,
                    "challengeUrl": challenge_url,
                    "reachable": True,
                    "payloadSignatureValid": getattr(
                        record,
                        "payload_signature_valid",
                        None,
                    ),
                    "challenge": challenge,
                    "challengeReceipt": challenge_receipt,
                    "challengeVerified": True,
                },
                policy=self.wallet_policy,
            )
        )
        return {
            "validatorId": validator_id,
            "workerNodeId": getattr(record, "node_id", None),
            "attested": True,
            "ignored": False,
            "accepted": True,
            "attestation": getattr(attestation, "__dict__", attestation),
            "note": "Validator accepted worker capability probe.",
        }

    def _attest_worker_submitted_capability(
        self,
        payload: dict[str, Any],
        *,
        config: Any,
        validator_id: str,
        source_url: str,
    ) -> dict[str, Any]:
        capability_payload = payload.get("capabilityPayload") or payload.get(
            "capability_payload"
        )
        if not isinstance(capability_payload, dict):
            return {
                "validatorId": validator_id,
                "attested": True,
                "ignored": False,
                "accepted": False,
                "note": "Worker-submitted capability payload is required.",
            }

        requested_node_id = str(
            payload.get("nodeId") or payload.get("node_id") or ""
        ).strip()
        source_label = str(
            source_url
            or payload.get("submittedSourceUrl")
            or payload.get("submitted_source_url")
            or capability_payload.get("sourceUrl")
            or "worker-submitted"
        ).strip()
        if not source_label:
            source_label = "worker-submitted"
        if source_label.startswith(("http://", "https://")):
            source_label = _worker_capability_source_endpoint(source_label)

        try:
            self.modules.node_capabilities.merge_remote_node_capabilities_payload(
                capability_payload,
                source_url=source_label,
                policy=self.wallet_policy,
                only_node_id=requested_node_id or None,
            )
        except ValueError as exc:
            return {
                "validatorId": validator_id,
                "attested": True,
                "ignored": False,
                "accepted": False,
                "note": str(exc),
                "probe": {
                    "sourceUrl": source_label,
                    "attestationMode": "worker_submitted",
                    "reachable": None,
                },
            }

        records = self.modules.node_capabilities.list_node_capabilities(
            self.wallet_policy
        )
        candidates = [
            record
            for record in records
            if not requested_node_id
            or str(getattr(record, "node_id", "") or "").strip() == requested_node_id
        ]
        record = next(
            (item for item in candidates if getattr(item, "worker_verified", False)),
            None,
        )
        if record is None:
            return {
                "validatorId": validator_id,
                "attested": True,
                "ignored": False,
                "accepted": False,
                "note": "No signed verified worker capability was found in submitted payload.",
                "probe": {
                    "sourceUrl": source_label,
                    "attestationMode": "worker_submitted",
                    "reachable": None,
                },
            }

        validator_wallet = self.modules.wallet.find_wallet_by_id(
            getattr(config, "validator_wallet_id", None),
            self.wallet_policy,
        )
        signer = (
            self.modules.wallet.load_unlocked_wallet_signing_material(
                validator_wallet,
                self.wallet_policy,
            )
            if validator_wallet is not None
            else None
        )
        if not signer:
            return {
                "validatorId": validator_id,
                "attested": True,
                "ignored": False,
                "accepted": False,
                "note": "Validator wallet must be unlocked to sign worker capability attestation.",
            }

        challenge_receipt = payload.get("challengeReceipt") or payload.get(
            "challenge_receipt"
        )
        if not isinstance(challenge_receipt, dict):
            challenge = (
                self.modules.worker_capability_attestations.create_worker_capability_challenge(
                    record,
                    validator_id=validator_id,
                    validator_public_key_b64=str(signer.get("public_key_b64") or ""),
                    validator_signing_seed_b64=str(
                        signer.get("signing_seed_b64") or ""
                    ),
                    validator_pq_public_key_b64=str(
                        signer.get("pq_public_key_b64") or ""
                    ),
                    validator_pq_private_key_b64=str(
                        signer.get("pq_private_key_b64") or ""
                    ),
                    ttl_seconds=int(
                        payload.get("challengeTtlSeconds")
                        or payload.get("challenge_ttl_seconds")
                        or 60
                    ),
                    difficulty=int(
                        payload.get("challengeDifficulty")
                        or payload.get("challenge_difficulty")
                        or 2
                    ),
                )
            )
            return {
                "validatorId": validator_id,
                "workerNodeId": getattr(record, "node_id", None),
                "attested": True,
                "ignored": False,
                "accepted": False,
                "challengeRequired": True,
                "challenge": challenge,
                "note": "Worker capability challenge is required.",
                "probe": {
                    "sourceUrl": source_label,
                    "attestationMode": "worker_submitted",
                    "reachable": None,
                    "payloadSignatureValid": getattr(
                        record,
                        "payload_signature_valid",
                        None,
                    ),
                    "challengeVerified": False,
                },
            }

        challenge = payload.get("challenge")
        if not isinstance(challenge, dict):
            return {
                "validatorId": validator_id,
                "attested": True,
                "ignored": False,
                "accepted": False,
                "note": "Worker capability challenge is required with challengeReceipt.",
                "probe": {
                    "sourceUrl": source_label,
                    "attestationMode": "worker_submitted",
                    "reachable": None,
                    "challengeVerified": False,
                },
            }

        ok, error = (
            self.modules.worker_capability_attestations.verify_worker_capability_challenge(
                challenge,
                policy=self.wallet_policy,
                require_bonded_validator=True,
            )
        )
        if not ok:
            return {
                "validatorId": validator_id,
                "attested": True,
                "ignored": False,
                "accepted": False,
                "note": error or "Worker capability challenge is invalid.",
                "probe": {
                    "sourceUrl": source_label,
                    "attestationMode": "worker_submitted",
                    "reachable": None,
                    "challenge": challenge,
                    "challengeVerified": False,
                },
            }
        if str(challenge.get("validator_id") or "").strip().lower() != validator_id:
            return {
                "validatorId": validator_id,
                "attested": True,
                "ignored": False,
                "accepted": False,
                "note": "Worker capability challenge validator mismatch.",
                "probe": {
                    "sourceUrl": source_label,
                    "attestationMode": "worker_submitted",
                    "reachable": None,
                    "challenge": challenge,
                    "challengeReceipt": challenge_receipt,
                    "challengeVerified": False,
                },
            }

        ok, error = (
            self.modules.worker_capability_attestations.verify_worker_capability_challenge_receipt(
                challenge_receipt,
                challenge=challenge,
                record=record,
            )
        )
        if not ok:
            return {
                "validatorId": validator_id,
                "attested": True,
                "ignored": False,
                "accepted": False,
                "note": error or "Worker capability challenge receipt is invalid.",
                "probe": {
                    "sourceUrl": source_label,
                    "attestationMode": "worker_submitted",
                    "reachable": None,
                    "challenge": challenge,
                    "challengeReceipt": challenge_receipt,
                    "challengeVerified": False,
                },
            }

        attestation = (
            self.modules.worker_capability_attestations.record_worker_capability_attestation(
                record,
                validator_id=validator_id,
                validator_public_key_b64=str(signer.get("public_key_b64") or ""),
                validator_signing_seed_b64=str(signer.get("signing_seed_b64") or ""),
                validator_pq_public_key_b64=str(signer.get("pq_public_key_b64") or ""),
                validator_pq_private_key_b64=str(signer.get("pq_private_key_b64") or ""),
                ttl_seconds=int(payload.get("ttlSeconds") or payload.get("ttl_seconds") or 600),
                accepted=True,
                note="Validator accepted worker-submitted capability challenge.",
                probe_result={
                    "sourceUrl": source_label,
                    "attestationMode": "worker_submitted",
                    "reachable": None,
                    "payloadSignatureValid": getattr(
                        record,
                        "payload_signature_valid",
                        None,
                    ),
                    "challenge": challenge,
                    "challengeReceipt": challenge_receipt,
                    "challengeVerified": True,
                },
                policy=self.wallet_policy,
            )
        )
        return {
            "validatorId": validator_id,
            "workerNodeId": getattr(record, "node_id", None),
            "attested": True,
            "ignored": False,
            "accepted": True,
            "attestation": getattr(attestation, "__dict__", attestation),
            "note": "Validator accepted worker-submitted capability challenge.",
        }

    def attest_penalty_case(self, payload: dict[str, Any]) -> dict[str, Any]:
        eligible_validator_ids = [
            str(item).strip().lower()
            for item in (payload.get("eligible_validator_ids") or [])
            if str(item).strip()
        ]
        accepted_note = str(
            payload.get("accepted_note")
            or "Remote committee validator accepted penalty case."
        )
        state_payload = _load_state_payload(self.state_url)
        attestation_status = self.modules.node_config.get_validator_attestation_status(
            policy=self.wallet_policy,
            state_payload=state_payload,
            cai_url=self.cai_url,
        )
        config = self.modules.node_config.load_or_create_node_config(self.wallet_policy)
        validator_id = (
            attestation_status.validator_id
            or getattr(config, "validator_address", None)
        )
        if validator_id is None:
            return {
                "validatorId": None,
                "attested": False,
                "ignored": True,
                "accepted": None,
                "note": "No eligible local validator is configured on this node.",
            }
        validator_id = str(validator_id).strip().lower()
        if eligible_validator_ids and validator_id not in eligible_validator_ids:
            return {
                "validatorId": validator_id,
                "attested": False,
                "ignored": True,
                "accepted": None,
                "note": "Local validator is not eligible to attest this penalty case.",
            }
        if attestation_status.can_attest:
            return {
                "validatorId": validator_id,
                "attested": True,
                "ignored": False,
                "accepted": True,
                "note": accepted_note,
            }
        return {
            "validatorId": validator_id,
            "attested": True,
            "ignored": False,
            "accepted": False,
            "note": str(attestation_status.reason),
        }

    def history_page(
        self,
        *,
        section: str,
        offset: int = 0,
        limit: int = 10,
    ) -> dict[str, Any]:
        normalized_section = section.strip().lower()
        normalized_offset = max(0, int(offset))
        normalized_limit = min(max(1, int(limit)), 50)
        active_wallet = self.modules.wallet.get_active_wallet(self.wallet_policy)
        active_wallet_id = getattr(active_wallet, "wallet_id", None)
        to_coins = self.modules.wallet.atomic_to_coins

        if normalized_section == "journal":
            items = _wallet_activity_history_summaries(
                self.modules,
                active_wallet=active_wallet,
                active_wallet_id=active_wallet_id,
                wallet_policy=self.wallet_policy,
                money_policy=self.money_policy,
                to_coins=to_coins,
            )
            page = items[normalized_offset : normalized_offset + normalized_limit]
            return {
                "section": normalized_section,
                "offset": normalized_offset,
                "limit": normalized_limit,
                "hasMore": len(items) > normalized_offset + normalized_limit,
                "items": page,
            }

        if normalized_section == "jobs":
            items = self.modules.jobs.list_job_intents(self.wallet_policy)
            if active_wallet_id:
                items = [item for item in items if item.source_wallet_id == active_wallet_id]
            page = items[normalized_offset : normalized_offset + normalized_limit]
            receipt_ids = {
                getattr(item, "receipt_id", None) for item in page if getattr(item, "receipt_id", None)
            }
            receipts = self.modules.jobs.list_execution_receipts(self.wallet_policy)
            receipt_by_id = {
                getattr(item, "receipt_id", None): item
                for item in receipts
                if getattr(item, "receipt_id", None) in receipt_ids
            }
            return {
                "section": normalized_section,
                "offset": normalized_offset,
                "limit": normalized_limit,
                "hasMore": len(items) > normalized_offset + normalized_limit,
                "items": [
                    _job_to_history_summary(item, receipt_by_id.get(getattr(item, "receipt_id", None)))
                    for item in page
                ],
            }

        if normalized_section == "payouts":
            items = self.modules.settlement.list_worker_payouts(policy=self.wallet_policy)
            page = items[normalized_offset : normalized_offset + normalized_limit]
            return {
                "section": normalized_section,
                "offset": normalized_offset,
                "limit": normalized_limit,
                "hasMore": len(items) > normalized_offset + normalized_limit,
                "items": [
                    _payout_to_summary(item, self.money_policy, to_coins) for item in page
                ],
            }

        if normalized_section == "settlements":
            items = self.modules.settlement.list_settlements(policy=self.wallet_policy)
            page = items[normalized_offset : normalized_offset + normalized_limit]
            return {
                "section": normalized_section,
                "offset": normalized_offset,
                "limit": normalized_limit,
                "hasMore": len(items) > normalized_offset + normalized_limit,
                "items": [
                    _settlement_to_summary(
                        item,
                        self.money_policy,
                        to_coins,
                        chain_transactions=_settlement_chain_history_from_modules(
                            self.modules,
                            getattr(item, "settlement_id", None),
                            self.wallet_policy,
                        ),
                    )
                    for item in page
                ],
            }

        raise ValueError(f"Unsupported history section: {section}")

    def list_wallet_rows(self) -> list[dict[str, Any]]:
        self.modules.chain.ensure_chain_genesis(
            policy=self.wallet_policy,
            money_policy=self.money_policy,
        )
        session = self.modules.wallet.load_session(self.wallet_policy)
        rows: list[dict[str, Any]] = []
        for wallet in self.modules.wallet.list_wallets(self.wallet_policy):
            balance_atomic = self.modules.chain.wallet_chain_balance_or_local_atomic(
                wallet,
                self.wallet_policy,
            )
            rows.append(
                {
                    "wallet_id": wallet.wallet_id,
                    "selector": wallet.wallet_id,
                    "name": wallet.name,
                    "address": wallet.address,
                    "balance_coins": self.modules.wallet.atomic_to_coins(
                        balance_atomic,
                        self.money_policy,
                    ),
                    "balance_source": self.modules.chain.wallet_balance_source(
                        self.wallet_policy,
                    ),
                    "local_cached_balance_coins": self.modules.wallet.atomic_to_coins(
                        wallet.spendable_balance_atomic,
                        self.money_policy,
                    ),
                    "active": wallet.wallet_id == session.active_wallet_id,
                    "unlocked": wallet.wallet_id == session.unlocked_wallet_id,
                    "seed_backed": bool(getattr(wallet, "seed_fingerprint", None)),
                }
            )
        return rows

    def create_wallet(self, *, name: str, password: str) -> dict[str, Any]:
        wallet, seed_phrase = self.modules.wallet.create_seed_wallet(
            name,
            password,
            select=True,
            wallet_policy=self.wallet_policy,
        )
        self.modules.wallet.unlock_wallet(
            password,
            selector=wallet.wallet_id,
            wallet_policy=self.wallet_policy,
        )
        return {
            "message": f"Wallet '{wallet.name}' created.",
            "wallet": {
                "wallet_id": wallet.wallet_id,
                "name": wallet.name,
                "address": wallet.address,
                "created_at": wallet.created_at,
            },
            "seed_phrase": seed_phrase,
        }

    def restore_wallet(self, *, name: str, password: str, seed_phrase: str) -> dict[str, Any]:
        wallet = self.modules.wallet.restore_wallet_from_seed(
            name,
            password,
            seed_phrase=seed_phrase,
            select=True,
            wallet_policy=self.wallet_policy,
        )
        self.modules.wallet.unlock_wallet(
            password,
            selector=wallet.wallet_id,
            wallet_policy=self.wallet_policy,
        )
        return {
            "message": f"Wallet '{wallet.name}' restored.",
            "wallet": {
                "wallet_id": wallet.wallet_id,
                "name": wallet.name,
                "address": wallet.address,
                "created_at": wallet.created_at,
            },
        }

    def select_wallet(self, *, selector: str) -> dict[str, Any]:
        wallet = self.modules.wallet.select_active_wallet(selector, self.wallet_policy)
        config = self.modules.node_config.load_or_create_node_config(self.wallet_policy)
        validator_reset = False
        if config.validator_enabled and config.validator_wallet_id != wallet.wallet_id:
            self.modules.node_config.set_validator_mode(False, self.wallet_policy)
            validator_reset = True
        return {
            "message": (
                f"Active wallet set to {wallet.name}."
                if not validator_reset
                else f"Active wallet set to {wallet.name}. Validator mode was disabled because it was bound to another wallet."
            ),
            "wallet": {"wallet_id": wallet.wallet_id, "address": wallet.address},
        }

    def unlock_wallet(self, *, password: str, wallet: str | None = None) -> dict[str, Any]:
        unlocked = self.modules.wallet.unlock_wallet(
            password,
            selector=wallet,
            wallet_policy=self.wallet_policy,
        )
        return {
            "message": f"Wallet unlocked: {unlocked.name}.",
            "wallet": {"wallet_id": unlocked.wallet_id, "address": unlocked.address},
        }

    def lock_wallet(self) -> dict[str, Any]:
        session = self.modules.wallet.lock_wallet(self.wallet_policy)
        return {
            "message": "Wallet session locked.",
            "session": {
                "active_wallet_id": session.active_wallet_id,
                "unlocked_wallet_id": session.unlocked_wallet_id,
                "unlocked_at": session.unlocked_at,
            },
        }

    def logout_wallet(self) -> dict[str, Any]:
        config = self.modules.node_config.load_or_create_node_config(self.wallet_policy)
        if config.validator_enabled:
            self.modules.node_config.set_validator_mode(False, self.wallet_policy)
        session = self.modules.wallet.logout_wallet(self.wallet_policy)
        return {
            "message": "Wallet session closed.",
            "session": {
                "active_wallet_id": session.active_wallet_id,
                "unlocked_wallet_id": session.unlocked_wallet_id,
                "unlocked_at": session.unlocked_at,
            },
        }

    def send_wallet_transfer(self, *, to: str, amount: str) -> dict[str, Any]:
        self.modules.chain.ensure_chain_genesis(
            policy=self.wallet_policy,
            money_policy=self.money_policy,
        )
        active_wallet = self.modules.wallet.get_active_wallet(self.wallet_policy)
        if active_wallet is None:
            raise ValueError("Active wallet is not set.")
        session = self.modules.wallet.load_session(self.wallet_policy)
        if session.unlocked_wallet_id != active_wallet.wallet_id:
            raise ValueError("Active wallet must be unlocked before sending a transaction.")
        amount_atomic = self.modules.wallet.coins_to_atomic(amount, self.money_policy)
        tx_fee_atomic = self.modules.wallet.coins_to_atomic(
            self.money_policy.default_tx_fee_coins, self.money_policy
        )
        sender, recipient = self.modules.wallet.apply_wallet_transfer(
            sender_wallet_id=active_wallet.wallet_id,
            recipient_address=self.modules.wallet.normalize_address(to),
            amount_atomic=amount_atomic,
            tx_fee_atomic=tx_fee_atomic,
            wallet_policy=self.wallet_policy,
        )
        ledger = self.modules.wallet.load_or_create_ledger(self.money_policy, self.wallet_policy)
        ledger.tx_fee_pool_atomic += tx_fee_atomic
        self.modules.wallet.save_ledger(ledger, self.wallet_policy)
        created_at = datetime.now(tz=UTC).isoformat()
        self.modules.wallet.append_journal_entry(
            self.modules.wallet.JournalEntry(
                entry_id=secrets.token_hex(12),
                event_type="wallet_send",
                created_at=created_at,
                wallet_id=sender.wallet_id,
                counterparty_address=self.modules.wallet.normalize_address(to),
                amount_atomic=amount_atomic,
                tx_fee_atomic=tx_fee_atomic,
                note="Transfer sent from CAI dashboard.",
            ),
            self.wallet_policy,
        )
        if recipient is not None:
            self.modules.wallet.append_journal_entry(
                self.modules.wallet.JournalEntry(
                    entry_id=secrets.token_hex(12),
                    event_type="wallet_receive",
                    created_at=created_at,
                    wallet_id=recipient.wallet_id,
                    counterparty_address=sender.address,
                    amount_atomic=amount_atomic,
                    note="Transfer received from CAI dashboard.",
                ),
                self.wallet_policy,
            )
        return {
            "message": (
                f"Transfer sent to {self.modules.wallet.normalize_address(to)}."
                if recipient is None
                else f"Transfer sent to local wallet {recipient.name}."
            )
        }

    def set_validator_enabled(self, *, enabled: bool) -> dict[str, Any]:
        state_payload = None
        if enabled:
            snapshot = self.modules.ui_state.build_interface_snapshot(
                state_url=self.state_url,
                cai_url=self.cai_url,
                money_policy=self.money_policy,
                network_config=self.network_config,
                network_model_policy=self.network_model_policy,
            ).to_dict()
            validator_summary = snapshot.get("validator") or {}
            if not bool(validator_summary.get("validator_can_enable")):
                raise ValueError(
                    str(
                        validator_summary.get("validator_status_note")
                        or "Validator mode cannot be enabled on this node."
                    )
                )
            state_payload = _load_state_payload(self.state_url)
        config = self.modules.node_config.set_validator_mode(
            enabled,
            self.wallet_policy,
            self.money_policy,
            state_payload=state_payload,
            cai_url=self.cai_url,
        )
        state = "enabled" if config.validator_enabled else "disabled"
        message = f"Validator mode {state}."
        if config.validator_enabled and config.validator_address:
            bond_coins = self.modules.wallet.atomic_to_coins(
                config.validator_bond_atomic, self.money_policy
            )
            message = (
                f"Validator mode enabled for {config.validator_address} "
                f"with self-bond {bond_coins}."
            )
        elif config.validator_state == "unbonding":
            message = "Validator mode disabled. Unbonding started and bond remains locked."
        return {"message": message, "config": _dataclass_like(config)}

    def complete_validator_unbond(self) -> dict[str, Any]:
        config = self.modules.node_config.complete_validator_unbond(self.wallet_policy)
        return {
            "message": "Validator unbonding completed and bond released.",
            "config": _dataclass_like(config),
        }

    def clear_validator_jail(self) -> dict[str, Any]:
        config = self.modules.node_config.clear_validator_jail(self.wallet_policy)
        return {
            "message": "Validator jail cleared.",
            "config": _dataclass_like(config),
        }

    def set_validator_static_ip_confirmed(self, *, confirmed: bool) -> dict[str, Any]:
        config = self.modules.node_config.set_validator_static_ip_confirmation(
            confirmed, self.wallet_policy
        )
        state = "confirmed" if config.validator_static_ip_confirmed else "cleared"
        return {"message": f"Validator static IP confirmation {state}.", "config": _dataclass_like(config)}

    def set_worker_enabled(self, *, enabled: bool) -> dict[str, Any]:
        current_node_id: str | None = None
        if enabled:
            current = self.modules.node_config.load_or_create_node_config(self.wallet_policy)
            if getattr(current, "validator_state", None) in {"bonded", "unbonding"}:
                raise ValueError("Disable validator mode and finish unbonding before enabling worker mode.")
            self._require_worker_reward_destination_for_enable()
        current = self.modules.node_config.load_or_create_node_config(self.wallet_policy)
        config = self.modules.node_config.set_worker_mode(
            enabled=enabled,
            allowed_model_ids=list(current.worker_allowed_model_ids),
            max_parallel_jobs=current.worker_max_parallel_jobs,
            max_memory_mb=current.worker_max_memory_mb,
            policy=self.wallet_policy,
        )
        if enabled:
            current_node_id = self._ensure_local_worker_reward_binding()
        if enabled and current_node_id:
            self.modules.settlement.reconcile_worker_payouts(self.wallet_policy)
        state = "enabled" if config.worker_enabled else "disabled"
        return {"message": f"Worker mode {state}.", "config": _dataclass_like(config)}

    def set_relay_enabled(self, *, enabled: bool) -> dict[str, Any]:
        config = self.modules.node_config.set_relay_mode(
            bool(enabled),
            self.wallet_policy,
        )
        state = "enabled" if getattr(config, "relay_enabled", False) else "disabled"
        return {"message": f"Relay mode {state}.", "config": _dataclass_like(config)}

    def _resolve_chat_model_id(
        self,
        model_id: str,
    ) -> str:
        normalized_model_id = str(model_id or "").strip()
        return normalized_model_id

    def chat_completion(
        self,
        payload: dict[str, Any],
        *,
        reserve_client_ip: str | None = None,
    ) -> dict[str, Any]:
        request_payload = dict(payload)
        request_payload["stream"] = False
        request_payload.setdefault("reasoning_effort", "none")
        request_payload.setdefault("enable_thinking", False)

        requested_model_id = str(
            request_payload.get("model")
            or getattr(self.network_model_policy, "network_default_model_id", None)
            or self.network_model_policy.network_default_execution_model_id
        ).strip()
        model_id = self._resolve_chat_model_id(requested_model_id)
        request_payload["model"] = model_id
        if not model_id:
            raise ValueError("A model id is required for CAI metered chat.")

        prompt = _extract_display_prompt_from_chat_payload(request_payload)
        if not prompt:
            raise ValueError("CAI metered chat requires at least one user message.")

        self._ensure_local_worker_reward_binding()

        job = self.modules.jobs.create_job_intent(
            prompt=prompt,
            compute_amount_coins=None,
            payment_preference=self.modules.model.PaymentPreference.AUTO,
            cai_url=self.cai_url,
            execution_cai_url=self.execution_cai_url,
            model_id=model_id,
            requester_node_id=self.local_node_id,
            request_payload_preview=request_payload,
            reserve_client_ip=reserve_client_ip,
            money_policy=self.money_policy,
            network_model_policy=self.network_model_policy,
            wallet_policy=self.wallet_policy,
        )
        job, receipt = self.modules.jobs.execute_job_intent(
            job.job_id,
            money_policy=self.money_policy,
            wallet_policy=self.wallet_policy,
            request_timeout_sec=_env_positive_int(
                "CAI_CHAT_COMPLETION_TIMEOUT_SECONDS",
                1800,
            ),
            request_payload_override=request_payload,
        )
        settlement_summary = None
        payout_summaries: list[dict[str, Any]] = []
        settlement_id = getattr(job, "settlement_id", None)
        if settlement_id:
            try:
                settlement = self.modules.settlement.resolve_settlement(
                    settlement_id,
                    self.wallet_policy,
                )
                settlement_summary = _settlement_to_summary(
                    settlement,
                    self.money_policy,
                    self.modules.wallet.atomic_to_coins,
                    chain_transactions=_settlement_chain_history_from_modules(
                        self.modules,
                        settlement_id,
                        self.wallet_policy,
                    ),
                )
            except Exception:  # noqa: BLE001
                settlement_summary = None
            try:
                payout_summaries = [
                    _payout_to_summary(
                        item,
                        self.money_policy,
                        self.modules.wallet.atomic_to_coins,
                    )
                    for item in self.modules.settlement.list_worker_payouts(
                        settlement_id=settlement_id,
                        receipt_id=getattr(receipt, "receipt_id", None),
                        policy=self.wallet_policy,
                    )
                ]
            except Exception:  # noqa: BLE001
                payout_summaries = []
        return {
            "response": receipt.raw_response,
            "job": _job_to_summary(job),
            "receipt": _receipt_to_summary(receipt),
            "settlement": settlement_summary,
            "payouts": payout_summaries,
        }

    def _require_worker_reward_destination_for_enable(self) -> None:
        active_wallet = self.modules.wallet.get_active_wallet(self.wallet_policy)
        if active_wallet is not None:
            return
        if self._resolve_bound_local_worker_reward_address() is not None:
            return
        raise ValueError(
            "Select an active wallet or bind a worker reward address before enabling worker mode."
        )

    def _resolve_local_worker_node_id(self) -> str | None:
        try:
            state_payload = _load_state_payload(self.state_url)
        except Exception:
            return None

        node_id = _resolve_local_runtime_node_id(
            state_payload=state_payload,
            local_node_id=getattr(self, "local_node_id", None),
        )
        if node_id:
            return node_id

        network_status = self.modules.node_config.assess_validator_network_status(
            state_payload=state_payload,
            cai_url=self.cai_url,
            policy=self.wallet_policy,
        )
        return getattr(network_status, "current_node_id", None)

    def _resolve_bound_local_worker_reward_address(self) -> str | None:
        node_id = self._resolve_local_worker_node_id()
        if not node_id:
            return None
        return self.modules.node_config.resolve_worker_reward_address(
            node_id, self.wallet_policy
        )

    def _ensure_local_worker_reward_binding(self) -> str | None:
        active_wallet = self.modules.wallet.get_active_wallet(self.wallet_policy)
        if active_wallet is None:
            return None

        node_id = self._resolve_local_worker_node_id()
        if not node_id:
            return None

        existing_address = self.modules.node_config.resolve_worker_reward_address(
            node_id, self.wallet_policy
        )
        normalized_active_address = self.modules.wallet.normalize_address(
            active_wallet.address
        )
        if existing_address == normalized_active_address:
            return node_id

        self.modules.node_config.bind_worker_reward_address(
            node_id,
            active_wallet.address,
            policy=self.wallet_policy,
        )
        return node_id

    def _local_worker_reward_address(self) -> str | None:
        node_id = self._ensure_local_worker_reward_binding()
        if node_id:
            return self.modules.node_config.resolve_worker_reward_address(
                node_id, self.wallet_policy
            )

        resolved_node_id = self._resolve_local_worker_node_id()
        if not resolved_node_id:
            return None
        return self.modules.node_config.resolve_worker_reward_address(
            resolved_node_id, self.wallet_policy
        )


def load_cai_summary(
    *,
    state_url: str,
    cai_url: str | None = None,
    execution_cai_url: str | None = None,
    local_node_id: str | None = None,
    CAI_url: str | None = None,
) -> dict[str, Any]:
    try:
        return CaiBridgeService(
            state_url=state_url,
            cai_url=cai_url,
            execution_cai_url=execution_cai_url,
            local_node_id=local_node_id,
            CAI_url=CAI_url,
        ).summary()
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "error": str(exc),
        }


def make_cai_service(
    *,
    state_url: str,
    cai_url: str | None = None,
    execution_cai_url: str | None = None,
    local_node_id: str | None = None,
    CAI_url: str | None = None,
) -> CaiBridgeService:
    return CaiBridgeService(
        state_url=state_url,
        cai_url=cai_url,
        execution_cai_url=execution_cai_url,
        local_node_id=local_node_id,
        CAI_url=CAI_url,
    )


def _load_state_payload(state_url: str) -> dict[str, Any]:
    with urlopen(state_url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _sync_result_int_attr(result: Any, name: str) -> int:
    value = getattr(result, name, 0)
    return value if isinstance(value, int) else 0


def _sync_result_list_attr(result: Any, name: str) -> list[Any]:
    value = getattr(result, name, [])
    return list(value) if isinstance(value, (list, tuple)) else []


def _peer_sync_result_summary(result: Any) -> dict[str, Any]:
    return {
        "attemptedPeers": _sync_result_int_attr(result, "attempted_peers"),
        "successfulPeers": _sync_result_int_attr(result, "successful_peers"),
        "failedPeers": _sync_result_int_attr(result, "failed_peers"),
        "peerUrls": _sync_result_list_attr(result, "peer_urls"),
        "failedPeerUrls": _sync_result_list_attr(result, "failed_peer_urls"),
        "peerErrors": _sync_result_list_attr(result, "peer_errors"),
    }


def _operation_error_payload(operation: str, exc: Exception) -> dict[str, str]:
    return {
        "operation": operation,
        "errorType": type(exc).__name__,
        "message": str(exc),
    }


def _unsigned_peer_payload(
    payload: dict[str, Any],
    *,
    reason: str,
    exc: Exception | None = None,
) -> dict[str, Any]:
    unsigned = dict(payload)
    status: dict[str, Any] = {
        "signed": False,
        "reason": reason,
    }
    if exc is not None:
        status["errorType"] = type(exc).__name__
        status["message"] = str(exc)
    unsigned["signatureStatus"] = status
    return unsigned


def _resolve_local_runtime_node_id(
    *, state_payload: dict[str, Any], local_node_id: str | None
) -> str | None:
    identities = state_payload.get("nodeIdentities")
    if not isinstance(identities, dict):
        return local_node_id

    if local_node_id and local_node_id in identities:
        return local_node_id

    host_candidates = {
        value.strip().lower()
        for value in (
            os.environ.get("COMPUTERNAME"),
            os.environ.get("HOSTNAME"),
            socket.gethostname(),
        )
        if value and value.strip()
    }

    matched_ids = [
        str(node_id)
        for node_id, identity in identities.items()
        if isinstance(identity, dict)
        and str(identity.get("friendlyName", "")).strip().lower() in host_candidates
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

    return local_node_id


def _local_node_resource_summary(
    state_payload: dict[str, Any] | None,
    local_node_id: str | None,
) -> dict[str, Any]:
    if not isinstance(state_payload, dict):
        return {}
    node_id = str(local_node_id or "").strip()
    if not node_id:
        return {}
    summary: dict[str, Any] = {}

    identities = state_payload.get("nodeIdentities")
    identity = identities.get(node_id) if isinstance(identities, dict) else None
    if isinstance(identity, dict):
        resources = identity.get("resources") or identity.get("resourceSummary")
        if isinstance(resources, dict):
            summary.update(resources)
        for source_key, target_key in (
            ("ramBytes", "ramBytes"),
            ("ram_bytes", "ramBytes"),
            ("vramBytes", "vramBytes"),
            ("vram_bytes", "vramBytes"),
            ("totalVramBytes", "vramBytes"),
            ("total_vram_bytes", "vramBytes"),
            ("cpuCores", "cpuCores"),
            ("cpu_cores", "cpuCores"),
            ("cpuPhysicalCores", "cpuCores"),
            ("cpu_physical_cores", "cpuCores"),
        ):
            value = identity.get(source_key)
            if value is not None:
                summary.setdefault(target_key, value)

    node_memory = state_payload.get("nodeMemory")
    memory = node_memory.get(node_id) if isinstance(node_memory, dict) else None
    if isinstance(memory, dict):
        ram_total = _bridge_byte_count(
            memory.get("ramTotal") or memory.get("ram_total")
        )
        ram_available = _bridge_byte_count(
            memory.get("ramAvailable") or memory.get("ram_available")
        )
        if ram_total is not None:
            summary["ramBytes"] = ram_total
        if ram_available is not None:
            summary["ramAvailableBytes"] = ram_available
    return summary


def _local_node_readiness_summary(
    state_payload: dict[str, Any] | None,
    local_node_id: str | None,
) -> dict[str, Any]:
    if not isinstance(state_payload, dict):
        return {}
    node_id = str(local_node_id or "").strip()
    if not node_id:
        return {}
    identities = state_payload.get("nodeIdentities")
    identity = identities.get(node_id) if isinstance(identities, dict) else None
    if not isinstance(identity, dict):
        return {}

    readiness = identity.get("readiness")
    if isinstance(readiness, dict):
        return dict(readiness)

    model_readiness = identity.get("modelReadiness") or identity.get("model_readiness")
    if isinstance(model_readiness, dict):
        return {"models": dict(model_readiness)}
    return {}


def _bridge_byte_count(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("inBytes", "in_bytes", "bytes", "value"):
            if key in value:
                return _bridge_byte_count(value.get(key))
        return None
    if isinstance(value, bool):
        return None
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _build_local_model_shard_inventory(
    state_payload: dict[str, Any] | None,
    local_node_id: str | None,
    *,
    model_distribution: Any | None = None,
    wallet_policy: Any | None = None,
) -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    if not isinstance(state_payload, dict):
        state_payload = {}
    node_id = str(local_node_id or "").strip()
    downloads = state_payload.get("downloads")
    progress_items = []
    if node_id and isinstance(downloads, dict):
        raw_progress_items = downloads.get(node_id) or downloads.get(str(node_id))
        if isinstance(raw_progress_items, list):
            progress_items = raw_progress_items

    _merge_download_model_shard_inventory(inventory, progress_items)
    _merge_chunk_cache_model_shard_inventory(
        inventory,
        model_distribution=model_distribution,
        wallet_policy=wallet_policy,
    )
    for entry in inventory.values():
        shards = entry.get("shards") if isinstance(entry, dict) else []
        ready_shards = [
            item
            for item in shards
            if isinstance(item, dict) and bool(item.get("ready"))
        ]
        if ready_shards:
            entry["status"] = "partial_ready"
        elif shards:
            entry["status"] = str(shards[0].get("status") or "not_ready")
    return inventory


def _merge_download_model_shard_inventory(
    inventory: dict[str, Any],
    progress_items: list[Any],
) -> None:
    for progress_item in progress_items:
        tag, payload = _unwrap_tagged_payload(progress_item)
        if not isinstance(payload, dict):
            continue
        shard_tag, shard = _unwrap_tagged_payload(
            payload.get("shardMetadata") or payload.get("shard_metadata")
        )
        if not isinstance(shard, dict):
            continue
        model_card = _as_mapping(
            shard.get("modelCard")
            or shard.get("model_card")
            or payload.get("modelCard")
            or payload.get("model_card")
        )
        model_id = str(
            model_card.get("modelId")
            or model_card.get("model_id")
            or shard.get("modelId")
            or shard.get("model_id")
            or ""
        ).strip()
        if not model_id:
            continue

        layer_start = _optional_int_field(shard, "startLayer", "start_layer")
        layer_end = _optional_int_field(shard, "endLayer", "end_layer")
        n_layers = _optional_int_field(shard, "nLayers", "n_layers")
        if layer_start is None and layer_end is None and n_layers is not None:
            layer_start, layer_end = 0, n_layers
        if layer_start is None or layer_end is None or layer_end <= layer_start:
            continue

        status, ready = _download_status_for_inventory(tag, payload)
        shard_item: dict[str, Any] = {
            "layerStart": layer_start,
            "layerEnd": layer_end,
            "status": status,
            "ready": ready,
            "downloaded": ready,
            "cached": ready,
            "source": "state.downloads",
        }
        if shard_tag:
            shard_item["shardType"] = shard_tag
        for source_key, target_key in (
            ("deviceRank", "deviceRank"),
            ("device_rank", "deviceRank"),
            ("worldSize", "worldSize"),
            ("world_size", "worldSize"),
            ("nLayers", "nLayers"),
            ("n_layers", "nLayers"),
        ):
            if source_key in shard:
                shard_item[target_key] = shard[source_key]
        _copy_shard_readiness_fields(shard_item, payload, shard)
        error_message = payload.get("errorMessage") or payload.get("error_message")
        if error_message:
            shard_item["errorMessage"] = str(error_message)

        entry = _model_shard_inventory_entry(
            inventory,
            model_id,
            source="state.downloads",
        )
        entry["shards"].append(shard_item)


def _merge_chunk_cache_model_shard_inventory(
    inventory: dict[str, Any],
    *,
    model_distribution: Any | None,
    wallet_policy: Any | None,
) -> None:
    if model_distribution is None:
        return
    try:
        manifests = model_distribution.list_model_package_manifests(
            policy=wallet_policy,
        )
        cached_records = model_distribution.list_cached_chunks(policy=wallet_policy)
    except Exception:
        return
    records_by_manifest: dict[tuple[str, str], dict[str, Any]] = {}
    for record in cached_records or []:
        catalog_id = str(getattr(record, "catalog_id", "") or "").strip()
        version = str(getattr(record, "version", "") or "").strip()
        chunk_id = str(getattr(record, "chunk_id", "") or "").strip()
        if catalog_id and version and chunk_id:
            records_by_manifest.setdefault((catalog_id, version), {})[chunk_id] = record

    for manifest in manifests or []:
        model_id = str(getattr(manifest, "model_id", "") or "").strip()
        catalog_id = str(getattr(manifest, "catalog_id", "") or "").strip()
        version = str(getattr(manifest, "version", "") or "").strip()
        if not model_id or not catalog_id or not version:
            continue
        manifest_records = records_by_manifest.get((catalog_id, version), {})
        if not manifest_records:
            continue
        present_chunk_ids = set(manifest_records.keys())
        try:
            default_coverage = manifest.compute_default_chunk_coverage(
                present_chunk_ids,
            )
            missing_default_chunk_ids = [
                str(chunk_id) for chunk_id in default_coverage.missing_chunk_ids
            ]
            default_ready = bool(default_coverage.ready)
        except Exception:
            missing_default_chunk_ids = []
            default_ready = True

        entry = _model_shard_inventory_entry(
            inventory,
            model_id,
            source="state.downloads+chunk-cache"
            if model_id in inventory
            else "chunk-cache",
        )
        entry["catalogId"] = catalog_id
        entry["manifestVersion"] = version
        chunks = getattr(manifest, "chunks", []) or []
        for chunk in chunks:
            chunk_id = str(getattr(chunk, "chunk_id", "") or "").strip()
            if chunk_id not in manifest_records:
                continue
            layer_start = getattr(chunk, "layer_start", None)
            layer_end = getattr(chunk, "layer_end", None)
            try:
                layer_start = int(layer_start)
                layer_end = int(layer_end)
            except (TypeError, ValueError):
                continue
            if layer_end <= layer_start:
                continue
            encrypted_at_rest = bool(getattr(chunk, "encrypted_at_rest", False))
            ready = bool(default_ready and not encrypted_at_rest)
            shard_item = {
                "layerStart": layer_start,
                "layerEnd": layer_end,
                "status": "cached" if ready else "cached_blocked",
                "ready": ready,
                "downloaded": True,
                "cached": True,
                "source": "chunk-cache",
                "catalogId": catalog_id,
                "manifestVersion": version,
                "chunkId": chunk_id,
                "chunkManifestVerified": True,
                "cacheVerified": True,
                "encryptedAtRest": encrypted_at_rest,
                "decryptionKeyAvailable": not encrypted_at_rest,
                "defaultChunkCoverageReady": default_ready,
            }
            if missing_default_chunk_ids:
                shard_item["missingDefaultChunkIds"] = missing_default_chunk_ids
            entry["shards"].append(shard_item)


def _model_shard_inventory_entry(
    inventory: dict[str, Any],
    model_id: str,
    *,
    source: str,
) -> dict[str, Any]:
    entry = inventory.setdefault(
        model_id,
        {
            "modelId": model_id,
            "source": source,
            "status": "unknown",
            "shards": [],
        },
    )
    entry["source"] = source
    return entry


def _copy_shard_readiness_fields(
    target: dict[str, Any],
    *sources: dict[str, Any],
) -> None:
    field_pairs = (
        ("chunkManifestVerified", "chunkManifestVerified"),
        ("chunk_manifest_verified", "chunkManifestVerified"),
        ("verifiedChunkManifest", "chunkManifestVerified"),
        ("verified_chunk_manifest", "chunkManifestVerified"),
        ("manifestVerified", "chunkManifestVerified"),
        ("manifest_verified", "chunkManifestVerified"),
        ("integrityVerified", "integrityVerified"),
        ("integrity_verified", "integrityVerified"),
        ("hashVerified", "integrityVerified"),
        ("hash_verified", "integrityVerified"),
        ("cacheVerified", "cacheVerified"),
        ("cache_verified", "cacheVerified"),
        ("encryptedAtRest", "encryptedAtRest"),
        ("encrypted_at_rest", "encryptedAtRest"),
        ("encryptedCache", "encryptedAtRest"),
        ("encrypted_cache", "encryptedAtRest"),
        ("decryptionKeyAvailable", "decryptionKeyAvailable"),
        ("decryption_key_available", "decryptionKeyAvailable"),
        ("shardKeyAvailable", "decryptionKeyAvailable"),
        ("shard_key_available", "decryptionKeyAvailable"),
        ("materialized", "materialized"),
        ("canLoadBeforeDeadline", "canLoadBeforeDeadline"),
        ("can_load_before_deadline", "canLoadBeforeDeadline"),
        ("canDownloadBeforeDeadline", "canLoadBeforeDeadline"),
        ("can_download_before_deadline", "canLoadBeforeDeadline"),
        ("downloadDeadlineAt", "downloadDeadlineAt"),
        ("download_deadline_at", "downloadDeadlineAt"),
        ("loadDeadlineAt", "downloadDeadlineAt"),
        ("load_deadline_at", "downloadDeadlineAt"),
        ("deadlineAt", "downloadDeadlineAt"),
        ("deadline_at", "downloadDeadlineAt"),
    )
    for source in sources:
        for source_key, target_key in field_pairs:
            if source_key in source and target_key not in target:
                target[target_key] = source[source_key]


def _unwrap_tagged_payload(value: Any) -> tuple[str | None, dict[str, Any] | None]:
    if not isinstance(value, dict):
        return None, None
    if len(value) == 1:
        tag, payload = next(iter(value.items()))
        if isinstance(payload, dict):
            return str(tag), payload
    tag = str(value.get("type") or value.get("status") or "").strip() or None
    return tag, value


def _as_mapping(value: Any) -> dict[str, Any]:
    tag, payload = _unwrap_tagged_payload(value)
    if payload is not None:
        return payload
    return dict(value) if isinstance(value, dict) else {}


def _optional_int_field(payload: dict[str, Any], *field_names: str) -> int | None:
    for field_name in field_names:
        if field_name not in payload:
            continue
        try:
            return int(payload[field_name])
        except (TypeError, ValueError):
            return None
    return None


def _download_status_for_inventory(
    tag: str | None,
    payload: dict[str, Any],
) -> tuple[str, bool]:
    raw = str(tag or payload.get("status") or "").strip()
    normalized = raw.lower()
    if "completed" in normalized or normalized in {"ready", "downloaded", "cached"}:
        return "downloaded", True
    if "ongoing" in normalized or normalized in {"loading", "downloading"}:
        return "loading", False
    if "pending" in normalized or normalized == "queued":
        return "pending", False
    if "failed" in normalized or normalized == "error":
        return "failed", False
    return "unknown", False


def _ensure_cai_on_path() -> None:
    for candidate in _candidate_cai_src_paths():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        return


def _candidate_cai_src_paths() -> list[Path]:
    candidates: list[Path] = []

    env_src = os.environ.get("CAI_RUNTIME_SRC")
    if env_src:
        candidates.append(Path(env_src))

    env_repo = os.environ.get("CAI_RUNTIME_REPO")
    if env_repo:
        candidates.append(Path(env_repo) / "src")

    repo_root = Path(__file__).resolve().parents[3]
    candidates.append(repo_root.parent / "cai-compute-chain" / "src")

    candidates.append(Path("/mnt/d/MyFiles/MyWorkFile/vav/cai-compute-chain/src"))

    return [path for path in candidates if path.exists()]


def _compute_cell_worker_node_ids(state_payload: dict[str, Any]) -> list[str]:
    identities = state_payload.get("nodeIdentities") or {}
    if not isinstance(identities, dict):
        identities = {}

    explicit_workers = [
        str(node_id).strip()
        for node_id, identity in identities.items()
        if str(node_id).strip()
        and isinstance(identity, dict)
        and bool(identity.get("workerEnabled") or identity.get("worker_enabled"))
    ]
    if explicit_workers:
        return sorted(dict.fromkeys(explicit_workers))

    topology = state_payload.get("topology") or {}
    if isinstance(topology, dict) and isinstance(topology.get("nodes"), list):
        topology_nodes = [
            str(node_id).strip()
            for node_id in topology.get("nodes", [])
            if str(node_id).strip()
        ]
        if topology_nodes:
            return sorted(dict.fromkeys(topology_nodes))

    return sorted(
        dict.fromkeys(
            str(node_id).strip()
            for node_id in identities.keys()
            if str(node_id).strip()
        )
    )


def _execution_attempts_from_sources(*sources: Any) -> list[dict[str, Any]]:
    for source in sources:
        raw_attempts: Any = None
        if isinstance(source, dict):
            raw_attempts = source.get("executionAttempts") or source.get(
                "execution_attempts"
            )
        elif source is not None:
            raw_attempts = getattr(source, "execution_attempts", None)
        attempts = _normalize_execution_attempts(raw_attempts)
        if attempts:
            return attempts
    return []


def _normalize_execution_attempts(raw_attempts: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_attempts, list):
        return []
    attempts: list[dict[str, Any]] = []
    for item in raw_attempts:
        if not isinstance(item, dict):
            continue
        attempt = _optional_int(item.get("attempt"))
        status = str(item.get("status") or "").strip()
        participants = _string_list(item.get("participantNodeIds"))
        excluded = _string_list(item.get("excludedNodeIds"))
        message = str(item.get("message") or "").strip() or None
        error_type = str(item.get("errorType") or "").strip() or None
        phase = str(item.get("phase") or "").strip() or None
        phase_message = str(item.get("phaseMessage") or "").strip() or None
        timeout_sec = _optional_float(item.get("timeoutSec"))
        attempt_duration_ms = _optional_int(item.get("attemptDurationMs"))
        readiness_duration_ms = _optional_int(item.get("readinessDurationMs"))
        response_duration_ms = _optional_int(item.get("responseDurationMs"))
        attempt_summary = {
            "attempt": attempt,
            "status": status or None,
            "startedAt": item.get("startedAt"),
            "completedAt": item.get("completedAt"),
            "participantNodeIds": participants,
            "excludedNodeIds": excluded,
            "instanceId": item.get("instanceId"),
            "retryScheduled": bool(item.get("retryScheduled")),
            "errorType": error_type,
            "message": message,
            "phase": phase,
            "phaseStartedAt": item.get("phaseStartedAt"),
            "phaseMessage": phase_message,
        }
        if timeout_sec is not None:
            attempt_summary["timeoutSec"] = timeout_sec
        if attempt_duration_ms is not None:
            attempt_summary["attemptDurationMs"] = attempt_duration_ms
        if readiness_duration_ms is not None:
            attempt_summary["readinessDurationMs"] = readiness_duration_ms
        if response_duration_ms is not None:
            attempt_summary["responseDurationMs"] = response_duration_ms
        attempts.append(attempt_summary)
    return attempts


def _execution_attempt_status(
    attempts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not attempts:
        return None
    current = attempts[-1]
    attempt_number = _optional_int(current.get("attempt")) or len(attempts)
    max_attempt_number = max(
        [_env_positive_int("CAI_JOB_EXECUTION_MAX_ATTEMPTS", 3), attempt_number]
    )
    failed_attempt_count = sum(
        1
        for item in attempts
        if str(item.get("status") or "").strip() in {"failed", "retrying"}
    )
    status = {
        "attempt": attempt_number,
        "maxAttempts": max_attempt_number,
        "status": current.get("status"),
        "message": current.get("message"),
        "errorType": current.get("errorType"),
        "phase": current.get("phase"),
        "phaseStartedAt": current.get("phaseStartedAt"),
        "phaseMessage": current.get("phaseMessage"),
        "retryScheduled": bool(current.get("retryScheduled")),
        "participantNodeIds": current.get("participantNodeIds") or [],
        "excludedNodeIds": current.get("excludedNodeIds") or [],
        "failedAttemptCount": failed_attempt_count,
        "lastCompletedAt": current.get("completedAt"),
    }
    timeout_sec = _optional_float(current.get("timeoutSec"))
    if timeout_sec is not None:
        status["timeoutSec"] = timeout_sec
    attempt_duration_ms = _optional_int(current.get("attemptDurationMs"))
    if attempt_duration_ms is not None:
        status["attemptDurationMs"] = attempt_duration_ms
    readiness_duration_ms = _optional_int(current.get("readinessDurationMs"))
    if readiness_duration_ms is not None:
        status["readinessDurationMs"] = readiness_duration_ms
    response_duration_ms = _optional_int(current.get("responseDurationMs"))
    if response_duration_ms is not None:
        status["responseDurationMs"] = response_duration_ms
    return status


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        str(item).strip()
        for item in value
        if str(item).strip()
    ]


def _job_to_summary(job: Any) -> dict[str, Any] | None:
    if job is None:
        return None
    execution_attempts = _execution_attempts_from_sources(job)
    execution_attempt_status = _execution_attempt_status(execution_attempts)
    return {
        "jobId": getattr(job, "job_id", None),
        "status": getattr(job, "status", None),
        "modelId": getattr(job, "model_id", None),
        "requesterNodeId": getattr(job, "requester_node_id", None),
        "promptRedacted": True,
        "pricingMode": getattr(job, "pricing_mode", None),
        "pricingBasis": getattr(job, "pricing_basis", None),
        "reservedPromptTokens": getattr(job, "reserved_prompt_tokens", None),
        "reservedCompletionTokens": getattr(job, "reserved_completion_tokens", None),
        "reservedComputeCostAtomic": getattr(job, "requested_compute_cost_atomic", None),
        "receiptId": getattr(job, "receipt_id", None),
        "settlementId": getattr(job, "settlement_id", None),
        "executionAttempts": execution_attempts,
        "executionAttemptStatus": execution_attempt_status,
        "executionAttemptCount": len(execution_attempts),
    }


def _receipt_to_summary(receipt: Any) -> dict[str, Any] | None:
    if receipt is None:
        return None
    network_audit = getattr(receipt, "network_audit", None)
    execution_attempts = _execution_attempts_from_sources(network_audit)
    execution_attempt_status = _execution_attempt_status(execution_attempts)
    execution_strategy = (
        network_audit.get("llamaCppExecutionStrategy")
        if isinstance(network_audit, dict)
        else None
    )
    worker_payouts = [
        _receipt_worker_payout_to_summary(item)
        for item in (getattr(receipt, "worker_payouts", []) or [])
    ]
    worker_payout_total_atomic = sum(
        int(item.get("rewardAtomic") or 0)
        for item in worker_payouts
    )
    decentralized_chain_audit = _receipt_decentralized_chain_audit(
        receipt,
        worker_payouts=worker_payouts,
    )
    return {
        "receiptId": getattr(receipt, "receipt_id", None),
        "jobId": getattr(receipt, "job_id", None),
        "finishReason": getattr(receipt, "finish_reason", None),
        "outputText": getattr(receipt, "output_text", None),
        "instanceId": getattr(receipt, "instance_id", None),
        "pricingMode": getattr(receipt, "pricing_mode", None),
        "pricingBasis": getattr(receipt, "pricing_basis", None),
        "promptTokens": getattr(receipt, "prompt_tokens", None),
        "completionTokens": getattr(receipt, "completion_tokens", None),
        "totalTokens": getattr(receipt, "total_tokens", None),
        "reservedPromptTokens": getattr(receipt, "reserved_prompt_tokens", None),
        "reservedCompletionTokens": getattr(receipt, "reserved_completion_tokens", None),
        "reservedComputeCostAtomic": getattr(receipt, "reserved_compute_cost_atomic", None),
        "actualComputeCostAtomic": getattr(receipt, "actual_compute_cost_atomic", None),
        "reservationSurplusAtomic": getattr(receipt, "reservation_surplus_atomic", None),
        "usagePriced": getattr(receipt, "usage_priced", False),
        "payoutCount": len(worker_payouts),
        "workerPayoutTotalAtomic": worker_payout_total_atomic,
        "workerPayouts": worker_payouts,
        "decentralizedChainAudit": decentralized_chain_audit,
        "transportMode": (
            network_audit.get("transportMode")
            if isinstance(network_audit, dict)
            else None
        ),
        "participantCount": (
            network_audit.get("participantCount")
            if isinstance(network_audit, dict)
            else None
        ),
        "decentralizedExecution": (
            network_audit.get("decentralizedExecution")
            if isinstance(network_audit, dict)
            else None
        ),
        "llamaCppExecutionMode": (
            execution_strategy.get("executionMode")
            if isinstance(execution_strategy, dict)
            else None
        ),
        "caiOwnedTransportExecuted": (
            network_audit.get("caiOwnedTransportExecuted")
            if isinstance(network_audit, dict)
            else None
        ),
        "caiOwnedTransportProofError": (
            network_audit.get("caiOwnedTransportProofError")
            if isinstance(network_audit, dict)
            else None
        ),
        "executionAttempts": execution_attempts,
        "executionAttemptStatus": execution_attempt_status,
        "executionAttemptCount": len(execution_attempts),
        "networkAudit": network_audit if isinstance(network_audit, dict) else None,
    }


def _receipt_worker_payout_to_summary(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"value": item}
    return {
        "nodeId": item.get("node_id") or item.get("nodeId"),
        "runnerId": item.get("runner_id") or item.get("runnerId"),
        "layerStart": item.get("layer_start") or item.get("layerStart"),
        "layerEnd": item.get("layer_end") or item.get("layerEnd"),
        "layerCount": item.get("layer_count") or item.get("layerCount"),
        "shareBps": item.get("share_bps") or item.get("shareBps"),
        "rewardAtomic": int(item.get("reward_atomic") or item.get("rewardAtomic") or 0),
    }


def _receipt_decentralized_chain_audit(
    receipt: Any,
    *,
    worker_payouts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    network_audit = getattr(receipt, "network_audit", None)
    network_audit = network_audit if isinstance(network_audit, dict) else {}
    proof = network_audit.get("caiOwnedTransportExecutionProof")
    proof = proof if isinstance(proof, dict) else {}
    execution_audit = proof.get("executionAudit")
    execution_audit = execution_audit if isinstance(execution_audit, dict) else {}
    execution_dag = execution_audit.get("executionDag")
    execution_dag = execution_dag if isinstance(execution_dag, dict) else {}
    shard_receipts = proof.get("shardReceipts")
    shard_receipts = shard_receipts if isinstance(shard_receipts, list) else []
    payout_items = (
        list(worker_payouts)
        if worker_payouts is not None
        else [
            _receipt_worker_payout_to_summary(item)
            for item in (getattr(receipt, "worker_payouts", []) or [])
        ]
    )

    executor_node_ids = _clean_text_list(proof.get("executorNodeIds"))
    if not executor_node_ids:
        executor_node_ids = _clean_text_list(execution_dag.get("executorNodeIds"))
    if not executor_node_ids:
        executor_node_ids = _clean_text_list(
            item.get("nodeId") for item in payout_items if isinstance(item, dict)
        )
    if not executor_node_ids:
        executor_node_ids = _clean_text_list(network_audit.get("participantNodeIds"))

    participant_node_ids = _clean_text_list(network_audit.get("participantNodeIds"))
    if not participant_node_ids:
        participant_node_ids = _clean_text_list(proof.get("participantNodeIds"))
    if not participant_node_ids:
        participant_node_ids = _clean_text_list(execution_dag.get("participantNodeIds"))
    requester_node_id = (
        str(
            network_audit.get("requesterNodeId")
            or proof.get("requesterNodeId")
            or execution_dag.get("requesterNodeId")
            or ""
        ).strip()
        or None
    )
    coordinator_node_id = (
        str(
            network_audit.get("coordinatorNodeId")
            or proof.get("coordinatorNodeId")
            or execution_dag.get("coordinatorNodeId")
            or requester_node_id
            or ""
        ).strip()
        or None
    )
    expected_stage_ids = _clean_text_list(execution_dag.get("expectedStageIds"))
    processed_stage_ids = _clean_text_list(execution_dag.get("processedStageIds"))
    final_output_batch_ids = _clean_text_list(execution_dag.get("finalOutputBatchIds"))
    proof_metrics = _aggregate_shard_receipt_metrics(shard_receipts)
    worker_payout_total_atomic = sum(
        int(item.get("rewardAtomic") or 0)
        for item in payout_items
        if isinstance(item, dict)
    )

    return {
        "schemaVersion": 1,
        "requesterNodeId": requester_node_id,
        "coordinatorNodeId": coordinator_node_id,
        "participantNodeIds": participant_node_ids,
        "participantCount": _optional_int(network_audit.get("participantCount"))
        or len(participant_node_ids),
        "executorNodeIds": executor_node_ids,
        "executorCount": len(executor_node_ids),
        "transportMode": network_audit.get("transportMode"),
        "decentralizedExecution": network_audit.get("decentralizedExecution"),
        "route": {
            "directSocketLinkCount": _optional_int(
                network_audit.get("directSocketLinkCount")
            )
            or 0,
            "directBidirectionalLinkCount": _optional_int(
                network_audit.get("directBidirectionalLinkCount")
            )
            or 0,
            "overlayLinkCount": _optional_int(network_audit.get("overlayLinkCount"))
            or 0,
            "relayHopsUsed": bool(network_audit.get("relayHopsUsed")),
            "relayBottleneckRisk": bool(network_audit.get("relayBottleneckRisk")),
            "checkedDirectSocketLinkCount": len(
                network_audit.get("checkedDirectSocketLinks") or []
            )
            if isinstance(network_audit.get("checkedDirectSocketLinks"), list)
            else 0,
            "checkedRelayRouteCount": len(network_audit.get("checkedRelayRoutes") or [])
            if isinstance(network_audit.get("checkedRelayRoutes"), list)
            else 0,
            "relayRouteCandidateCount": _optional_int(
                network_audit.get("relayRouteCandidateCount")
            )
            or 0,
            "relayCoordinatorCandidateCount": _optional_int(
                network_audit.get("relayCoordinatorCandidateCount")
            )
            or 0,
            "activeRelayTransitNodeIds": _clean_text_list(
                network_audit.get("activeRelayTransitNodeIds")
            ),
        },
        "proof": {
            "executed": bool(network_audit.get("caiOwnedTransportExecuted")),
            "error": network_audit.get("caiOwnedTransportProofError"),
            "sessionId": proof.get("sessionId"),
            "instanceId": proof.get("instanceId"),
            "verified": bool(execution_audit.get("verified")),
            "stageCount": len(expected_stage_ids),
            "processedStageCount": len(processed_stage_ids),
            "missingStageCount": len(_clean_text_list(execution_dag.get("missingStageIds"))),
            "finalOutputBatchCount": len(final_output_batch_ids),
        },
        "tokens": {
            "source": getattr(receipt, "token_usage_source", None),
            "promptTokens": getattr(receipt, "prompt_tokens", None),
            "completionTokens": getattr(receipt, "completion_tokens", None),
            "totalTokens": getattr(receipt, "total_tokens", None),
            "proofPromptTokenCount": proof_metrics["promptTokenCount"],
            "proofCompletionTokenCount": proof_metrics["completionTokenCount"],
            "proofInputTokenCount": proof_metrics["inputTokenCount"],
            "proofOutputTokenCount": proof_metrics["outputTokenCount"],
        },
        "bytes": {
            "payloadSizeBytes": proof_metrics["payloadSizeBytes"],
            "outputPayloadSizeBytes": proof_metrics["outputPayloadSizeBytes"],
        },
        "reward": {
            "payoutCount": len(payout_items),
            "workerPayoutTotalAtomic": worker_payout_total_atomic,
            "payoutNodes": _clean_text_list(
                item.get("nodeId") for item in payout_items if isinstance(item, dict)
            ),
        },
    }


def _aggregate_shard_receipt_metrics(
    shard_receipts: list[Any],
) -> dict[str, int]:
    totals = {
        "promptTokenCount": 0,
        "completionTokenCount": 0,
        "inputTokenCount": 0,
        "outputTokenCount": 0,
        "payloadSizeBytes": 0,
        "outputPayloadSizeBytes": 0,
    }
    for receipt in shard_receipts:
        if not isinstance(receipt, dict):
            continue
        metrics = receipt.get("metrics")
        if not isinstance(metrics, dict):
            continue
        for field_name in totals:
            value = _optional_int(metrics.get(field_name))
            if value is not None:
                totals[field_name] += max(0, value)
    return totals


def _clean_text_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        candidates = [values]
    else:
        try:
            candidates = list(values)
        except TypeError:
            candidates = [values]
    cleaned: list[str] = []
    for value in candidates:
        text = str(value or "").strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _build_safety_summary(
    *,
    money_policy: Any,
    chain_status: dict[str, Any],
    validator_summary: dict[str, Any],
) -> dict[str, Any]:
    network = str(
        chain_status.get("network")
        or getattr(getattr(money_policy, "chain_network", None), "value", None)
        or "mainnet"
    ).strip().lower()
    warnings: list[dict[str, str]] = []

    if network == "testnet":
        warnings.append(
            {
                "code": "testnet",
                "severity": "warning",
                "title": "Testnet mode",
                "message": "This node is using testnet state; balances and rewards are isolated from mainnet.",
            }
        )
        mode = "testnet"
    else:
        warnings.append(
            {
                "code": "mainnet_alpha",
                "severity": "warning",
                "title": "Mainnet alpha",
                "message": "This is an alpha mainnet path; finality, rewards and decentralized compute are still guarded.",
            }
        )
        mode = "mainnet_alpha"

    validator_set_size = _optional_int(
        _mapping_value(
            validator_summary,
            "validator_set_size",
            "validatorSetSize",
        )
    )
    if validator_set_size is None or validator_set_size <= 1:
        warnings.append(
            {
                "code": "single_validator_guarded_alpha",
                "severity": "warning",
                "title": "Guarded validator mode",
                "message": "Validator finality is not quorum-backed yet; treat settlement as guarded alpha.",
            }
        )

    if chain_status.get("valid") is False:
        warnings.append(
            {
                "code": "chain_invalid",
                "severity": "critical",
                "title": "Chain validation warning",
                "message": "Local chain summary reports validation errors; do not trust displayed balances until repaired.",
            }
        )

    return {
        "mode": mode,
        "network": network,
        "warningCount": len(warnings),
        "warnings": warnings,
    }


def _journal_to_summary(entry: Any, money_policy: Any, to_coins: Any) -> dict[str, Any] | None:
    if entry is None:
        return None
    return {
        "entryId": getattr(entry, "entry_id", None),
        "source": "local",
        "eventType": getattr(entry, "event_type", None),
        "createdAt": getattr(entry, "created_at", None),
        "counterpartyAddress": getattr(entry, "counterparty_address", None),
        "amountAtomic": getattr(entry, "amount_atomic", None),
        "amountCoins": _atomic_to_coins(getattr(entry, "amount_atomic", None), money_policy, to_coins),
        "txFeeAtomic": getattr(entry, "tx_fee_atomic", None),
        "txFeeCoins": _atomic_to_coins(getattr(entry, "tx_fee_atomic", None), money_policy, to_coins),
        "note": getattr(entry, "note", None),
    }


def _wallet_activity_history_summaries(
    modules: Any,
    *,
    active_wallet: Any | None,
    active_wallet_id: Any,
    wallet_policy: Any,
    money_policy: Any,
    to_coins: Any,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    chain_items = _wallet_chain_history_from_modules(
        modules,
        active_wallet=active_wallet,
        wallet_policy=wallet_policy,
        limit=limit,
    )
    if chain_items:
        return [
            _chain_wallet_history_to_journal_summary(item, money_policy, to_coins)
            for item in chain_items
        ]

    journal_items = modules.wallet.list_journal_entries(
        wallet_id=active_wallet_id,
        limit=limit,
        wallet_policy=wallet_policy,
    )
    return [
        item
        for item in (
            _journal_to_summary(entry, money_policy, to_coins) for entry in journal_items
        )
        if item is not None
    ]


def _wallet_chain_history_from_modules(
    modules: Any,
    *,
    active_wallet: Any | None,
    wallet_policy: Any,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    address = str(getattr(active_wallet, "address", "") or "").strip()
    if not address:
        return []
    chain_module = getattr(modules, "chain", None)
    history_fn = getattr(chain_module, "chain_address_history", None)
    if history_fn is None:
        return []
    try:
        history = history_fn(
            address,
            wallet_policy,
            limit=limit,
            newest_first=True,
        )
    except Exception:  # noqa: BLE001
        return []
    return history if isinstance(history, list) else []


def _chain_wallet_history_to_journal_summary(
    item: Any,
    money_policy: Any,
    to_coins: Any,
) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"source": "chain", "value": item}
    amount_atomic = _optional_int(_mapping_value(item, "delta_atomic", "deltaAtomic"))
    balance_after_atomic = _optional_int(
        _mapping_value(item, "balance_after_atomic", "balanceAfterAtomic")
    )
    tx_id = _mapping_value(item, "tx_id", "txId")
    return {
        "entryId": tx_id,
        "source": "chain",
        "eventType": _mapping_value(item, "tx_type", "txType"),
        "createdAt": _mapping_value(item, "created_at", "createdAt")
        or _mapping_value(item, "block_created_at", "blockCreatedAt"),
        "counterpartyAddress": None,
        "amountAtomic": amount_atomic,
        "amountCoins": _atomic_to_coins(amount_atomic, money_policy, to_coins),
        "txFeeAtomic": None,
        "txFeeCoins": None,
        "note": _mapping_value(item, "note"),
        "txId": tx_id,
        "blockHeight": _mapping_value(item, "block_height", "blockHeight"),
        "blockHash": _mapping_value(item, "block_hash", "blockHash"),
        "balanceAfterAtomic": balance_after_atomic,
        "balanceAfterCoins": _atomic_to_coins(
            balance_after_atomic,
            money_policy,
            to_coins,
        ),
    }


def _job_to_history_summary(job: Any, receipt: Any | None = None) -> dict[str, Any] | None:
    if job is None:
        return None
    output_text = getattr(receipt, "output_text", None) if receipt is not None else None
    network_audit = getattr(receipt, "network_audit", None) if receipt is not None else None
    execution_attempts = _execution_attempts_from_sources(job, network_audit)
    execution_attempt_status = _execution_attempt_status(execution_attempts)
    execution_strategy = (
        network_audit.get("llamaCppExecutionStrategy")
        if isinstance(network_audit, dict)
        else None
    )
    decentralized_chain_audit = (
        _receipt_decentralized_chain_audit(receipt)
        if receipt is not None
        else None
    )
    return {
        "jobId": getattr(job, "job_id", None),
        "createdAt": getattr(job, "created_at", None),
        "status": getattr(job, "status", None),
        "modelId": getattr(job, "model_id", None),
        "requesterNodeId": getattr(job, "requester_node_id", None),
        "promptRedacted": True,
        "pricingMode": getattr(job, "pricing_mode", None),
        "pricingBasis": getattr(job, "pricing_basis", None),
        "receiptId": getattr(job, "receipt_id", None),
        "settlementId": getattr(job, "settlement_id", None),
        "outputText": output_text,
        "lastError": getattr(job, "last_error", None),
        "transportMode": (
            network_audit.get("transportMode")
            if isinstance(network_audit, dict)
            else None
        ),
        "participantCount": (
            network_audit.get("participantCount")
            if isinstance(network_audit, dict)
            else None
        ),
        "decentralizedExecution": (
            network_audit.get("decentralizedExecution")
            if isinstance(network_audit, dict)
            else None
        ),
        "relayBottleneckRisk": (
            network_audit.get("relayBottleneckRisk")
            if isinstance(network_audit, dict)
            else None
        ),
        "llamaCppExecutionMode": (
            execution_strategy.get("executionMode")
            if isinstance(execution_strategy, dict)
            else None
        ),
        "caiOwnedTransportExecuted": (
            network_audit.get("caiOwnedTransportExecuted")
            if isinstance(network_audit, dict)
            else None
        ),
        "caiOwnedTransportProofError": (
            network_audit.get("caiOwnedTransportProofError")
            if isinstance(network_audit, dict)
            else None
        ),
        "executionAttempts": execution_attempts,
        "executionAttemptStatus": execution_attempt_status,
        "executionAttemptCount": len(execution_attempts),
        "decentralizedChainAudit": decentralized_chain_audit,
        "networkAudit": network_audit if isinstance(network_audit, dict) else None,
    }


def _payout_to_summary(
    payout: Any,
    money_policy: Any | None = None,
    to_coins: Any | None = None,
) -> dict[str, Any] | None:
    if payout is None:
        return None
    return {
        "createdAt": getattr(payout, "created_at", None),
        "nodeId": getattr(payout, "node_id", None),
        "status": getattr(payout, "status", None),
        "shareBps": getattr(payout, "share_bps", None),
        "rewardAtomic": getattr(payout, "reward_atomic", None),
        "rewardCoins": _atomic_to_coins(getattr(payout, "reward_atomic", None), money_policy, to_coins),
        "settlementId": getattr(payout, "settlement_id", None),
    }


def _resolve_latest_settlement_for_job(settlements: list[Any], job: Any | None) -> Any | None:
    settlement_id = getattr(job, "settlement_id", None) if job is not None else None
    if settlement_id:
        for item in settlements:
            if getattr(item, "settlement_id", None) == settlement_id:
                return item
    return settlements[0] if settlements else None


def _settlement_to_summary(
    settlement: Any,
    money_policy: Any | None = None,
    to_coins: Any | None = None,
    *,
    chain_transactions: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if settlement is None:
        return None
    compute_cost_atomic = int(getattr(settlement, "compute_cost_atomic", 0) or 0)
    tx_fee_atomic = int(getattr(settlement, "tx_fee_atomic", 0) or 0)
    settlement_fee_atomic = int(getattr(settlement, "settlement_fee_atomic", 0) or 0)
    ai_development_fee_atomic = int(
        getattr(settlement, "ai_development_fee_atomic", 0) or 0
    )
    worker_reward_atomic = int(getattr(settlement, "worker_reward_atomic", 0) or 0)
    chain_transaction_summaries = [
        _chain_transaction_to_summary(item, money_policy, to_coins)
        for item in (chain_transactions or [])
    ]
    return {
        "settlementId": getattr(settlement, "settlement_id", None),
        "status": getattr(settlement, "status", None),
        "fundingSource": getattr(settlement, "funding_source", None),
        "computeCostAtomic": compute_cost_atomic,
        "computeCostCoins": _atomic_to_coins(compute_cost_atomic, money_policy, to_coins),
        "txFeeAtomic": tx_fee_atomic,
        "settlementFeeAtomic": settlement_fee_atomic,
        "aiDevelopmentFeeAtomic": ai_development_fee_atomic,
        "workerRewardAtomic": worker_reward_atomic,
        "sourceWalletDebitAtomic": int(
            getattr(settlement, "source_wallet_debit_atomic", 0) or 0
        ),
        "reserveDebitAtomic": int(getattr(settlement, "reserve_debit_atomic", 0) or 0),
        "acceptedAttestations": getattr(settlement, "accepted_attestations", None),
        "rejectedAttestations": getattr(settlement, "rejected_attestations", None),
        "acceptedBondAtomic": getattr(settlement, "accepted_bond_atomic", None),
        "committeeValidatorIds": list(
            getattr(settlement, "committee_validator_ids", []) or []
        ),
        "appliedAt": getattr(settlement, "applied_at", None),
        "balanceAudit": dict(getattr(settlement, "balance_audit", {}) or {}),
        "chainRecorded": bool(chain_transaction_summaries),
        "chainTransactionCount": len(chain_transaction_summaries),
        "chainTransactions": chain_transaction_summaries,
    }


def _settlement_chain_history_from_modules(
    modules: Any,
    settlement_id: Any,
    wallet_policy: Any,
    *,
    limit: int = 32,
) -> list[dict[str, Any]]:
    normalized_settlement_id = str(settlement_id or "").strip()
    if not normalized_settlement_id:
        return []
    chain_module = getattr(modules, "chain", None)
    history_fn = getattr(chain_module, "chain_settlement_history", None)
    if history_fn is None:
        return []
    try:
        history = history_fn(
            normalized_settlement_id,
            wallet_policy,
            limit=limit,
            newest_first=True,
        )
    except Exception:  # noqa: BLE001
        return []
    return history if isinstance(history, list) else []


def _chain_transaction_to_summary(
    item: Any,
    money_policy: Any | None = None,
    to_coins: Any | None = None,
) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"value": item}
    delta_atomic = _optional_int(_mapping_value(item, "delta_atomic", "deltaAtomic"))
    balance_after_atomic = _optional_int(
        _mapping_value(item, "balance_after_atomic", "balanceAfterAtomic")
    )
    return {
        "txId": _mapping_value(item, "tx_id", "txId"),
        "txType": _mapping_value(item, "tx_type", "txType"),
        "address": _mapping_value(item, "address"),
        "walletId": _mapping_value(item, "wallet_id", "walletId"),
        "deltaAtomic": delta_atomic,
        "deltaCoins": _atomic_to_coins(delta_atomic, money_policy, to_coins),
        "balanceAfterAtomic": balance_after_atomic,
        "balanceAfterCoins": _atomic_to_coins(
            balance_after_atomic,
            money_policy,
            to_coins,
        ),
        "jobId": _mapping_value(item, "job_id", "jobId"),
        "receiptId": _mapping_value(item, "receipt_id", "receiptId"),
        "settlementId": _mapping_value(item, "settlement_id", "settlementId"),
        "payoutId": _mapping_value(item, "payout_id", "payoutId"),
        "validatorId": _mapping_value(item, "validator_id", "validatorId"),
        "nonce": _mapping_value(item, "nonce"),
        "note": _mapping_value(item, "note"),
        "metadata": _mapping_value(item, "metadata") or {},
        "createdAt": _mapping_value(item, "created_at", "createdAt"),
        "blockHeight": _mapping_value(item, "block_height", "blockHeight"),
        "blockHash": _mapping_value(item, "block_hash", "blockHash"),
        "blockCreatedAt": _mapping_value(
            item,
            "block_created_at",
            "blockCreatedAt",
        ),
    }


def _mapping_value(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item:
            return item[key]
    return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_base64_bytes(value: Any, field_name: str) -> bytes | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a base64 string.")
    if not value:
        return b""
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except Exception as exc:
        raise ValueError(f"{field_name} must be valid base64.") from exc


def _optional_dict(payload: dict[str, Any], *keys: str) -> dict[str, Any] | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return None


def _optional_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    clean = str(value).strip().lower()
    if clean in {"1", "true", "yes", "on"}:
        return True
    if clean in {"0", "false", "no", "off"}:
        return False
    return default


def _worker_capability_source_endpoint(source_url: str) -> str:
    normalized = str(source_url or "").strip().rstrip("/")
    if not normalized:
        raise ValueError("Worker capability source URL is required.")
    if normalized.endswith("/v1/cai/node-capabilities"):
        return normalized
    return f"{normalized}/v1/cai/node-capabilities"


def _worker_capability_challenge_endpoint(source_url: str) -> str:
    normalized = str(source_url or "").strip().rstrip("/")
    if not normalized:
        raise ValueError("Worker capability source URL is required.")
    if normalized.endswith("/v1/cai/worker-capability/challenge"):
        return normalized
    if normalized.endswith("/v1/cai/node-capabilities"):
        return (
            normalized[: -len("/v1/cai/node-capabilities")]
            + "/v1/cai/worker-capability/challenge"
        )
    return f"{normalized}/v1/cai/worker-capability/challenge"


def _prepare_record_for_worker_challenge(
    record: Any,
    *,
    signer: dict[str, Any],
    active_wallet: Any | None,
    modules: Any,
) -> None:
    public_key_b64 = str(signer.get("public_key_b64") or "").strip()
    if not public_key_b64:
        raise ValueError("Worker challenge signer public key is missing.")
    pq_public_key_b64 = str(signer.get("pq_public_key_b64") or "").strip()
    if pq_public_key_b64:
        worker_key_address = modules.wallet_signing.hybrid_address_from_public_keys_b64(
            ed25519_public_key_b64=public_key_b64,
            pq_public_key_b64=pq_public_key_b64,
        )
    else:
        worker_key_address = modules.wallet_signing.address_from_public_key_b64(
            public_key_b64
        )
    raw_worker_reward_address = str(
        getattr(record, "worker_reward_address", "") or ""
    ).strip()
    worker_reward_address = (
        modules.wallet.normalize_address(raw_worker_reward_address)
        if raw_worker_reward_address
        else ""
    )
    active_wallet_address = str(getattr(active_wallet, "address", "") or "").strip()
    normalized_active_wallet_address = modules.wallet.normalize_address(
        active_wallet_address
    ) if active_wallet_address else ""
    if worker_reward_address and worker_reward_address != worker_key_address:
        raise ValueError("Worker capability signer does not match reward address.")
    if (
        normalized_active_wallet_address
        and normalized_active_wallet_address != worker_key_address
    ):
        raise ValueError("Active worker wallet does not match challenge signer.")
    if not getattr(record, "node_public_key_b64", None):
        setattr(record, "node_public_key_b64", public_key_b64)
    if pq_public_key_b64 and not getattr(record, "node_pq_public_key_b64", None):
        setattr(record, "node_pq_public_key_b64", pq_public_key_b64)
    if not getattr(record, "node_public_key_address", None):
        setattr(record, "node_public_key_address", worker_key_address)
    if not getattr(record, "payload_public_key_address", None):
        setattr(record, "payload_public_key_address", worker_key_address)


def _jsonable_transport_payload(value: Any) -> Any:
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, dict):
        return {
            str(key): _jsonable_transport_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_jsonable_transport_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable_transport_payload(item) for item in value]
    return value


def _dataclass_like(value: Any) -> dict[str, Any]:
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {"value": value}


def _env_positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, "") or default)
    except ValueError:
        return default
    return max(1, value)


def _atomic_to_coins(value: Any, money_policy: Any | None, to_coins: Any | None) -> str | None:
    if value is None or money_policy is None or to_coins is None:
        return None
    try:
        return to_coins(int(value), money_policy)
    except Exception:  # noqa: BLE001
        return None


def _extract_display_prompt_from_chat_payload(payload: dict[str, Any]) -> str:
    messages = payload.get("messages") or []
    if not isinstance(messages, list):
        return ""

    last_user_text = ""
    transcript_lines: list[str] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user").strip().lower() or "user"
        content_text = _stringify_chat_content(item.get("content"))
        if not content_text:
            continue
        transcript_lines.append(f"{role.upper()}: {content_text}")
        if role == "user":
            last_user_text = content_text

    return last_user_text or "\n\n".join(transcript_lines)


def _stringify_chat_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        content_type = str(content.get("type") or "").strip().lower()
        if content_type == "text":
            return str(content.get("text") or "").strip()
        if content_type == "image_url":
            return "[image]"
        return ""

    if isinstance(content, list):
        parts = [_stringify_chat_content(item) for item in content]
        return "\n".join(part for part in parts if part)

    return ""
