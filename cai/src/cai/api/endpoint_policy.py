# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class EndpointAccess(StrEnum):
    PUBLIC = "public"
    PEER = "peer"
    LOCAL_ONLY = "local_only"
    ADMIN_ONLY = "admin_only"


@dataclass(frozen=True)
class EndpointPolicy:
    method: str
    path: str
    access: EndpointAccess
    rationale: str


CAI_ENDPOINT_POLICIES: tuple[EndpointPolicy, ...] = (
    EndpointPolicy("GET", "/cai/summary", EndpointAccess.PUBLIC, "Read-only UI/network summary; can be restricted by CAI_SUMMARY_LOCAL_ONLY."),
    EndpointPolicy("GET", "/v1/cai/summary", EndpointAccess.PUBLIC, "Read-only UI/network summary; can be restricted by CAI_SUMMARY_LOCAL_ONLY."),
    EndpointPolicy("GET", "/v1/cai/chain", EndpointAccess.PEER, "Peer chain export with network metadata and optional signature."),
    EndpointPolicy("POST", "/v1/cai/chain/sync", EndpointAccess.PEER, "Peer chain import; validates network metadata and signed payloads."),
    EndpointPolicy("GET", "/v1/cai/validators", EndpointAccess.PEER, "Peer validator set export."),
    EndpointPolicy("POST", "/v1/cai/validators/sync", EndpointAccess.PEER, "Peer validator set sync."),
    EndpointPolicy("GET", "/v1/cai/validator-evidence", EndpointAccess.PEER, "Peer validator evidence export."),
    EndpointPolicy("POST", "/v1/cai/validator-evidence/sync", EndpointAccess.PEER, "Peer validator evidence sync."),
    EndpointPolicy("GET", "/v1/cai/node-capabilities", EndpointAccess.PEER, "Peer node capability export."),
    EndpointPolicy("POST", "/v1/cai/node-capabilities/sync", EndpointAccess.PEER, "Peer node capability sync."),
    EndpointPolicy("GET", "/v1/cai/worker-capability-attestations", EndpointAccess.PEER, "Peer validator-signed worker capability attestation export."),
    EndpointPolicy("POST", "/v1/cai/worker-capability-attestations/sync", EndpointAccess.PEER, "Peer validator-signed worker capability attestation sync."),
    EndpointPolicy("GET", "/v1/cai/route-health", EndpointAccess.PEER, "Peer route health export."),
    EndpointPolicy("GET", "/v1/cai/compute-cells", EndpointAccess.PEER, "Peer compute-cell readiness export."),
    EndpointPolicy("GET", "/v1/cai/transport/sessions", EndpointAccess.LOCAL_ONLY, "Local CAI-owned transport session/proof state."),
    EndpointPolicy("GET", "/v1/cai/transport/batch-inbox", EndpointAccess.LOCAL_ONLY, "Local CAI-owned transport batch inbox for shard runtime."),
    EndpointPolicy("POST", "/v1/cai/transport/batch-inbox/claim-next", EndpointAccess.LOCAL_ONLY, "Local CAI-owned transport claim-next for shard runtime."),
    EndpointPolicy("POST", "/v1/cai/transport/sessions", EndpointAccess.LOCAL_ONLY, "Local CAI-owned transport session creation."),
    EndpointPolicy("POST", "/v1/cai/transport/sessions/{session_id}/offer", EndpointAccess.PEER, "Peer CAI-owned transport session offer for shard participants."),
    EndpointPolicy("POST", "/v1/cai/transport/sessions/{session_id}/batches", EndpointAccess.LOCAL_ONLY, "Local CAI-owned transport batch metadata append."),
    EndpointPolicy("POST", "/v1/cai/transport/sessions/{session_id}/batch-envelopes", EndpointAccess.PEER, "Peer CAI-owned transport batch envelope ingress for shard participants."),
    EndpointPolicy("POST", "/v1/cai/transport/overlay/send", EndpointAccess.PEER, "Peer CAI-owned transport overlay relay ingress for shard participants behind asymmetric NAT/proxy routes."),
    EndpointPolicy("POST", "/v1/cai/transport/sessions/{session_id}/batches/{batch_id}/status", EndpointAccess.LOCAL_ONLY, "Local CAI-owned transport batch runtime status update."),
    EndpointPolicy("POST", "/v1/cai/transport/sessions/{session_id}/batches/{batch_id}/claim", EndpointAccess.LOCAL_ONLY, "Local CAI-owned transport batch runtime claim."),
    EndpointPolicy("POST", "/v1/cai/transport/sessions/{session_id}/batches/{batch_id}/heartbeat", EndpointAccess.LOCAL_ONLY, "Local CAI-owned transport batch runtime lease heartbeat."),
    EndpointPolicy("POST", "/v1/cai/transport/sessions/{session_id}/batches/{batch_id}/complete-work-item", EndpointAccess.LOCAL_ONLY, "Local CAI-owned transport complete-work-item for shard runtime."),
    EndpointPolicy("POST", "/v1/cai/transport/sessions/{session_id}/batches/{batch_id}/fail-work-item", EndpointAccess.LOCAL_ONLY, "Local CAI-owned transport retry/fail policy for shard runtime."),
    EndpointPolicy("GET", "/v1/cai/transport/sessions/{session_id}/batches/{batch_id}/payload", EndpointAccess.LOCAL_ONLY, "Local CAI-owned transport batch payload download for shard runtime."),
    EndpointPolicy("POST", "/v1/cai/transport/sessions/{session_id}/shard-receipts", EndpointAccess.PEER, "Peer worker CAI-owned transport shard receipt submission."),
    EndpointPolicy("POST", "/v1/cai/transport/sessions/{session_id}/complete", EndpointAccess.LOCAL_ONLY, "Local CAI-owned transport proof completion."),
    EndpointPolicy("POST", "/v1/cai/transport/sessions/{session_id}/completion-notice", EndpointAccess.PEER, "Peer CAI-owned transport completion notice after requester proof verification."),
    EndpointPolicy("GET", "/v1/cai/transport/sessions/{session_id}/final-output", EndpointAccess.LOCAL_ONLY, "Local CAI-owned transport final output read."),
    EndpointPolicy("POST", "/v1/cai/transport/sessions/{session_id}/await-final-result", EndpointAccess.LOCAL_ONLY, "Local CAI-owned transport final result wait and proof completion."),
    EndpointPolicy("POST", "/v1/cai/transport/sessions/{session_id}/reconcile-timeouts", EndpointAccess.LOCAL_ONLY, "Local CAI-owned transport coordinator timeout reconciliation."),
    EndpointPolicy("POST", "/v1/cai/route-health/probe", EndpointAccess.LOCAL_ONLY, "Active route probing is local node control."),
    EndpointPolicy("GET", "/v1/cai/chunk-inventory", EndpointAccess.PEER, "Peer chunk inventory export."),
    EndpointPolicy("POST", "/v1/cai/chunk-inventory/sync", EndpointAccess.PEER, "Peer chunk inventory sync."),
    EndpointPolicy("GET", "/v1/cai/chunks", EndpointAccess.PEER, "Public-shared chunk transport; private packages are blocked."),
    EndpointPolicy("GET", "/v1/cai/history", EndpointAccess.LOCAL_ONLY, "Wallet/job/settlement history can reveal local user activity."),
    EndpointPolicy("GET", "/v1/cai/desktop/preferences", EndpointAccess.LOCAL_ONLY, "Desktop preferences are local UI state."),
    EndpointPolicy("PUT", "/v1/cai/desktop/preferences", EndpointAccess.LOCAL_ONLY, "Desktop preferences are local UI state."),
    EndpointPolicy("GET", "/v1/cai/update-manifest", EndpointAccess.ADMIN_ONLY, "Update server must be explicitly enabled."),
    EndpointPolicy("GET", "/v1/cai/update-package", EndpointAccess.ADMIN_ONLY, "Update server must be explicitly enabled."),
    EndpointPolicy("GET", "/v1/cai/update-package.zip", EndpointAccess.ADMIN_ONLY, "Update server must be explicitly enabled."),
    EndpointPolicy("POST", "/v1/cai/update/cancel", EndpointAccess.LOCAL_ONLY, "Local user control for a pending portable update."),
    EndpointPolicy("POST", "/v1/cai/update/activity", EndpointAccess.LOCAL_ONLY, "Local dashboard heartbeat for safe portable update timing."),
    EndpointPolicy("POST", "/v1/cai/chat/completions", EndpointAccess.PUBLIC, "User inference endpoint; bearer auth is enforced when configured."),
    EndpointPolicy("POST", "/v1/cai/settlement/attest", EndpointAccess.PEER, "Validator committee attestation endpoint."),
    EndpointPolicy("POST", "/v1/cai/worker-capability/challenge", EndpointAccess.PEER, "Peer worker answers a validator-signed capability freshness challenge."),
    EndpointPolicy("POST", "/v1/cai/worker-capability/attest", EndpointAccess.PEER, "Bonded validator worker capability probe and attestation endpoint."),
    EndpointPolicy("POST", "/v1/cai/validator-penalty/attest", EndpointAccess.PEER, "Validator penalty attestation endpoint."),
    EndpointPolicy("POST", "/v1/cai/wallet/create", EndpointAccess.LOCAL_ONLY, "Wallet mutation and secrets."),
    EndpointPolicy("POST", "/v1/cai/wallet/restore", EndpointAccess.LOCAL_ONLY, "Wallet mutation and seed phrase."),
    EndpointPolicy("POST", "/v1/cai/wallet/select", EndpointAccess.LOCAL_ONLY, "Wallet session mutation."),
    EndpointPolicy("POST", "/v1/cai/wallet/unlock", EndpointAccess.LOCAL_ONLY, "Wallet password unlock."),
    EndpointPolicy("POST", "/v1/cai/wallet/lock", EndpointAccess.LOCAL_ONLY, "Wallet session mutation."),
    EndpointPolicy("POST", "/v1/cai/wallet/logout", EndpointAccess.LOCAL_ONLY, "Wallet session mutation."),
    EndpointPolicy("POST", "/v1/cai/wallet/send", EndpointAccess.LOCAL_ONLY, "Wallet transfer mutation."),
    EndpointPolicy("POST", "/v1/cai/node/validator", EndpointAccess.LOCAL_ONLY, "Local node role mutation."),
    EndpointPolicy("POST", "/v1/cai/node/validator/unbond-complete", EndpointAccess.LOCAL_ONLY, "Local validator state mutation."),
    EndpointPolicy("POST", "/v1/cai/node/validator/unjail", EndpointAccess.LOCAL_ONLY, "Local validator state mutation."),
    EndpointPolicy("POST", "/v1/cai/node/validator/static-ip", EndpointAccess.LOCAL_ONLY, "Local validator configuration mutation."),
    EndpointPolicy("POST", "/v1/cai/node/worker", EndpointAccess.LOCAL_ONLY, "Local node role mutation."),
    EndpointPolicy("POST", "/v1/cai/node/relay", EndpointAccess.LOCAL_ONLY, "Local node role mutation."),
    EndpointPolicy("GET", "/v1/cai/relay/rpc/probe", EndpointAccess.PEER, "Relay connectivity probe endpoint."),
)


def endpoint_policy_index() -> dict[tuple[str, str], EndpointPolicy]:
    return {(item.method.upper(), item.path): item for item in CAI_ENDPOINT_POLICIES}


def lookup_endpoint_policy(method: str, path: str) -> EndpointPolicy | None:
    return endpoint_policy_index().get((str(method).upper(), path))
