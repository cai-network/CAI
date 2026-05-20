// SPDX-FileCopyrightText: 2025 cai Technologies Ltd
// SPDX-FileCopyrightText: 2026 CAI contributors
// SPDX-License-Identifier: Apache-2.0
/**
 * AppStore - Central state management for the cai dashboard
 *
 * Manages:
 * - Chat state (whether a conversation has started)
 * - Topology data from the cai server
 * - UI state for the topology/chat transition
 */

import { browser } from "$app/environment";
import { t, tr, trf } from "$lib/stores/i18n.svelte";

export const CAI_WALLET_ACCESS_REQUIRED_EVENT = "cai:wallet-access-required";

function dispatchWalletAccessRequiredEvent(message: string): void {
  if (!browser || typeof window === "undefined") {
    return;
  }
  window.dispatchEvent(
    new CustomEvent(CAI_WALLET_ACCESS_REQUIRED_EVENT, {
      detail: { message },
    }),
  );
}

// UUID generation fallback for browsers without crypto.randomUUID
function generateUUID(): string {
  if (
    typeof crypto !== "undefined" &&
    typeof crypto.randomUUID === "function"
  ) {
    return crypto.randomUUID();
  }
  // Fallback implementation
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

export interface NodeInfo {
  system_info?: {
    model_id?: string;
    chip?: string;
    memory?: number;
  };
  network_interfaces?: Array<{
    name?: string;
    addresses?: string[];
  }>;
  ip_to_interface?: Record<string, string>;
  macmon_info?: {
    memory?: {
      ram_usage: number;
      ram_total: number;
    };
    temp?: {
      gpu_temp_avg: number;
    };
    gpu_usage?: [number, number];
    sys_power?: number;
    load_label?: string;
  };
  last_macmon_update: number;
  friendly_name?: string;
  os_version?: string;
}

export interface TopologyEdge {
  source: string;
  target: string;
  sendBackIp?: string;
  sendBackInterface?: string;
  sourceRdmaIface?: string;
  sinkRdmaIface?: string;
}

export interface TopologyData {
  nodes: Record<string, NodeInfo>;
  edges: TopologyEdge[];
}

export interface Instance {
  shardAssignments?: {
    modelId?: string;
    runnerToShard?: Record<string, unknown>;
    nodeToRunner?: Record<string, string>;
  };
}

// Granular node state types from the new state structure
interface RawNodeIdentity {
  modelId?: string;
  chipId?: string;
  friendlyName?: string;
  osVersion?: string;
  osBuildVersion?: string;
}

interface RawMemoryUsage {
  ramTotal?: { inBytes: number };
  ramAvailable?: { inBytes: number };
  swapTotal?: { inBytes: number };
  swapAvailable?: { inBytes: number };
}

interface RawSystemPerformanceProfile {
  gpuUsage?: number;
  temp?: number;
  sysPower?: number;
  pcpuUsage?: number;
  ecpuUsage?: number;
}

interface RawNetworkInterfaceInfo {
  name?: string;
  ipAddress?: string;
  addresses?: Array<{ address?: string } | string>;
  ipv4?: string;
  ipv6?: string;
  ipAddresses?: string[];
  ips?: string[];
}

interface RawNodeNetworkInfo {
  interfaces?: RawNetworkInterfaceInfo[];
}

interface RawSocketConnection {
  sinkMultiaddr?: {
    address?: string;
    ip_address?: string;
    address_type?: string;
    port?: number;
  };
}

interface RawRDMAConnection {
  sourceRdmaIface?: string;
  sinkRdmaIface?: string;
}

type RawConnectionEdge = RawSocketConnection | RawRDMAConnection;

// New nested mapping format: { source: { sink: [edge1, edge2, ...] } }
type RawConnectionsMap = Record<string, Record<string, RawConnectionEdge[]>>;

interface RawTopology {
  nodes: string[];
  connections?: RawConnectionsMap;
}

export interface DownloadProgress {
  totalBytes: number;
  downloadedBytes: number;
  speed: number;
  etaMs: number;
  percentage: number;
  completedFiles: number;
  totalFiles: number;
  files: Array<{
    name: string;
    totalBytes: number;
    downloadedBytes: number;
    speed: number;
    etaMs: number;
    percentage: number;
  }>;
}

export interface ModelDownloadStatus {
  isDownloading: boolean;
  progress: DownloadProgress | null;
  nodeDetails: Array<{
    nodeId: string;
    nodeName: string;
    progress: DownloadProgress;
  }>;
}

// Placement preview from the API
export interface PlacementPreview {
  model_id: string;
  sharding: "Pipeline" | "Tensor";
  instance_meta: "MlxRing" | "MlxJaccl" | "LlamaCpp";
  instance: unknown | null;
  memory_delta_by_node: Record<string, number> | null;
  error: string | null;
}

export interface PlacementPreviewResponse {
  previews: PlacementPreview[];
}

export interface CaiChainTransactionSummary {
  txId?: string | null;
  txType?: string | null;
  address?: string | null;
  walletId?: string | null;
  deltaAtomic?: number | null;
  deltaCoins?: string | null;
  balanceAfterAtomic?: number | null;
  balanceAfterCoins?: string | null;
  jobId?: string | null;
  receiptId?: string | null;
  settlementId?: string | null;
  payoutId?: string | null;
  validatorId?: string | null;
  note?: string | null;
  createdAt?: string | null;
  blockHeight?: number | null;
  blockHash?: string | null;
  blockCreatedAt?: string | null;
  metadata?: Record<string, unknown>;
}

export interface CaiSettlementSummary {
  settlementId?: string | null;
  status?: string | null;
  fundingSource?: string | null;
  computeCostAtomic?: number | null;
  computeCostCoins?: string | null;
  txFeeAtomic?: number | null;
  settlementFeeAtomic?: number | null;
  aiDevelopmentFeeAtomic?: number | null;
  workerRewardAtomic?: number | null;
  sourceWalletDebitAtomic?: number | null;
  reserveDebitAtomic?: number | null;
  appliedAt?: string | null;
  balanceAudit?: Record<string, unknown>;
  chainRecorded?: boolean;
  chainTransactionCount?: number;
  chainTransactions?: Array<CaiChainTransactionSummary>;
}

export interface CaiWorkerRuntimeQueue {
  localNodeId?: string | null;
  ready?: boolean;
  reason?: string | null;
  recordCount?: number;
  receivedCount?: number;
  processingCount?: number;
  processedCount?: number;
  failedCount?: number;
  timedOutCount?: number;
  deliveredCount?: number;
  lastError?: string | null;
  currentBatch?: {
    sessionId?: string | null;
    batch?: {
      batchId?: string | null;
      status?: string | null;
      phase?: string | null;
    };
  } | null;
}

export interface CaiExecutionAttemptSummary {
  attempt?: number | null;
  status?: string | null;
  startedAt?: string | null;
  completedAt?: string | null;
  participantNodeIds?: string[];
  excludedNodeIds?: string[];
  instanceId?: string | null;
  retryScheduled?: boolean;
  errorType?: string | null;
  message?: string | null;
  phase?: string | null;
  phaseStartedAt?: string | null;
  phaseMessage?: string | null;
  timeoutSec?: number | null;
  attemptDurationMs?: number | null;
  readinessDurationMs?: number | null;
  responseDurationMs?: number | null;
}

export interface CaiExecutionAttemptStatus {
  attempt?: number | null;
  maxAttempts?: number | null;
  status?: string | null;
  message?: string | null;
  errorType?: string | null;
  phase?: string | null;
  phaseStartedAt?: string | null;
  phaseMessage?: string | null;
  retryScheduled?: boolean;
  participantNodeIds?: string[];
  excludedNodeIds?: string[];
  failedAttemptCount?: number | null;
  lastCompletedAt?: string | null;
  timeoutSec?: number | null;
  attemptDurationMs?: number | null;
  readinessDurationMs?: number | null;
  responseDurationMs?: number | null;
}

export interface CaiSummary {
  available: boolean;
  error?: string;
  currency?: {
    code?: string | null;
    name?: string | null;
    decimals?: number | null;
  };
  runtime?: {
    version?: string | null;
    versionLabel?: string | null;
    gitCommit?: string | null;
    gitBranch?: string | null;
    gitDirty?: boolean;
    buildId?: string | null;
    buildNumber?: number | null;
    buildNumberLabel?: string | null;
  };
  updates?: {
    autoUpdateEnabled?: boolean;
    channel?: string | null;
    provider?: string | null;
    repository?: string | null;
    targetBranch?: string | null;
    sourceUrl?: string | null;
    baseUrl?: string | null;
    checked?: boolean;
    checkedAt?: string | null;
    lastUpdatedAt?: string | null;
    updated?: boolean;
    updateAvailable?: boolean;
    remoteGitCommit?: string | null;
    remoteGitBranch?: string | null;
    remoteVersion?: string | null;
    remoteBuildId?: string | null;
    remoteInstallKind?: string | null;
    status?: string | null;
    phase?: string | null;
    progress?: number | null;
    downloadedBytes?: number | null;
    totalBytes?: number | null;
    downloadPercent?: number | null;
    downloadSpeedBytesPerSec?: number | null;
    message?: string | null;
    canApply?: boolean;
    applyReason?: string | null;
    canCancel?: boolean;
    cancelRequested?: boolean;
    restartScheduled?: boolean;
    restartRequired?: boolean;
    activity?: {
      idle?: boolean;
      reason?: string | null;
      idleRequiredSeconds?: number | null;
      idleForSeconds?: number | null;
      activeRequestCount?: number | null;
      lastUserActivityAt?: string | null;
      activeRequestUpdatedAt?: string | null;
      activityUpdatedAt?: string | null;
      activityStale?: boolean;
    };
    archivePath?: string | null;
    portableUpdatePlanPath?: string | null;
    portableUpdateScriptPath?: string | null;
    portableUpdateCancelPath?: string | null;
    sourceResolutionError?: {
      errorType?: string | null;
      message?: string | null;
    } | null;
    dashboardBuildStatus?: string | null;
    dashboardBuildMessage?: string | null;
  };
  wallet?: {
    has_active_wallet?: boolean;
    wallet_name?: string | null;
    address?: string | null;
    balance_coins?: string | null;
    balance_source?: string | null;
    local_cached_balance_coins?: string | null;
    unlocked?: boolean;
  };
  chainStatus?: {
    network?: string | null;
    block_count?: number | null;
    transaction_count?: number | null;
    tip_height?: number | null;
    tip_hash?: string | null;
    finalized_height?: number | null;
    last_sync_at?: string | null;
    valid?: boolean;
  };
  safety?: {
    mode?: string | null;
    network?: string | null;
    warningCount?: number;
    warnings?: Array<{
      code?: string | null;
      severity?: string | null;
      title?: string | null;
      message?: string | null;
    }>;
  };
  wallets?: Array<{
    wallet_id: string;
    selector: string;
    name: string;
    address: string;
    balance_coins: string;
    active: boolean;
    unlocked: boolean;
    seed_backed?: boolean;
  }>;
  validator?: {
    validator_enabled?: boolean;
    validator_state?: string;
    validator_address?: string | null;
    validator_unbonding_started_at?: string | null;
    validator_unbonding_available_at?: string | null;
    validator_jailed_at?: string | null;
    validator_unjail_available_at?: string | null;
    validator_jail_reason?: string | null;
    validator_can_enable?: boolean;
    validator_can_attest?: boolean;
    validator_attestation_note?: string;
    validator_status_note?: string;
    validator_network_ok?: boolean;
    validator_static_ip_confirmed?: boolean;
    validator_current_node_id?: string | null;
    validator_advertised_api_host?: string | null;
    validator_advertised_data_host?: string | null;
    validator_bond_coins?: string;
    validator_required_bond_coins?: string;
    active_wallet_spendable_coins?: string;
    validator_fee_pool_coins?: string;
    settlement_count?: number;
    attestation_count?: number;
    evidence_count?: number;
    evidence_case_count?: number;
    evidence_case_pending_quorum_count?: number;
    evidence_case_finalized_count?: number;
    evidence_case_applied_count?: number;
    penalty_case_count?: number;
    penalty_case_pending_count?: number;
    penalty_case_pending_attestation_count?: number;
    penalty_case_finalized_count?: number;
    penalty_case_applied_count?: number;
    latest_penalty_case_status?: string | null;
    latest_penalty_case_scope?: string | null;
    latest_penalty_case_validator_id?: string | null;
  };
  worker?: {
    worker_enabled?: boolean;
    worker_reward_address?: string | null;
    network_default_model_id?: string;
    network_default_execution_model_id?: string;
    allowed_model_ids?: string[];
    private_model_minimum_shards?: number;
    reserve_balance_coins?: string;
    local_worker_earnings_coins?: string;
    payout_records?: number;
    runtimeQueue?: CaiWorkerRuntimeQueue;
    runtime_queue?: CaiWorkerRuntimeQueue;
  };
  reward?: {
    payout_records?: number;
    settlement_records?: number;
    pending_count?: number;
    finalized_count?: number;
    applied_count?: number;
    unbound_count?: number;
    pending_coins?: string | null;
    finalized_coins?: string | null;
    applied_coins?: string | null;
    unbound_coins?: string | null;
    chain_recorded_count?: number;
    latest_status?: string | null;
    latest_settlement_id?: string | null;
    latest_payout_id?: string | null;
  };
  compute?: {
    job_intents?: number;
    execution_receipts?: number;
    network_default_model_id?: string;
  };
  history?: {
    journal?: Array<{
      entryId?: string | null;
      source?: string | null;
      eventType?: string | null;
      createdAt?: string | null;
      counterpartyAddress?: string | null;
      amountAtomic?: number | null;
      amountCoins?: string | null;
      txFeeAtomic?: number | null;
      txFeeCoins?: string | null;
      txId?: string | null;
      blockHeight?: number | null;
      blockHash?: string | null;
      balanceAfterAtomic?: number | null;
      balanceAfterCoins?: string | null;
      note?: string | null;
    }>;
    jobs?: Array<{
      jobId?: string | null;
      createdAt?: string | null;
      status?: string | null;
      modelId?: string | null;
      promptRedacted?: boolean;
      receiptId?: string | null;
      settlementId?: string | null;
      outputText?: string | null;
      lastError?: string | null;
      transportMode?: string | null;
      participantCount?: number | null;
      decentralizedExecution?: boolean | null;
      relayBottleneckRisk?: boolean | null;
      executionAttempts?: CaiExecutionAttemptSummary[];
      executionAttemptStatus?: CaiExecutionAttemptStatus | null;
      executionAttemptCount?: number | null;
      decentralizedChainAudit?: {
        requesterNodeId?: string | null;
        coordinatorNodeId?: string | null;
        participantCount?: number | null;
        participantNodeIds?: string[];
        executorCount?: number | null;
        executorNodeIds?: string[];
        transportMode?: string | null;
        decentralizedExecution?: boolean | null;
        route?: {
          directSocketLinkCount?: number | null;
          directBidirectionalLinkCount?: number | null;
          overlayLinkCount?: number | null;
          relayHopsUsed?: boolean | null;
          relayBottleneckRisk?: boolean | null;
          checkedDirectSocketLinkCount?: number | null;
          checkedRelayRouteCount?: number | null;
          relayRouteCandidateCount?: number | null;
          relayCoordinatorCandidateCount?: number | null;
          activeRelayTransitNodeIds?: string[];
        };
        proof?: {
          executed?: boolean | null;
          verified?: boolean | null;
          error?: string | null;
          stageCount?: number | null;
          processedStageCount?: number | null;
          finalOutputBatchCount?: number | null;
        };
        tokens?: {
          promptTokens?: number | null;
          completionTokens?: number | null;
          totalTokens?: number | null;
          proofPromptTokenCount?: number | null;
          proofCompletionTokenCount?: number | null;
        };
        bytes?: {
          payloadSizeBytes?: number | null;
          outputPayloadSizeBytes?: number | null;
        };
        reward?: {
          payoutCount?: number | null;
          workerPayoutTotalAtomic?: number | null;
          payoutNodes?: string[];
        };
      } | null;
      networkAudit?: Record<string, unknown> | null;
    }>;
    payouts?: Array<{
      createdAt?: string | null;
      nodeId?: string | null;
      status?: string | null;
      shareBps?: number | null;
      rewardAtomic?: number | null;
      rewardCoins?: string | null;
      settlementId?: string | null;
    }>;
    settlements?: Array<CaiSettlementSummary>;
  };
  latestJob?: {
    jobId?: string | null;
    status?: string | null;
    modelId?: string | null;
    promptRedacted?: boolean;
    executionAttempts?: CaiExecutionAttemptSummary[];
    executionAttemptStatus?: CaiExecutionAttemptStatus | null;
    executionAttemptCount?: number | null;
  } | null;
  latestReceipt?: {
    receiptId?: string | null;
    finishReason?: string | null;
    outputText?: string | null;
    payoutCount?: number | null;
    executionAttempts?: CaiExecutionAttemptSummary[];
    executionAttemptStatus?: CaiExecutionAttemptStatus | null;
    executionAttemptCount?: number | null;
  } | null;
  latestPayout?: {
    nodeId?: string | null;
    status?: string | null;
    shareBps?: number | null;
    rewardAtomic?: number | null;
  } | null;
  latestSettlement?: CaiSettlementSummary | null;
}

interface ImageApiResponse {
  created: number;
  data: Array<{ b64_json?: string; url?: string }>;
}

// Trace API response types
export interface TraceCategoryStats {
  totalUs: number;
  count: number;
  minUs: number;
  maxUs: number;
  avgUs: number;
}

export interface TraceRankStats {
  byCategory: Record<string, TraceCategoryStats>;
}

export interface TraceStatsResponse {
  taskId: string;
  totalWallTimeUs: number;
  byCategory: Record<string, TraceCategoryStats>;
  byRank: Record<number, TraceRankStats>;
}

export interface TraceListItem {
  taskId: string;
  createdAt: string;
  fileSize: number;
}

export interface TraceListResponse {
  traces: TraceListItem[];
}

interface RawStateResponse {
  networkSummary?: {
    knownNodes?: number;
    knownWorkers?: number;
    knownRelays?: number;
    knownConnections?: number;
    localOverlayPeers?: number;
    totalRamBytes?: number;
    totalAvailableRamBytes?: number;
    totalVramBytes?: number;
    totalCpuCores?: number;
    workerTotalRamBytes?: number;
    workerTotalAvailableRamBytes?: number;
    workerTotalVramBytes?: number;
    workerTotalCpuCores?: number;
    workerDirectSocketLinks?: number;
    workerOverlayLinks?: number;
    llamaCppLargestDirectWorkerCycle?: number;
    llamaCppDistributedReady?: boolean;
    llamaCppDistributedReason?: string | null;
    caiOwnedTransportReadiness?: {
      protocol?: string | null;
      status?: string | null;
      ready?: boolean;
      runtimeReady?: boolean;
      reason?: string | null;
      workerCount?: number;
      runtimeReadyWorkerCount?: number;
      implementedWorkerCount?: number;
      failedWorkerCount?: number;
      missingWorkerCount?: number;
      observedStatuses?: string[];
      runtimeReadyWorkerIds?: string[];
      implementedWorkerIds?: string[];
      failedWorkerIds?: string[];
      missingWorkerIds?: string[];
    };
  };
  topology?: RawTopology;
  instances?: Record<
    string,
    {
      MlxRingInstance?: Instance;
      MlxJacclInstance?: Instance;
    }
  >;
  runners?: Record<string, unknown>;
  downloads?: Record<string, unknown[]>;
  // New granular node state fields
  nodeIdentities?: Record<string, RawNodeIdentity>;
  nodeMemory?: Record<string, RawMemoryUsage>;
  nodeSystem?: Record<string, RawSystemPerformanceProfile>;
  nodeNetwork?: Record<string, RawNodeNetworkInfo>;
  // Thunderbolt identifiers per node
  nodeThunderbolt?: Record<
    string,
    {
      interfaces: Array<{
        rdmaInterface: string;
        domainUuid: string;
        linkSpeed: string;
      }>;
    }
  >;
  // RDMA ctl status per node
  nodeRdmaCtl?: Record<string, { enabled: boolean }>;
  // Thunderbolt bridge status per node
  nodeThunderboltBridge?: Record<
    string,
    { enabled: boolean; exists: boolean; serviceName?: string | null }
  >;
  // Thunderbolt bridge cycles (nodes with bridge enabled forming loops)
  thunderboltBridgeCycles?: string[][];
  // Disk usage per node
  nodeDisk?: Record<
    string,
    { total: { inBytes: number }; available: { inBytes: number } }
  >;
}

export interface MessageAttachment {
  type: "image" | "text" | "file" | "generated-image" | "pdf";
  name: string;
  content?: string;
  preview?: string;
  mimeType?: string;
  pageImages?: string[];
}

export interface TopLogprob {
  token: string;
  logprob: number;
  bytes: number[] | null;
}

export interface TokenData {
  token: string;
  logprob: number;
  probability: number;
  topLogprobs: TopLogprob[];
}

export interface CaiExecutionMeta {
  schemaVersion?: number;
  source?: string | null;
  jobId?: string | null;
  receiptId?: string | null;
  settlementId?: string | null;
  settlementStatus?: string | null;
  chainRecorded?: boolean | null;
  chainTransactionCount?: number | null;
  proofExecuted?: boolean | null;
  proofVerified?: boolean | null;
  proofError?: string | null;
  sessionId?: string | null;
  finalOutputBatchCount?: number | null;
  executionAttemptCount?: number | null;
  executionAttempts?: CaiExecutionAttemptSummary[];
  executionAttemptStatus?: CaiExecutionAttemptStatus | null;
  executorNodeIds?: string[];
  participantNodeIds?: string[];
  rewardPayoutSource?: string | null;
  rewardPayoutNodeIds?: string[];
  rewardSkippedNodeIdsWithoutShardReceipt?: string[];
  payoutCount?: number | null;
  payoutNodes?: string[];
  workerPayoutTotalAtomic?: number | null;
  payoutStatuses?: Array<string | null>;
}

export interface PrefillProgress {
  processed: number;
  total: number;
  /** Timestamp (performance.now()) when prefill started. */
  startedAt: number;
}

export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  thinking?: string;
  attachments?: MessageAttachment[];
  ttftMs?: number; // Time to first token in ms (for assistant messages)
  tps?: number; // Tokens per second (for assistant messages)
  requestType?: "chat" | "image-generation" | "image-editing";
  sourceImageDataUrl?: string; // For image editing regeneration
  tokens?: TokenData[];
  caiExecution?: CaiExecutionMeta;
}

export interface Conversation {
  id: string;
  name: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
  modelId: string | null;
  sharding: string | null;
  instanceType: string | null;
  enableThinking: boolean | null;
}

const STORAGE_KEY = "cai-conversations";
const IMAGE_PARAMS_STORAGE_KEY = "cai-image-generation-params";

// Image generation params interface matching backend API
export interface ImageGenerationParams {
  // Basic params
  size:
    | "auto"
    | "512x512"
    | "768x768"
    | "1024x1024"
    | "1024x768"
    | "768x1024"
    | "1024x1536"
    | "1536x1024";
  quality: "low" | "medium" | "high";
  outputFormat: "png" | "jpeg";
  numImages: number;
  // Streaming params
  stream: boolean;
  partialImages: number;
  // Advanced params
  seed: number | null;
  numInferenceSteps: number | null;
  guidance: number | null;
  negativePrompt: string | null;
  numSyncSteps: number | null;
  // Edit mode params
  inputFidelity: "low" | "high";
}

// Image being edited
export interface EditingImage {
  imageDataUrl: string;
  sourceMessage: Message;
}

const DEFAULT_IMAGE_PARAMS: ImageGenerationParams = {
  size: "auto",
  quality: "medium",
  outputFormat: "png",
  numImages: 1,
  stream: true,
  partialImages: 3,
  seed: null,
  numInferenceSteps: null,
  guidance: null,
  negativePrompt: null,
  numSyncSteps: null,
  inputFidelity: "low",
};

interface GranularNodeState {
  nodeIdentities?: Record<string, RawNodeIdentity>;
  nodeMemory?: Record<string, RawMemoryUsage>;
  nodeSystem?: Record<string, RawSystemPerformanceProfile>;
  nodeNetwork?: Record<string, RawNodeNetworkInfo>;
}

function transformNetworkInterface(iface: RawNetworkInterfaceInfo): {
  name?: string;
  addresses: string[];
} {
  const addresses: string[] = [];
  if (iface.ipAddress && typeof iface.ipAddress === "string") {
    addresses.push(iface.ipAddress);
  }
  if (Array.isArray(iface.addresses)) {
    for (const addr of iface.addresses) {
      if (typeof addr === "string") addresses.push(addr);
      else if (addr && typeof addr === "object" && addr.address)
        addresses.push(addr.address);
    }
  }
  if (Array.isArray(iface.ipAddresses)) {
    addresses.push(
      ...iface.ipAddresses.filter((a): a is string => typeof a === "string"),
    );
  }
  if (Array.isArray(iface.ips)) {
    addresses.push(
      ...iface.ips.filter((a): a is string => typeof a === "string"),
    );
  }
  if (iface.ipv4 && typeof iface.ipv4 === "string") addresses.push(iface.ipv4);
  if (iface.ipv6 && typeof iface.ipv6 === "string") addresses.push(iface.ipv6);

  return {
    name: iface.name,
    addresses: Array.from(new Set(addresses)),
  };
}

function transformTopology(
  raw: RawTopology,
  granularState: GranularNodeState,
): TopologyData {
  const nodes: Record<string, NodeInfo> = {};
  const edges: TopologyEdge[] = [];

  for (const nodeId of raw.nodes || []) {
    if (!nodeId) continue;

    // Get data from granular state mappings
    const identity = granularState.nodeIdentities?.[nodeId];
    const memory = granularState.nodeMemory?.[nodeId];
    const system = granularState.nodeSystem?.[nodeId];
    const network = granularState.nodeNetwork?.[nodeId];

    const ramTotal = memory?.ramTotal?.inBytes ?? 0;
    const ramAvailable = memory?.ramAvailable?.inBytes ?? 0;
    const ramUsage = Math.max(ramTotal - ramAvailable, 0);
    const loadValue = system?.gpuUsage ?? system?.pcpuUsage;
    const loadLabel =
      system?.gpuUsage !== undefined
        ? "GPU"
        : system?.pcpuUsage !== undefined
          ? "CPU"
          : undefined;

    const rawInterfaces = network?.interfaces || [];
    const networkInterfaces = rawInterfaces.map(transformNetworkInterface);

    const ipToInterface: Record<string, string> = {};
    for (const iface of networkInterfaces) {
      for (const addr of iface.addresses || []) {
        ipToInterface[addr] = iface.name ?? "";
      }
    }

    nodes[nodeId] = {
      system_info: {
        model_id: identity?.modelId ?? "Unknown",
        chip: identity?.chipId,
        memory: ramTotal,
      },
      network_interfaces: networkInterfaces,
      ip_to_interface: ipToInterface,
      macmon_info: {
        memory: {
          ram_usage: ramUsage,
          ram_total: ramTotal,
        },
        temp:
          system?.temp !== undefined
            ? { gpu_temp_avg: system.temp }
            : undefined,
        gpu_usage: loadValue !== undefined ? [0, loadValue] : undefined,
        sys_power: system?.sysPower,
        load_label: loadLabel,
      },
      last_macmon_update: Date.now() / 1000,
      friendly_name: identity?.friendlyName,
      os_version: identity?.osVersion,
    };
  }

  // Handle connections - nested mapping format { source: { sink: [edges] } }
  const connections = raw.connections;
  if (connections && typeof connections === "object") {
    for (const [source, sinks] of Object.entries(connections)) {
      if (!sinks || typeof sinks !== "object") continue;
      for (const [sink, edgeList] of Object.entries(sinks)) {
        if (!Array.isArray(edgeList)) continue;
        for (const edge of edgeList) {
          let sendBackIp: string | undefined;
          let sourceRdmaIface: string | undefined;
          let sinkRdmaIface: string | undefined;
          if (edge && typeof edge === "object" && "sinkMultiaddr" in edge) {
            const multiaddr = edge.sinkMultiaddr;
            if (multiaddr) {
              sendBackIp =
                multiaddr.ip_address ||
                extractIpFromMultiaddr(multiaddr.address);
            }
          } else if (
            edge &&
            typeof edge === "object" &&
            "sourceRdmaIface" in edge
          ) {
            sourceRdmaIface = edge.sourceRdmaIface;
            sinkRdmaIface = edge.sinkRdmaIface;
          }

          if (nodes[source] && nodes[sink] && source !== sink) {
            edges.push({
              source,
              target: sink,
              sendBackIp,
              sourceRdmaIface,
              sinkRdmaIface,
            });
          }
        }
      }
    }
  }

  return { nodes, edges };
}

function extractIpFromMultiaddr(ma?: string): string | undefined {
  if (!ma) return undefined;
  const parts = ma.split("/");
  const ip4Idx = parts.indexOf("ip4");
  const ip6Idx = parts.indexOf("ip6");
  const idx = ip4Idx >= 0 ? ip4Idx : ip6Idx;
  if (idx >= 0 && parts.length > idx + 1) {
    return parts[idx + 1];
  }
  return undefined;
}

class AppStore {
  // Conversation state
  conversations = $state<Conversation[]>([]);
  activeConversationId = $state<string | null>(null);

  // Chat state
  hasStartedChat = $state(true);
  messages = $state<Message[]>([]);
  currentResponse = $state("");
  isLoading = $state(false);

  // Performance metrics
  ttftMs = $state<number | null>(null); // Time to first token in ms
  tps = $state<number | null>(null); // Tokens per second
  totalTokens = $state<number>(0); // Total tokens in current response
  prefillProgress = $state<PrefillProgress | null>(null);

  // Abort controller for stopping generation
  private currentAbortController: AbortController | null = null;

  // Topology state
  topologyData = $state<TopologyData | null>(null);
  instances = $state<Record<string, unknown>>({});
  runners = $state<Record<string, unknown>>({});
  downloads = $state<Record<string, unknown[]>>({});
  nodeDisk = $state<
    Record<
      string,
      { total: { inBytes: number }; available: { inBytes: number } }
    >
  >({});
  placementPreviews = $state<PlacementPreview[]>([]);
  selectedPreviewModelId = $state<string | null>(null);
  isLoadingPreviews = $state(false);
  lastUpdate = $state<number | null>(null);
  nodeIdentities = $state<Record<string, RawNodeIdentity>>({});
  thunderboltBridgeCycles = $state<string[][]>([]);
  nodeThunderbolt = $state<
    Record<
      string,
      {
        interfaces: Array<{
          rdmaInterface: string;
          domainUuid: string;
          linkSpeed: string;
        }>;
      }
    >
  >({});
  nodeRdmaCtl = $state<Record<string, { enabled: boolean }>>({});
  nodeThunderboltBridge = $state<
    Record<
      string,
      { enabled: boolean; exists: boolean; serviceName?: string | null }
    >
  >({});
  caiSummary = $state<CaiSummary | null>(null);
  networkSummary = $state<RawStateResponse["networkSummary"] | null>(null);

  // UI state
  isTopologyMinimized = $state(true);
  isSidebarOpen = $state(false); // Hidden by default, shown when in chat mode
  debugMode = $state(false);
  chatSidebarVisible = $state(true); // Shown by default
  mobileChatSidebarOpen = $state(false); // Mobile drawer state
  mobileRightSidebarOpen = $state(false); // Mobile right drawer state

  // Image generation params
  imageGenerationParams = $state<ImageGenerationParams>({
    ...DEFAULT_IMAGE_PARAMS,
  });

  // Image editing state
  editingImage = $state<EditingImage | null>(null);

  /** True when the backend is reachable. */
  isConnected = $state<boolean>(true);
  /** Number of consecutive fetch failures. */
  private consecutiveFailures = 0;
  private static readonly CONNECTION_LOST_THRESHOLD = 3;

  private fetchInterval: ReturnType<typeof setInterval> | null = null;
  private previewsInterval: ReturnType<typeof setInterval> | null = null;
  private updateActivityInterval: ReturnType<typeof setInterval> | null = null;
  private lastConversationPersistTs = 0;
  private previousNodeIds: Set<string> = new Set();
  private lastCaiFetchTs = 0;
  private lastUserActivityAt = Date.now();
  private lastUpdateActivityReportTs = 0;
  private readonly updateActivityHandler = () => this.noteUpdateUserActivity();

  constructor() {
    if (browser) {
      this.startPolling();
      this.loadConversationsFromStorage();
      this.loadDebugModeFromStorage();
      this.loadChatSidebarVisibleFromStorage();
      this.loadImageGenerationParamsFromStorage();
      this.startUpdateActivityReporting();
    }
  }

  /**
   * Load conversations from localStorage
   */
  private loadConversationsFromStorage() {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as Array<Partial<Conversation>>;
        this.conversations = parsed.map((conversation) => ({
          id: conversation.id ?? generateUUID(),
          name: conversation.name ?? "Chat",
          messages: conversation.messages ?? [],
          createdAt: conversation.createdAt ?? Date.now(),
          updatedAt: conversation.updatedAt ?? Date.now(),
          modelId: conversation.modelId ?? null,
          sharding: conversation.sharding ?? null,
          instanceType: conversation.instanceType ?? null,
          enableThinking: conversation.enableThinking ?? null,
        }));
      }
    } catch (error) {
      console.error("Failed to load conversations:", error);
    }
  }

  /**
   * Save conversations to localStorage
   */
  private saveConversationsToStorage() {
    try {
      // Strip tokens from messages before saving to avoid bloating localStorage
      const stripped = this.conversations.map((conv) => ({
        ...conv,
        messages: conv.messages.map((msg) => {
          if (msg.tokens) {
            const { tokens: _, ...rest } = msg;
            return rest;
          }
          return msg;
        }),
      }));
      localStorage.setItem(STORAGE_KEY, JSON.stringify(stripped));
    } catch (error) {
      console.error("Failed to save conversations:", error);
    }
  }

  private loadDebugModeFromStorage() {
    try {
      const stored = localStorage.getItem("cai-debug-mode");
      if (stored !== null) {
        this.debugMode = stored === "true";
      }
    } catch (error) {
      console.error("Failed to load debug mode:", error);
    }
  }

  private saveDebugModeToStorage() {
    try {
      localStorage.setItem("cai-debug-mode", this.debugMode ? "true" : "false");
    } catch (error) {
      console.error("Failed to save debug mode:", error);
    }
  }

  private loadChatSidebarVisibleFromStorage() {
    try {
      const stored = localStorage.getItem("cai-chat-sidebar-visible");
      if (stored !== null) {
        this.chatSidebarVisible = stored === "true";
      }
    } catch (error) {
      console.error("Failed to load chat sidebar visibility:", error);
    }
  }

  private saveChatSidebarVisibleToStorage() {
    try {
      localStorage.setItem(
        "cai-chat-sidebar-visible",
        this.chatSidebarVisible ? "true" : "false",
      );
    } catch (error) {
      console.error("Failed to save chat sidebar visibility:", error);
    }
  }

  private loadImageGenerationParamsFromStorage() {
    try {
      const stored = localStorage.getItem(IMAGE_PARAMS_STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as Partial<ImageGenerationParams>;
        this.imageGenerationParams = {
          ...DEFAULT_IMAGE_PARAMS,
          ...parsed,
        };
      }
    } catch (error) {
      console.error("Failed to load image generation params:", error);
    }
  }

  private saveImageGenerationParamsToStorage() {
    try {
      localStorage.setItem(
        IMAGE_PARAMS_STORAGE_KEY,
        JSON.stringify(this.imageGenerationParams),
      );
    } catch (error) {
      console.error("Failed to save image generation params:", error);
    }
  }

  getImageGenerationParams(): ImageGenerationParams {
    return this.imageGenerationParams;
  }

  setImageGenerationParams(params: Partial<ImageGenerationParams>) {
    this.imageGenerationParams = {
      ...this.imageGenerationParams,
      ...params,
    };
    this.saveImageGenerationParamsToStorage();
  }

  resetImageGenerationParams() {
    this.imageGenerationParams = { ...DEFAULT_IMAGE_PARAMS };
    this.saveImageGenerationParamsToStorage();
  }

  setEditingImage(imageDataUrl: string, sourceMessage: Message) {
    this.editingImage = { imageDataUrl, sourceMessage };
  }

  clearEditingImage() {
    this.editingImage = null;
  }

  /**
   * Create a new conversation
   */
  createConversation(name?: string): string {
    const id = generateUUID();
    const now = Date.now();

    // Try to derive model and strategy immediately from selected model or running instances
    let derivedModelId = this.selectedChatModel || null;
    let derivedInstanceType: string | null = null;
    let derivedSharding: string | null = null;

    // If no selected model, fall back to the first running instance
    if (!derivedModelId) {
      const firstInstance = Object.values(this.instances)[0];
      if (firstInstance) {
        const candidateModel = this.extractInstanceModelId(firstInstance);
        derivedModelId = candidateModel ?? null;
        const details = this.describeInstance(firstInstance);
        derivedInstanceType = details.instanceType;
        derivedSharding = details.sharding;
      }
    } else {
      // If selected model is set, attempt to get its details from instances
      for (const [, instanceWrapper] of Object.entries(this.instances)) {
        const candidateModelId = this.extractInstanceModelId(instanceWrapper);
        if (candidateModelId === derivedModelId) {
          const details = this.describeInstance(instanceWrapper);
          derivedInstanceType = details.instanceType;
          derivedSharding = details.sharding;
          break;
        }
      }
    }

    const conversation: Conversation = {
      id,
      name:
        name ||
        `Chat ${new Date(now).toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}`,
      messages: [],
      createdAt: now,
      updatedAt: now,
      modelId: derivedModelId,
      sharding: derivedSharding,
      instanceType: derivedInstanceType,
      enableThinking: null,
    };

    this.conversations.unshift(conversation);
    this.activeConversationId = id;
    this.messages = [];
    this.hasStartedChat = true;
    this.isTopologyMinimized = true;
    this.isSidebarOpen = true; // Auto-open sidebar when chatting

    this.saveConversationsToStorage();
    return id;
  }

  /**
   * Load a conversation by ID
   */
  loadConversation(id: string): boolean {
    const conversation = this.conversations.find((c) => c.id === id);
    if (!conversation) return false;

    this.activeConversationId = id;
    this.messages = [...conversation.messages];
    this.hasStartedChat = true;
    this.isTopologyMinimized = true;
    this.isSidebarOpen = true; // Auto-open sidebar when chatting
    this.thinkingEnabled = conversation.enableThinking ?? true;
    this.refreshConversationModelFromInstances();

    // Sync global selection to the loaded conversation's model so reactive
    // effects in +page.svelte can determine the correct chat launch state.
    this.selectedChatModel = conversation.modelId || "";

    return true;
  }

  /**
   * Delete a conversation by ID
   */
  deleteConversation(id: string) {
    this.conversations = this.conversations.filter((c) => c.id !== id);

    if (this.activeConversationId === id) {
      this.activeConversationId = null;
      this.messages = [];
      this.hasStartedChat = true;
      this.isTopologyMinimized = true;
    }

    this.saveConversationsToStorage();
  }

  /**
   * Delete all conversations
   */
  deleteAllConversations() {
    this.conversations = [];
    this.activeConversationId = null;
    this.messages = [];
    this.hasStartedChat = true;
    this.isTopologyMinimized = true;
    this.saveConversationsToStorage();
  }

  /**
   * Rename a conversation
   */
  renameConversation(id: string, newName: string) {
    const conversation = this.conversations.find((c) => c.id === id);
    if (conversation) {
      conversation.name = newName;
      conversation.updatedAt = Date.now();
      this.saveConversationsToStorage();
    }
  }

  private getTaggedValue(obj: unknown): [string | null, unknown] {
    if (!obj || typeof obj !== "object") return [null, null];
    const keys = Object.keys(obj as Record<string, unknown>);
    if (keys.length === 1) {
      return [keys[0], (obj as Record<string, unknown>)[keys[0]]];
    }
    return [null, null];
  }

  private extractInstanceModelId(instanceWrapped: unknown): string | null {
    const [, instance] = this.getTaggedValue(instanceWrapped);
    if (!instance || typeof instance !== "object") return null;
    const inst = instance as { shardAssignments?: { modelId?: string } };
    return inst.shardAssignments?.modelId ?? null;
  }

  private extractInstanceInferenceBackend(instanceWrapped: unknown): string | null {
    const [, instance] = this.getTaggedValue(instanceWrapped);
    if (!instance || typeof instance !== "object") return null;
    const inst = instance as {
      shardAssignments?: { runnerToShard?: Record<string, unknown> };
    };
    const runnerToShard = inst.shardAssignments?.runnerToShard || {};
    const firstShardWrapped = Object.values(runnerToShard)[0];
    if (!firstShardWrapped) return null;

    const [, shard] = this.getTaggedValue(firstShardWrapped);
    if (!shard || typeof shard !== "object") return null;
    const shardData = shard as Record<string, unknown>;
    const modelMeta = shardData.model_card ?? shardData.modelCard;
    if (!modelMeta || typeof modelMeta !== "object") return null;

    const meta = modelMeta as Record<string, unknown>;
    const backend = meta.inference_backend ?? meta.inferenceBackend;
    return typeof backend === "string" ? backend : null;
  }

  private hasRunningInstanceForModel(modelId: string | null | undefined): boolean {
    if (!modelId) return false;
    return Object.values(this.instances).some(
      (instanceWrapper) => this.extractInstanceModelId(instanceWrapper) === modelId,
    );
  }

  private describeInstance(instanceWrapped: unknown): {
    sharding: string | null;
    instanceType: string | null;
  } {
    const [instanceTag, instance] = this.getTaggedValue(instanceWrapped);
    if (!instance || typeof instance !== "object") {
      return { sharding: null, instanceType: null };
    }

    let instanceType: string | null = null;
    const modelId = this.extractInstanceModelId(instanceWrapped);
    const inferenceBackend =
      this.extractInstanceInferenceBackend(instanceWrapped);
    if (instanceTag === "MlxRingInstance") {
      instanceType =
        inferenceBackend === "llama_cpp" ||
        modelId?.toLowerCase().includes("gguf")
          ? "GGUF / llama.cpp"
          : "MLX Ring";
    } else if (instanceTag === "MlxJacclInstance") {
      instanceType = "MLX RDMA";
    }

    let sharding: string | null = null;
    const inst = instance as {
      shardAssignments?: { runnerToShard?: Record<string, unknown> };
    };
    const runnerToShard = inst.shardAssignments?.runnerToShard || {};
    const firstShardWrapped = Object.values(runnerToShard)[0];
    if (firstShardWrapped) {
      const [shardTag] = this.getTaggedValue(firstShardWrapped);
      if (shardTag === "PipelineShardMetadata") sharding = "Pipeline";
      else if (shardTag === "TensorShardMetadata") sharding = "Tensor";
      else if (shardTag === "PrefillDecodeShardMetadata")
        sharding = "Prefill/Decode";
    }

    return { sharding, instanceType };
  }

  private buildConversationModelInfo(modelId: string): {
    modelId: string;
    sharding: string | null;
    instanceType: string | null;
  } {
    let sharding: string | null = null;
    let instanceType: string | null = null;

    for (const [, instanceWrapper] of Object.entries(this.instances)) {
      const candidateModelId = this.extractInstanceModelId(instanceWrapper);
      if (candidateModelId === modelId) {
        const details = this.describeInstance(instanceWrapper);
        sharding = details.sharding;
        instanceType = details.instanceType;
        break;
      }
    }

    return { modelId, sharding, instanceType };
  }

  private applyConversationModelInfo(info: {
    modelId: string;
    sharding: string | null;
    instanceType: string | null;
  }) {
    if (!this.activeConversationId) return;
    const conversation = this.conversations.find(
      (c) => c.id === this.activeConversationId,
    );
    if (!conversation) return;

    // Keep the first known modelId stable; only backfill if missing
    if (!conversation.modelId) {
      conversation.modelId = info.modelId;
    }
    conversation.sharding = info.sharding;
    conversation.instanceType = info.instanceType;
    this.saveConversationsToStorage();
  }

  private getModelTail(modelId: string): string {
    const parts = modelId.split("/");
    return (parts[parts.length - 1] || modelId).toLowerCase();
  }

  private isBetterModelId(
    currentId: string | null,
    candidateId: string | null,
  ): boolean {
    if (!candidateId) return false;
    if (!currentId) return true;
    const currentTail = this.getModelTail(currentId);
    const candidateTail = this.getModelTail(candidateId);
    return (
      candidateTail.length > currentTail.length &&
      candidateTail.startsWith(currentTail)
    );
  }

  private refreshConversationModelFromInstances() {
    if (!this.activeConversationId) return;
    const conversation = this.conversations.find(
      (c) => c.id === this.activeConversationId,
    );
    if (!conversation) return;

    // Prefer stored model; do not replace it once set. Only backfill when missing.
    let modelId = conversation.modelId;

    // If missing, try the selected model
    if (!modelId && this.selectedChatModel) {
      modelId = this.selectedChatModel;
    }

    // If still missing, fall back to first instance model
    if (!modelId) {
      const firstInstance = Object.values(this.instances)[0];
      if (firstInstance) {
        modelId = this.extractInstanceModelId(firstInstance);
      }
    }

    if (!modelId) return;

    // If a more specific instance modelId is available (e.g., adds "-4bit"), prefer it
    let preferredModelId = modelId;
    for (const [, instanceWrapper] of Object.entries(this.instances)) {
      const candidate = this.extractInstanceModelId(instanceWrapper);
      if (!candidate) continue;
      if (candidate === preferredModelId) {
        break;
      }
      if (this.isBetterModelId(preferredModelId, candidate)) {
        preferredModelId = candidate;
      }
    }

    if (this.isBetterModelId(conversation.modelId, preferredModelId)) {
      conversation.modelId = preferredModelId;
    }

    const info = this.buildConversationModelInfo(preferredModelId);
    const hasNewInfo = Boolean(
      info.sharding || info.instanceType || !conversation.modelId,
    );
    if (hasNewInfo) {
      this.applyConversationModelInfo(info);
    }
  }

  getDebugMode(): boolean {
    return this.debugMode;
  }

  /**
   * Update the active conversation with current messages
   */
  private updateActiveConversation() {
    if (!this.activeConversationId) return;

    const conversation = this.conversations.find(
      (c) => c.id === this.activeConversationId,
    );
    if (conversation) {
      conversation.messages = [...this.messages];
      conversation.updatedAt = Date.now();

      // Auto-generate name from first user message if still has default name
      if (conversation.name.startsWith("Chat ")) {
        const firstUserMsg = conversation.messages.find(
          (m) => m.role === "user" && m.content.trim(),
        );
        if (firstUserMsg) {
          // Clean up the content - remove file context markers and whitespace
          let content = firstUserMsg.content
            .replace(/\[File:.*?\][\s\S]*?```[\s\S]*?```/g, "") // Remove file attachments
            .trim();

          if (content) {
            const preview = content.slice(0, 50);
            conversation.name =
              preview.length < content.length ? preview + "..." : preview;
          }
        }
      }

      this.saveConversationsToStorage();
    }
  }

  private persistActiveConversation(throttleMs = 400) {
    const now = Date.now();
    if (now - this.lastConversationPersistTs < throttleMs) return;
    this.lastConversationPersistTs = now;
    this.updateActiveConversation();
  }

  /**
   * Update a message in a specific conversation by ID.
   * Returns false if conversation or message not found.
   */
  private updateConversationMessage(
    conversationId: string,
    messageId: string,
    updater: (message: Message) => void,
  ): boolean {
    const conversation = this.conversations.find(
      (c) => c.id === conversationId,
    );
    if (!conversation) return false;

    const message = conversation.messages.find((m) => m.id === messageId);
    if (!message) return false;

    updater(message);
    return true;
  }

  /**
   * Sync this.messages from the target conversation if it matches the active conversation.
   */
  private syncActiveMessagesIfNeeded(conversationId: string): void {
    if (this.activeConversationId === conversationId) {
      const conversation = this.conversations.find(
        (c) => c.id === conversationId,
      );
      if (conversation) {
        this.messages = [...conversation.messages];
      }
    }
  }

  /**
   * Check if a conversation still exists.
   */
  private conversationExists(conversationId: string): boolean {
    return this.conversations.some((c) => c.id === conversationId);
  }

  /**
   * Persist a specific conversation to storage.
   */
  private persistConversation(conversationId: string, throttleMs = 400): void {
    const now = Date.now();
    if (now - this.lastConversationPersistTs < throttleMs) return;
    this.lastConversationPersistTs = now;

    const conversation = this.conversations.find(
      (c) => c.id === conversationId,
    );
    if (conversation) {
      conversation.updatedAt = Date.now();

      // Auto-generate name from first user message if still has default name
      if (conversation.name.startsWith("Chat ")) {
        const firstUserMsg = conversation.messages.find(
          (m) => m.role === "user" && m.content.trim(),
        );
        if (firstUserMsg) {
          let content = firstUserMsg.content
            .replace(/\[File:.*?\][\s\S]*?```[\s\S]*?```/g, "")
            .trim();

          if (content) {
            const preview = content.slice(0, 50);
            conversation.name =
              preview.length < content.length ? preview + "..." : preview;
          }
        }
      }

      this.saveConversationsToStorage();
    }
  }

  /**
   * Add a message directly to a specific conversation.
   * Returns the message if added, null if conversation not found.
   */
  private addMessageToConversation(
    conversationId: string,
    role: "user" | "assistant",
    content: string,
  ): Message | null {
    const conversation = this.conversations.find(
      (c) => c.id === conversationId,
    );
    if (!conversation) return null;

    const message: Message = {
      id: generateUUID(),
      role,
      content,
      timestamp: Date.now(),
    };
    conversation.messages.push(message);
    return message;
  }

  /**
   * Toggle sidebar visibility
   */
  toggleSidebar() {
    this.isSidebarOpen = !this.isSidebarOpen;
  }

  setDebugMode(enabled: boolean) {
    this.debugMode = enabled;
    this.saveDebugModeToStorage();
  }

  toggleDebugMode() {
    this.debugMode = !this.debugMode;
    this.saveDebugModeToStorage();
  }

  getChatSidebarVisible(): boolean {
    return this.chatSidebarVisible;
  }

  setChatSidebarVisible(visible: boolean) {
    this.chatSidebarVisible = visible;
    this.saveChatSidebarVisibleToStorage();
  }

  toggleChatSidebarVisible() {
    this.chatSidebarVisible = !this.chatSidebarVisible;
    this.saveChatSidebarVisibleToStorage();
  }

  getMobileChatSidebarOpen(): boolean {
    return this.mobileChatSidebarOpen;
  }

  setMobileChatSidebarOpen(open: boolean) {
    this.mobileChatSidebarOpen = open;
  }

  toggleMobileChatSidebar() {
    this.mobileChatSidebarOpen = !this.mobileChatSidebarOpen;
  }

  getMobileRightSidebarOpen(): boolean {
    return this.mobileRightSidebarOpen;
  }

  setMobileRightSidebarOpen(open: boolean) {
    this.mobileRightSidebarOpen = open;
  }

  toggleMobileRightSidebar() {
    this.mobileRightSidebarOpen = !this.mobileRightSidebarOpen;
  }

  startPolling() {
    this.fetchState();
    this.fetchInterval = setInterval(() => this.fetchState(), 1000);
  }

  stopPolling() {
    if (this.fetchInterval) {
      clearInterval(this.fetchInterval);
      this.fetchInterval = null;
    }
    this.stopPreviewsPolling();
    this.stopUpdateActivityReporting();
  }

  noteUpdateUserActivity(): void {
    this.lastUserActivityAt = Date.now();
    if (Date.now() - this.lastUpdateActivityReportTs > 2000) {
      void this.reportUpdateActivity("user");
    }
  }

  private startUpdateActivityReporting(): void {
    if (!browser || this.updateActivityInterval) return;
    for (const eventName of ["pointerdown", "keydown", "wheel", "touchstart", "input"]) {
      window.addEventListener(eventName, this.updateActivityHandler, {
        passive: true,
        capture: true,
      });
    }
    void this.reportUpdateActivity("startup");
    this.updateActivityInterval = setInterval(() => {
      void this.reportUpdateActivity("heartbeat");
    }, 5000);
  }

  private stopUpdateActivityReporting(): void {
    if (!browser) return;
    if (this.updateActivityInterval) {
      clearInterval(this.updateActivityInterval);
      this.updateActivityInterval = null;
    }
    for (const eventName of ["pointerdown", "keydown", "wheel", "touchstart", "input"]) {
      window.removeEventListener(eventName, this.updateActivityHandler, {
        capture: true,
      });
    }
  }

  private async reportUpdateActivity(reason: string): Promise<void> {
    if (!browser) return;
    this.lastUpdateActivityReportTs = Date.now();
    try {
      await fetch("/v1/cai/update/activity", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          source: "dashboard",
          userActive: Date.now() - this.lastUserActivityAt < 45000,
          lastUserActivityAt: new Date(this.lastUserActivityAt).toISOString(),
          activeRequestCount: this.isLoading ? 1 : 0,
          metadata: { reason },
        }),
      });
    } catch {
      // Activity reporting is best-effort; update checks remain safe without it.
    }
  }

  async fetchState() {
    try {
      const response = await fetch("/dashboard/state");
      if (!response.ok) {
        throw new Error(`Failed to fetch state: ${response.status}`);
      }
      const data: RawStateResponse = await response.json();

      if (data.topology) {
        this.topologyData = transformTopology(data.topology, {
          nodeIdentities: data.nodeIdentities,
          nodeMemory: data.nodeMemory,
          nodeSystem: data.nodeSystem,
          nodeNetwork: data.nodeNetwork,
        });
        // Handle topology changes for preview filter
        this.handleTopologyChange();
      }
      if (data.instances) {
        this.instances = data.instances;
        this.refreshConversationModelFromInstances();
      }
      if (data.runners) {
        this.runners = data.runners;
      }
      if (data.downloads) {
        this.downloads = data.downloads;
      }
      if (data.nodeDisk) {
        this.nodeDisk = data.nodeDisk;
      }
      this.networkSummary = data.networkSummary ?? null;
      // Node identities (for OS version mismatch detection)
      this.nodeIdentities = data.nodeIdentities ?? {};
      // Thunderbolt identifiers per node
      this.nodeThunderbolt = data.nodeThunderbolt ?? {};
      // RDMA ctl status per node
      this.nodeRdmaCtl = data.nodeRdmaCtl ?? {};
      // Thunderbolt bridge cycles
      this.thunderboltBridgeCycles = data.thunderboltBridgeCycles ?? [];
      // Thunderbolt bridge status per node
      this.nodeThunderboltBridge = data.nodeThunderboltBridge ?? {};
      this.lastUpdate = Date.now();
      if (
        this.lastCaiFetchTs === 0 ||
        Date.now() - this.lastCaiFetchTs >= 5000
      ) {
        this.lastCaiFetchTs = Date.now();
        void this.fetchCaiSummary();
      }
      // Connection recovered
      if (!this.isConnected) {
        this.isConnected = true;
      }
      this.consecutiveFailures = 0;
    } catch (error) {
      this.consecutiveFailures++;
      if (
        this.consecutiveFailures >= AppStore.CONNECTION_LOST_THRESHOLD &&
        this.isConnected
      ) {
        this.isConnected = false;
      }
      console.error("Error fetching state:", error);
    }
  }

  async fetchCaiSummary() {
    try {
      const response = await fetch("/v1/cai/summary");
      if (!response.ok) {
        throw new Error(`Failed to fetch cai summary: ${response.status}`);
      }
      const data: CaiSummary = await response.json();
      this.caiSummary = data;
    } catch (error) {
      console.error("Error fetching cai summary:", error);
      this.caiSummary = {
        available: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  async cancelCaiUpdate() {
    const response = await fetch("/v1/cai/update/cancel", {
      method: "POST",
    });
    if (!response.ok) {
      throw new Error(`Failed to cancel CAI update: ${response.status}`);
    }
    const result = await response.json();
    await this.fetchCaiSummary();
    return result;
  }

  async checkCaiUpdate() {
    const response = await fetch("/v1/cai/update/check", {
      method: "POST",
    });
    if (!response.ok) {
      throw new Error(`Failed to check CAI update: ${response.status}`);
    }
    const result = await response.json();
    await this.fetchCaiSummary();
    return result;
  }

  async applyCaiUpdate() {
    const response = await fetch("/v1/cai/update/apply", {
      method: "POST",
    });
    if (!response.ok) {
      throw new Error(`Failed to apply CAI update: ${response.status}`);
    }
    const result = await response.json();
    await this.fetchCaiSummary();
    return result;
  }

  async fetchPlacementPreviews(modelId: string, showLoading = true) {
    if (!modelId) return;

    if (showLoading) {
      this.isLoadingPreviews = true;
    }
    this.selectedPreviewModelId = modelId;

    try {
      let url = `/instance/previews?model_id=${encodeURIComponent(modelId)}`;
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(
          `Failed to fetch placement previews: ${response.status}`,
        );
      }
      const data: PlacementPreviewResponse = await response.json();
      this.placementPreviews = data.previews;
    } catch (error) {
      console.error("Error fetching placement previews:", error);
      this.placementPreviews = [];
    } finally {
      if (showLoading) {
        this.isLoadingPreviews = false;
      }
    }
  }

  startPreviewsPolling(modelId: string) {
    // Stop any existing preview polling
    this.stopPreviewsPolling();

    // Fetch immediately
    this.fetchPlacementPreviews(modelId);

    // Then poll every 15 seconds (don't show loading spinner for subsequent fetches)
    this.previewsInterval = setInterval(() => {
      if (this.selectedPreviewModelId) {
        this.fetchPlacementPreviews(this.selectedPreviewModelId, false);
      }
    }, 15000);
  }

  stopPreviewsPolling() {
    if (this.previewsInterval) {
      clearInterval(this.previewsInterval);
      this.previewsInterval = null;
    }
  }

  selectPreviewModel(modelId: string | null) {
    if (modelId) {
      this.startPreviewsPolling(modelId);
    } else {
      this.stopPreviewsPolling();
      this.selectedPreviewModelId = null;
      this.placementPreviews = [];
    }
  }

  /**
   * Handle topology changes and refresh placement previews when needed.
   */
  private handleTopologyChange() {
    if (!this.topologyData) return;

    const currentNodeIds = new Set(Object.keys(this.topologyData.nodes));

    // Check if nodes have changed
    const nodesAdded = [...currentNodeIds].some(
      (id) => !this.previousNodeIds.has(id),
    );
    const nodesRemoved = [...this.previousNodeIds].some(
      (id) => !currentNodeIds.has(id),
    );

    if (nodesAdded || nodesRemoved) {
      // Re-fetch previews if we have a selected model (topology changed)
      if (this.selectedPreviewModelId) {
        this.fetchPlacementPreviews(this.selectedPreviewModelId, false);
      }
    }

    // Update tracked node IDs for next comparison
    this.previousNodeIds = currentNodeIds;
  }

  /**
   * Starts a chat conversation - triggers the topology minimization animation
   * Creates a new conversation if none is active
   */
  startChat() {
    if (!this.activeConversationId) {
      this.createConversation();
    } else {
      this.hasStartedChat = true;
      this.isSidebarOpen = true; // Auto-open sidebar when chatting
      // Small delay before minimizing for a nice visual effect
      setTimeout(() => {
        this.isTopologyMinimized = true;
      }, 100);
    }
  }

  /**
   * Add a message to the conversation
   */
  addMessage(role: "user" | "assistant", content: string) {
    const message: Message = {
      id: generateUUID(),
      role,
      content,
      timestamp: Date.now(),
    };
    this.messages.push(message);
    return message;
  }

  /**
   * Delete a message and all subsequent messages
   */
  deleteMessage(messageId: string) {
    const messageIndex = this.messages.findIndex((m) => m.id === messageId);
    if (messageIndex === -1) return;

    // Remove this message and all subsequent messages
    this.messages = this.messages.slice(0, messageIndex);
    this.updateActiveConversation();
  }

  /**
   * Edit a user message content (does not regenerate response)
   */
  editMessage(messageId: string, newContent: string) {
    const message = this.messages.find((m) => m.id === messageId);
    if (!message) return;

    message.content = newContent;
    message.timestamp = Date.now();
    this.updateActiveConversation();
  }

  /**
   * Edit a user message and regenerate the response
   */
  async editAndRegenerate(
    messageId: string,
    newContent: string,
  ): Promise<void> {
    const messageIndex = this.messages.findIndex((m) => m.id === messageId);
    if (messageIndex === -1) return;

    const message = this.messages[messageIndex];
    if (message.role !== "user") return;

    // Update the message content
    message.content = newContent;
    message.timestamp = Date.now();

    // Remove all messages after this one (including the assistant response)
    this.messages = this.messages.slice(0, messageIndex + 1);

    // Regenerate the response
    await this.regenerateLastResponse();
  }

  /**
   * Regenerate the last assistant response
   */
  async regenerateLastResponse(): Promise<void> {
    if (this.isLoading) return;

    // Find the last user message
    let lastUserIndex = -1;
    for (let i = this.messages.length - 1; i >= 0; i--) {
      if (this.messages[i].role === "user") {
        lastUserIndex = i;
        break;
      }
    }

    if (lastUserIndex === -1) return;

    const lastUserMessage = this.messages[lastUserIndex];
    const requestType = lastUserMessage.requestType || "chat";
    const prompt = lastUserMessage.content;

    // Remove messages after user message (including the user message for image requests
    // since generateImage/editImage will re-add it)
    this.messages = this.messages.slice(0, lastUserIndex);
    this.updateActiveConversation();

    switch (requestType) {
      case "image-generation":
        await this.generateImage(prompt);
        break;
      case "image-editing":
        if (lastUserMessage.sourceImageDataUrl) {
          await this.editImage(prompt, lastUserMessage.sourceImageDataUrl);
        } else {
          // Can't regenerate edit without source image - restore user message and show error
          this.messages.push(lastUserMessage);
          const errorMessage = this.addMessage("assistant", "");
          const idx = this.messages.findIndex((m) => m.id === errorMessage.id);
          if (idx !== -1) {
            this.messages[idx].content =
              "Error: Cannot regenerate image edit - source image not found";
          }
          this.updateActiveConversation();
        }
        break;
      case "chat":
      default:
        // Restore the user message for chat regeneration
        this.messages.push(lastUserMessage);
        await this.regenerateChatCompletion();
        break;
    }
  }

  /**
   * Regenerate response from a specific token index.
   * Truncates the assistant message at the given token and re-generates from there.
   */
  async regenerateFromToken(
    messageId: string,
    tokenIndex: number,
  ): Promise<void> {
    if (this.isLoading) return;

    const targetConversationId = this.activeConversationId;
    if (!targetConversationId) return;

    const msgIndex = this.messages.findIndex((m) => m.id === messageId);
    if (msgIndex === -1) return;

    const msg = this.messages[msgIndex];
    if (
      msg.role !== "assistant" ||
      !msg.tokens ||
      tokenIndex >= msg.tokens.length
    )
      return;

    // Keep tokens up to (not including) the specified index
    const tokensToKeep = msg.tokens.slice(0, tokenIndex);
    const prefixText = tokensToKeep.map((t) => t.token).join("");

    // Remove all messages after this assistant message
    this.messages = this.messages.slice(0, msgIndex + 1);

    // Update the message to show the prefix
    this.messages[msgIndex].content = prefixText;
    this.messages[msgIndex].tokens = tokensToKeep;
    this.updateActiveConversation();

    // Set up for continuation - modify the existing message in place
    this.isLoading = true;
    void this.reportUpdateActivity("request-start");
    this.currentResponse = prefixText;
    this.ttftMs = null;
    this.tps = null;
    this.totalTokens = tokensToKeep.length;

    try {
      // Build messages for API - include the partial assistant message
      const systemPrompt = {
        role: "system" as const,
        content:
          "You are a helpful AI assistant. Respond directly and concisely. Do not show your reasoning or thought process.",
      };

      const apiMessages = [
        systemPrompt,
        ...this.messages.map((m) => {
          let msgContent = m.content;
          if (m.attachments) {
            for (const attachment of m.attachments) {
              if (attachment.type === "text" && attachment.content) {
                msgContent += `\n\n[File: ${attachment.name}]\n\`\`\`\n${attachment.content}\n\`\`\``;
              }
            }
          }
          return { role: m.role, content: msgContent };
        }),
      ];

      const endpointModel = this.getModelForRequest();
      const routingModel = endpointModel ?? undefined;
      const modelToUse = this.shouldUseCaiMeteredChat(apiMessages, routingModel)
        ? this.resolveMeteredChatModelId(routingModel)
        : endpointModel;
      if (!modelToUse) {
        throw new Error(tr("No model available"));
      }

      const requestStartTime = performance.now();
      let firstTokenTime: number | null = null;
      let tokenCount = tokensToKeep.length;

      const response = await fetch(
        this.getChatCompletionsEndpoint(apiMessages, routingModel),
        {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: modelToUse,
          messages: apiMessages,
          stream: true,
          logprobs: true,
          top_logprobs: 5,
        }),
        },
      );

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
          trf("API error: {status} - {error}", {
            status: response.status,
            error: errorText,
          }),
        );
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error(tr("No response body"));

      let fullContent = prefixText;
      let streamedThinking = "";
      const collectedTokens: TokenData[] = [...tokensToKeep];

      interface ChatCompletionChunk {
        choices?: Array<{
          delta?: { content?: string; reasoning_content?: string };
          logprobs?: {
            content?: Array<{
              token: string;
              logprob: number;
              top_logprobs?: Array<{
                token: string;
                logprob: number;
                bytes: number[] | null;
              }>;
            }>;
          };
        }>;
      }

      await this.parseSSEStream<ChatCompletionChunk>(
        reader,
        targetConversationId,
        (parsed) => {
          const choice = parsed.choices?.[0];
          const delta = choice?.delta?.content;
          const thinkingDelta = choice?.delta?.reasoning_content;

          // Collect logprobs data
          const logprobsContent = choice?.logprobs?.content;
          if (logprobsContent) {
            for (const item of logprobsContent) {
              collectedTokens.push({
                token: item.token,
                logprob: item.logprob,
                probability: Math.exp(item.logprob),
                topLogprobs: (item.top_logprobs || []).map((t) => ({
                  token: t.token,
                  logprob: t.logprob,
                  bytes: t.bytes,
                })),
              });
            }
          }

          if (thinkingDelta) {
            streamedThinking += thinkingDelta;
          }

          if (delta || thinkingDelta) {
            if (firstTokenTime === null) {
              firstTokenTime = performance.now();
              this.ttftMs = firstTokenTime - requestStartTime;
            }

            tokenCount += 1;
            this.totalTokens = tokenCount;

            if (firstTokenTime !== null && tokenCount > tokensToKeep.length) {
              const elapsed = performance.now() - firstTokenTime;
              this.tps = ((tokenCount - tokensToKeep.length) / elapsed) * 1000;
            }

            if (delta) {
              fullContent += delta;
            }
            const { displayContent, thinkingContent: tagThinking } =
              this.stripThinkingTags(fullContent);
            const combinedThinking = [streamedThinking, tagThinking]
              .filter(Boolean)
              .join("\n\n");

            if (this.activeConversationId === targetConversationId) {
              this.currentResponse = displayContent;
            }

            // Update existing message in place
            this.updateConversationMessage(
              targetConversationId,
              messageId,
              (m) => {
                m.content = displayContent;
                m.thinking = combinedThinking || undefined;
                m.tokens = [...collectedTokens];
              },
            );
            this.syncActiveMessagesIfNeeded(targetConversationId);
            this.persistConversation(targetConversationId);
          }
        },
        {
          generation_stats: (data) => {
            const stats = data as { generation_tps: number };
            if (stats.generation_tps > 0) {
              this.tps = stats.generation_tps;
            }
          },
          cai_execution: (data) => {
            this.applyCaiExecutionEvent(targetConversationId, messageId, data);
          },
        },
      );

      // Final update
      if (this.conversationExists(targetConversationId)) {
        const { displayContent, thinkingContent: tagThinking } =
          this.stripThinkingTags(fullContent);
        const finalThinking = [streamedThinking, tagThinking]
          .filter(Boolean)
          .join("\n\n");
        this.updateConversationMessage(targetConversationId, messageId, (m) => {
          m.content = displayContent;
          m.thinking = finalThinking || undefined;
          m.tokens = [...collectedTokens];
          if (this.ttftMs !== null) m.ttftMs = this.ttftMs;
          if (this.tps !== null) m.tps = this.tps;
        });
        this.syncActiveMessagesIfNeeded(targetConversationId);
        this.persistConversation(targetConversationId);
      }
    } catch (error) {
      console.error("Error regenerating from token:", error);
      if (this.conversationExists(targetConversationId)) {
        this.updateConversationMessage(targetConversationId, messageId, (m) => {
          m.content = `${prefixText}\n\n${trf("Error: {message}", {
            message: error instanceof Error ? error.message : tr("Unknown error"),
          })}`;
        });
        this.syncActiveMessagesIfNeeded(targetConversationId);
        this.persistConversation(targetConversationId);
      }
    } finally {
      this.isLoading = false;
      void this.reportUpdateActivity("request-end");
      this.currentResponse = "";
      this.saveConversationsToStorage();
    }
  }

  /**
   * Helper method to regenerate a chat completion response
   */
  private async regenerateChatCompletion(): Promise<void> {
    // Capture the target conversation ID at the start of the request
    const targetConversationId = this.activeConversationId;
    if (!targetConversationId) return;

    const targetConversation = this.conversations.find(
      (c) => c.id === targetConversationId,
    );
    if (!targetConversation) return;

    this.isLoading = true;
    void this.reportUpdateActivity("request-start");
    this.currentResponse = "";

    // Create placeholder for assistant message directly in target conversation
    const assistantMessage = this.addMessageToConversation(
      targetConversationId,
      "assistant",
      "",
    );
    if (!assistantMessage) {
      this.isLoading = false;
      void this.reportUpdateActivity("request-end");
      return;
    }

    // Sync to this.messages if viewing the target conversation
    this.syncActiveMessagesIfNeeded(targetConversationId);

    try {
      const systemPrompt = {
        role: "system" as const,
        content:
          "You are a helpful AI assistant. Respond directly and concisely. Do not show your reasoning or thought process.",
      };

      const apiMessages = [
        systemPrompt,
        ...targetConversation.messages.slice(0, -1).map((m) => {
          return { role: m.role, content: m.content };
        }),
      ];

      // Determine which model to use
      const endpointModel = this.getModelForRequest();
      const routingModel = endpointModel ?? undefined;
      const modelToUse = this.shouldUseCaiMeteredChat(apiMessages, routingModel)
        ? this.resolveMeteredChatModelId(routingModel)
        : endpointModel;
      if (!modelToUse) {
        this.updateConversationMessage(
          targetConversationId,
          assistantMessage.id,
          (msg) => {
            msg.content = tr(
              "No model is loaded yet. Select a model from the sidebar to get started - it will download and load automatically.",
            );
          },
        );
        this.syncActiveMessagesIfNeeded(targetConversationId);
        this.isLoading = false;
        void this.reportUpdateActivity("request-end");
        this.saveConversationsToStorage();
        return;
      }

      const response = await fetch(
        this.getChatCompletionsEndpoint(apiMessages, routingModel),
        {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: modelToUse,
          messages: apiMessages,
          stream: true,
          logprobs: true,
          top_logprobs: 5,
        }),
        },
      );

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`${response.status} - ${errorText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error(tr("No response stream available"));
      }

      let streamedContent = "";
      let streamedThinking = "";
      const collectedTokens: TokenData[] = [];

      interface ChatCompletionChunk {
        choices?: Array<{
          delta?: { content?: string; reasoning_content?: string };
          logprobs?: {
            content?: Array<{
              token: string;
              logprob: number;
              top_logprobs?: Array<{
                token: string;
                logprob: number;
                bytes: number[] | null;
              }>;
            }>;
          };
        }>;
      }

      await this.parseSSEStream<ChatCompletionChunk>(
        reader,
        targetConversationId,
        (parsed) => {
          const choice = parsed.choices?.[0];
          const delta = choice?.delta?.content;
          const thinkingDelta = choice?.delta?.reasoning_content;

          // Collect logprobs data
          const logprobsContent = choice?.logprobs?.content;
          if (logprobsContent) {
            for (const item of logprobsContent) {
              collectedTokens.push({
                token: item.token,
                logprob: item.logprob,
                probability: Math.exp(item.logprob),
                topLogprobs: (item.top_logprobs || []).map((t) => ({
                  token: t.token,
                  logprob: t.logprob,
                  bytes: t.bytes,
                })),
              });
            }
          }

          if (thinkingDelta) {
            streamedThinking += thinkingDelta;
          }

          if (delta || thinkingDelta) {
            if (delta) {
              streamedContent += delta;
            }
            const { displayContent, thinkingContent: tagThinking } =
              this.stripThinkingTags(streamedContent);
            const combinedThinking = [streamedThinking, tagThinking]
              .filter(Boolean)
              .join("\n\n");

            // Only update currentResponse if target conversation is active
            if (this.activeConversationId === targetConversationId) {
              this.currentResponse = displayContent;
            }

            // Update the assistant message in the target conversation
            this.updateConversationMessage(
              targetConversationId,
              assistantMessage.id,
              (msg) => {
                msg.content = displayContent;
                msg.thinking = combinedThinking || undefined;
                msg.tokens = [...collectedTokens];
              },
            );
            this.syncActiveMessagesIfNeeded(targetConversationId);
            this.persistConversation(targetConversationId);
          }
        },
        {
          generation_stats: (data) => {
            const stats = data as { generation_tps: number };
            if (stats.generation_tps > 0) {
              this.tps = stats.generation_tps;
            }
          },
          cai_execution: (data) => {
            this.applyCaiExecutionEvent(
              targetConversationId,
              assistantMessage.id,
              data,
            );
          },
        },
      );

      // Final cleanup of the message (if conversation still exists)
      if (this.conversationExists(targetConversationId)) {
        const { displayContent, thinkingContent: tagThinking } =
          this.stripThinkingTags(streamedContent);
        const finalThinking = [streamedThinking, tagThinking]
          .filter(Boolean)
          .join("\n\n");
        this.updateConversationMessage(
          targetConversationId,
          assistantMessage.id,
          (msg) => {
            msg.content = displayContent;
            msg.thinking = finalThinking || undefined;
            msg.tokens = [...collectedTokens];
          },
        );
        this.syncActiveMessagesIfNeeded(targetConversationId);
        this.persistConversation(targetConversationId);
      }
    } catch (error) {
      this.handleStreamingError(
        error,
        targetConversationId,
        assistantMessage.id,
        "Unknown error",
      );
    } finally {
      this.isLoading = false;
      void this.reportUpdateActivity("request-end");
      this.currentResponse = "";
      this.saveConversationsToStorage();
    }
  }

  /**
   * Whether thinking is enabled for the current conversation
   */
  thinkingEnabled = $state(true);

  /**
   * Selected model for chat (can be set by the UI)
   */
  selectedChatModel = $state("");

  /**
   * Set the model to use for chat
   */
  setSelectedModel(modelId: string) {
    this.selectedChatModel = modelId;
    // Clear stats when model changes
    this.ttftMs = null;
    this.tps = null;
  }

  /**
   * Strip thinking tags from content for display.
   * Handles both complete <think>...</think> blocks and in-progress <think>... blocks during streaming.
   */
  private stripThinkingTags(content: string): {
    displayContent: string;
    thinkingContent: string;
  } {
    const extracted: string[] = [];
    let displayContent = content;

    // Extract complete <think>...</think> blocks
    const completeBlockRegex = /<think>([\s\S]*?)<\/think>/gi;
    let match: RegExpExecArray | null;
    while ((match = completeBlockRegex.exec(content)) !== null) {
      const inner = match[1]?.trim();
      if (inner) extracted.push(inner);
    }
    displayContent = displayContent.replace(completeBlockRegex, "");

    // Handle in-progress thinking block (has <think> but no closing </think> yet)
    const openTagIndex = displayContent.lastIndexOf("<think>");
    if (openTagIndex !== -1) {
      const inProgressThinking = displayContent.slice(openTagIndex + 7).trim();
      if (inProgressThinking) {
        extracted.push(inProgressThinking);
      }
      displayContent = displayContent.slice(0, openTagIndex);
    }

    return {
      displayContent: displayContent.trim(),
      thinkingContent: extracted.join("\n\n"),
    };
  }

  /**
   * Parse an SSE stream and invoke a callback for each parsed JSON chunk.
   * Handles buffering, line splitting, and conversation deletion checks.
   *
   * @param reader - The stream reader from fetch response.body.getReader()
   * @param targetConversationId - The conversation ID to check for deletion
   * @param onChunk - Callback invoked with each parsed JSON object from the stream
   */
  private async parseSSEStream<T>(
    reader: ReadableStreamDefaultReader<Uint8Array>,
    targetConversationId: string,
    onChunk: (parsed: T) => void,
    onEvent?: Record<string, (data: unknown) => void>,
  ): Promise<void> {
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      if (!this.conversationExists(targetConversationId)) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;

        // Handle SSE comments (": key json") for prefill progress etc.
        if (trimmed.startsWith(": ") && onEvent) {
          const comment = trimmed.slice(2);
          const spaceIdx = comment.indexOf(" ");
          if (spaceIdx > 0) {
            const key = comment.slice(0, spaceIdx);
            if (onEvent[key]) {
              try {
                const parsed = JSON.parse(comment.slice(spaceIdx + 1));
                onEvent[key](parsed);
              } catch {
                // Skip malformed JSON in comment
              }
            }
          }
          continue;
        }

        if (trimmed.startsWith("data: ")) {
          const data = trimmed.slice(6);
          if (data === "[DONE]") continue;

          try {
            const parsed = JSON.parse(data) as T;
            onChunk(parsed);
          } catch {
            // Skip malformed JSON
          }
        }
      }
    }

    // Process any remaining data in the buffer
    if (buffer.trim() && this.conversationExists(targetConversationId)) {
      const trimmed = buffer.trim();
      if (trimmed.startsWith("data: ") && trimmed.slice(6) !== "[DONE]") {
        try {
          const parsed = JSON.parse(trimmed.slice(6)) as T;
          onChunk(parsed);
        } catch {
          // Skip malformed JSON
        }
      }
    }
  }

  /**
   * Handle streaming errors by updating the assistant message with an error.
   *
   * @param error - The caught error
   * @param targetConversationId - The conversation ID
   * @param assistantMessageId - The assistant message ID to update
   * @param errorPrefix - Optional prefix for the error message (e.g., "Failed to generate image")
   */
  private handleStreamingError(
    error: unknown,
    targetConversationId: string,
    assistantMessageId: string,
    errorPrefix = "Failed to get response",
  ): void {
    if (this.conversationExists(targetConversationId)) {
      this.updateConversationMessage(
        targetConversationId,
        assistantMessageId,
        (msg) => {
          msg.content = trf("Error: {message}", {
            message: error instanceof Error ? error.message : tr(errorPrefix),
          });
        },
      );
      this.syncActiveMessagesIfNeeded(targetConversationId);
      this.persistConversation(targetConversationId);
    }
  }

  private maybeDispatchWalletAccessRequired(error: unknown): void {
    const message = error instanceof Error ? error.message : "";
    if (!message) {
      return;
    }
    const walletLockedMessage =
      "Active wallet must be unlocked before executing a job.";
    const walletMessages = [
      t("error.walletNotReady"),
      t("error.walletCreate"),
      t("error.walletUnlock"),
      walletLockedMessage,
    ].filter(Boolean);
    if (!walletMessages.some((entry) => message.includes(entry))) {
      return;
    }
    const normalizedMessage = message.includes(walletLockedMessage)
      ? t("error.walletUnlock")
      : message.includes(t("error.walletNotReady"))
        ? t("error.walletNotReady")
        : message.includes(t("error.walletCreate"))
          ? t("error.walletCreate")
          : t("error.walletUnlock");
    dispatchWalletAccessRequiredEvent(normalizedMessage);
  }

  private applyCaiExecutionEvent(
    targetConversationId: string,
    assistantMessageId: string,
    data: unknown,
  ): void {
    if (!data || typeof data !== "object") {
      return;
    }
    const meta = data as CaiExecutionMeta;
    this.updateConversationMessage(
      targetConversationId,
      assistantMessageId,
      (msg) => {
        msg.caiExecution = {
          ...(msg.caiExecution ?? {}),
          ...meta,
          executionAttempts: Array.isArray(meta.executionAttempts)
            ? [...meta.executionAttempts]
            : (msg.caiExecution?.executionAttempts ?? []),
          executionAttemptStatus:
            meta.executionAttemptStatus ??
            msg.caiExecution?.executionAttemptStatus ??
            null,
          executorNodeIds: Array.isArray(meta.executorNodeIds)
            ? [...meta.executorNodeIds]
            : (msg.caiExecution?.executorNodeIds ?? []),
          participantNodeIds: Array.isArray(meta.participantNodeIds)
            ? [...meta.participantNodeIds]
            : (msg.caiExecution?.participantNodeIds ?? []),
          rewardPayoutNodeIds: Array.isArray(meta.rewardPayoutNodeIds)
            ? [...meta.rewardPayoutNodeIds]
            : (msg.caiExecution?.rewardPayoutNodeIds ?? []),
          rewardSkippedNodeIdsWithoutShardReceipt: Array.isArray(
            meta.rewardSkippedNodeIdsWithoutShardReceipt,
          )
            ? [...meta.rewardSkippedNodeIdsWithoutShardReceipt]
            : (msg.caiExecution?.rewardSkippedNodeIdsWithoutShardReceipt ?? []),
          payoutNodes: Array.isArray(meta.payoutNodes)
            ? [...meta.payoutNodes]
            : (msg.caiExecution?.payoutNodes ?? []),
          payoutStatuses: Array.isArray(meta.payoutStatuses)
            ? [...meta.payoutStatuses]
            : (msg.caiExecution?.payoutStatuses ?? []),
        };
      },
    );
    this.syncActiveMessagesIfNeeded(targetConversationId);
    this.persistConversation(targetConversationId);
  }

  /**
   * Get the model to use for a request.
   * Prefers the provided modelId, then selectedChatModel, then falls back to the first running instance.
   *
   * @param modelId - Optional explicit model ID
   * @returns The model ID to use, or null if none available
   */
  private getWorkerCompatibleTextModelIds(): Set<string> {
    const ids = new Set<string>();
    const workerSummary = this.caiSummary?.worker;
    const candidates = [
      ...(workerSummary?.allowed_model_ids ?? []),
      workerSummary?.network_default_model_id ?? "",
      workerSummary?.network_default_execution_model_id ?? "",
    ];
    for (const candidate of candidates) {
      const normalized = candidate.trim();
      if (normalized) {
        ids.add(normalized);
      }
    }
    return ids;
  }

  private isWorkerCompatibleTextModelId(
    modelId: string | null | undefined,
  ): boolean {
    const normalizedModelId = modelId?.trim() || "";
    if (!normalizedModelId) {
      return false;
    }
    const compatibleModelIds = this.getWorkerCompatibleTextModelIds();
    return compatibleModelIds.size > 0 && compatibleModelIds.has(normalizedModelId);
  }

  private resolvePreferredTextChatModel(
    modelId?: string | null,
  ): string | null {
    const workerSummary = this.caiSummary?.worker;
    const requestedModelId = modelId?.trim() || "";
    const networkDefaultModelId =
      workerSummary?.network_default_model_id?.trim() || "";
    const executionModelId =
      workerSummary?.network_default_execution_model_id?.trim() || "";
    const compatibleModelIds = this.getWorkerCompatibleTextModelIds();

    if (!requestedModelId) {
      return networkDefaultModelId || executionModelId || null;
    }
    if (
      this.hasRunningInstanceForModel(requestedModelId) ||
      compatibleModelIds.size === 0
    ) {
      return requestedModelId;
    }
    if (compatibleModelIds.has(requestedModelId)) {
      if (
        requestedModelId === executionModelId &&
        networkDefaultModelId &&
        executionModelId !== networkDefaultModelId
      ) {
        return networkDefaultModelId;
      }
      if (
        requestedModelId === networkDefaultModelId &&
        executionModelId &&
        executionModelId !== requestedModelId
      ) {
        return networkDefaultModelId;
      }
      return requestedModelId;
    }
    return networkDefaultModelId || executionModelId || requestedModelId;
  }

  private getModelForRequest(modelId?: string): string | null {
    const preferredModelId = this.resolvePreferredTextChatModel(
      modelId || this.selectedChatModel || null,
    );
    if (preferredModelId) return preferredModelId;

    // Try to get model from first running instance
    for (const [, instanceWrapper] of Object.entries(this.instances)) {
      if (instanceWrapper && typeof instanceWrapper === "object") {
        const keys = Object.keys(instanceWrapper as Record<string, unknown>);
        if (keys.length === 1) {
          const instance = (instanceWrapper as Record<string, unknown>)[
            keys[0]
          ] as { shardAssignments?: { modelId?: string } };
          if (instance?.shardAssignments?.modelId) {
            return instance.shardAssignments.modelId;
          }
        }
      }
    }
    return null;
  }

  private resolveMeteredChatModelId(modelId?: string): string | null {
    const resolvedModelId = this.getModelForRequest(modelId);
    if (!resolvedModelId) {
      return null;
    }
    const workerSummary = this.caiSummary?.worker;
    const networkDefaultModelId = workerSummary?.network_default_model_id;
    const executionModelId = workerSummary?.network_default_execution_model_id;
    if (
      resolvedModelId === executionModelId &&
      networkDefaultModelId &&
      executionModelId &&
      executionModelId !== networkDefaultModelId
    ) {
      return networkDefaultModelId;
    }
    return resolvedModelId;
  }

  private isTextOnlyChatMessages(
    messages: Array<{ content?: unknown }>,
  ): boolean {
    return messages.every((message) => typeof message.content === "string");
  }

  private shouldUseCaiMeteredChat(
    messages: Array<{ content?: unknown }>,
    modelId?: string,
  ): boolean {
    if (!this.shouldRouteViaCaiMeteredChat(messages, modelId)) return false;
    return Boolean(
      this.caiSummary?.available &&
        this.caiSummary.wallet?.has_active_wallet &&
        this.caiSummary.wallet?.unlocked,
    );
  }

  private shouldRouteViaCaiMeteredChat(
    messages: Array<{ content?: unknown }>,
    modelId?: string,
  ): boolean {
    if (!this.isTextOnlyChatMessages(messages)) {
      return false;
    }

    const meteredModelId = this.resolveMeteredChatModelId(modelId);
    return this.isWorkerCompatibleTextModelId(meteredModelId);
  }

  private requireCaiWalletForMeteredChat(
    messages: Array<{ content?: unknown }>,
    modelId?: string,
  ): void {
    if (!this.shouldRouteViaCaiMeteredChat(messages, modelId)) {
      return;
    }
    if (!this.caiSummary?.available) {
      throw new Error(t("error.walletNotReady"));
    }
    if (!this.caiSummary.wallet?.has_active_wallet) {
      throw new Error(t("error.walletCreate"));
    }
    if (!this.caiSummary.wallet?.unlocked) {
      throw new Error(t("error.walletUnlock"));
    }
  }

  private getChatCompletionsEndpoint(
    messages: Array<{ content?: unknown }>,
    modelId?: string,
  ): string {
    if (this.shouldRouteViaCaiMeteredChat(messages, modelId)) {
      this.requireCaiWalletForMeteredChat(messages, modelId);
      return "/v1/cai/chat/completions";
    }
    return "/v1/chat/completions";
  }

  /**
   * Send a message to the LLM and stream the response
   */
  async sendMessage(
    content: string,
    files?: {
      id: string;
      name: string;
      type: string;
      textContent?: string;
      preview?: string;
      pageImages?: string[];
    }[],
    enableThinking?: boolean | null,
  ): Promise<void> {
    if ((!content.trim() && (!files || files.length === 0)) || this.isLoading)
      return;

    if (!this.hasStartedChat) {
      this.startChat();
    }

    // Capture the target conversation ID at the start of the request
    const targetConversationId = this.activeConversationId;
    if (!targetConversationId) return;

    this.isLoading = true;
    void this.reportUpdateActivity("request-start");
    this.currentResponse = "";
    this.ttftMs = null;
    this.tps = null;
    this.totalTokens = 0;

    // Build attachments from files
    const attachments: MessageAttachment[] = [];
    let fileContext = "";

    if (files && files.length > 0) {
      for (const file of files) {
        const isImage = file.type.startsWith("image/");

        if (isImage && file.preview) {
          attachments.push({
            type: "image",
            name: file.name,
            preview: file.preview,
            mimeType: file.type,
          });
        } else if (
          file.pageImages ||
          (file.textContent && file.type === "application/pdf")
        ) {
          attachments.push({
            type: "pdf",
            name: file.name,
            content: file.textContent,
            pageImages: file.pageImages,
            mimeType: file.type,
          });
          if (file.textContent) {
            fileContext += `\n\n[File: ${file.name}]\n\`\`\`\n${file.textContent}\n\`\`\``;
          }
        } else if (file.textContent) {
          attachments.push({
            type: "text",
            name: file.name,
            content: file.textContent,
            mimeType: file.type,
          });
          // Add text file content to the message context
          fileContext += `\n\n[File: ${file.name}]\n\`\`\`\n${file.textContent}\n\`\`\``;
        } else {
          attachments.push({
            type: "file",
            name: file.name,
            mimeType: file.type,
          });
        }
      }
    }

    // Combine content with file context
    const fullContent = content + fileContext;

    // Add user message directly to the target conversation
    const userMessage: Message = {
      id: generateUUID(),
      role: "user",
      content: content, // Store original content for display
      timestamp: Date.now(),
      attachments: attachments.length > 0 ? attachments : undefined,
    };

    const targetConversation = this.conversations.find(
      (c) => c.id === targetConversationId,
    );
    if (!targetConversation) {
      this.isLoading = false;
      void this.reportUpdateActivity("request-end");
      return;
    }
    targetConversation.messages.push(userMessage);

    // Create placeholder for assistant message directly in target conversation
    const assistantMessage = this.addMessageToConversation(
      targetConversationId,
      "assistant",
      "",
    );
    if (!assistantMessage) {
      this.isLoading = false;
      void this.reportUpdateActivity("request-end");
      return;
    }

    // Sync to this.messages if viewing the target conversation
    this.syncActiveMessagesIfNeeded(targetConversationId);
    this.saveConversationsToStorage();

    try {
      // Build the messages array for the API with system prompt
      const systemPrompt = {
        role: "system" as const,
        content:
          "You are a helpful AI assistant. Respond directly and concisely. Do not show your reasoning or thought process. When files are shared with you, analyze them and respond helpfully.",
      };

      // Build API messages from the target conversation - include file content for text files
      const apiMessages = [
        systemPrompt,
        ...targetConversation.messages.slice(0, -1).map((m) => {
          // Check if this message has image or PDF attachments
          const visualAttachments = m.attachments?.filter(
            (a) =>
              (a.type === "image" && a.preview) ||
              (a.type === "pdf" && a.pageImages?.length),
          );

          if (visualAttachments && visualAttachments.length > 0) {
            // Build multimodal content array (OpenAI vision format)
            const contentParts: Array<
              | { type: "text"; text: string }
              | { type: "image_url"; image_url: { url: string } }
            > = [];

            // Add image parts first
            for (const att of visualAttachments) {
              if (att.type === "image" && att.preview) {
                contentParts.push({
                  type: "image_url",
                  image_url: { url: att.preview },
                });
              } else if (att.type === "pdf" && att.pageImages) {
                for (const pageImg of att.pageImages) {
                  contentParts.push({
                    type: "image_url",
                    image_url: { url: pageImg },
                  });
                }
              }
            }

            // Build text content including any text/pdf file attachments
            let textContent = m.content;
            if (m.attachments) {
              for (const attachment of m.attachments) {
                if (
                  (attachment.type === "text" || attachment.type === "pdf") &&
                  attachment.content
                ) {
                  textContent += `\n\n[File: ${attachment.name}]\n\`\`\`\n${attachment.content}\n\`\`\``;
                }
              }
            }

            if (textContent) {
              contentParts.push({ type: "text", text: textContent });
            }

            return {
              role: m.role,
              content: contentParts,
            };
          }

          // Text-only message (original path)
          let msgContent = m.content;

          // Add text/pdf attachments as context
          if (m.attachments) {
            for (const attachment of m.attachments) {
              if (
                (attachment.type === "text" || attachment.type === "pdf") &&
                attachment.content
              ) {
                msgContent += `\n\n[File: ${attachment.name}]\n\`\`\`\n${attachment.content}\n\`\`\``;
              }
            }
          }

          return {
            role: m.role,
            content: msgContent,
          };
        }),
      ];

      // Determine the model to use
      const endpointModel = this.getModelForRequest();
      const routingModel = endpointModel ?? undefined;
      const modelToUse = this.shouldUseCaiMeteredChat(apiMessages, routingModel)
        ? this.resolveMeteredChatModelId(routingModel)
        : endpointModel;
      if (!modelToUse) {
        throw new Error(
          tr(
            "No model is loaded yet. Select a model from the sidebar to get started - it will download and load automatically.",
          ),
        );
      }

      const conversationModelInfo = this.buildConversationModelInfo(modelToUse);
      this.applyConversationModelInfo(conversationModelInfo);

      // Start timing for TTFT measurement
      const requestStartTime = performance.now();
      let firstTokenTime: number | null = null;
      let tokenCount = 0;

      const abortController = new AbortController();
      this.currentAbortController = abortController;

      const response = await fetch(
        this.getChatCompletionsEndpoint(apiMessages, routingModel),
        {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: modelToUse,
          messages: apiMessages,
          temperature: 0.7,
          stream: true,
          logprobs: true,
          top_logprobs: 5,
          ...(enableThinking != null && {
            enable_thinking: enableThinking,
          }),
        }),
        signal: abortController.signal,
        },
      );

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
          trf("API error: {status} - {error}", {
            status: response.status,
            error: errorText,
          }),
        );
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error(tr("No response body"));
      }

      let streamedContent = "";
      let streamedThinking = "";
      let serverTpsReceived = false;
      interface ChatCompletionChunk {
        choices?: Array<{
          delta?: { content?: string; reasoning_content?: string };
          logprobs?: {
            content?: Array<{
              token: string;
              logprob: number;
              top_logprobs?: Array<{
                token: string;
                logprob: number;
                bytes: number[] | null;
              }>;
            }>;
          };
        }>;
      }

      const collectedTokens: TokenData[] = [];

      await this.parseSSEStream<ChatCompletionChunk>(
        reader,
        targetConversationId,
        (parsed) => {
          // Clear prefill progress when first token data arrives
          if (this.prefillProgress) {
            this.prefillProgress = null;
          }

          const choice = parsed.choices?.[0];
          const tokenContent = choice?.delta?.content;
          const thinkingContent = choice?.delta?.reasoning_content;

          // Collect logprobs data
          const logprobsContent = choice?.logprobs?.content;
          if (logprobsContent) {
            for (const item of logprobsContent) {
              collectedTokens.push({
                token: item.token,
                logprob: item.logprob,
                probability: Math.exp(item.logprob),
                topLogprobs: (item.top_logprobs || []).map((t) => ({
                  token: t.token,
                  logprob: t.logprob,
                  bytes: t.bytes,
                })),
              });
            }
          }

          if (thinkingContent) {
            streamedThinking += thinkingContent;
          }

          if (tokenContent || thinkingContent) {
            // Track first token for TTFT
            if (firstTokenTime === null) {
              firstTokenTime = performance.now();
              this.ttftMs = firstTokenTime - requestStartTime;
            }

            // Count tokens (each SSE chunk is typically one token)
            tokenCount += 1;
            this.totalTokens = tokenCount;

            if (firstTokenTime !== null && tokenCount > 1) {
              const elapsed = performance.now() - firstTokenTime;
              this.tps = (tokenCount / elapsed) * 1000;
            }

            if (tokenContent) {
              streamedContent += tokenContent;
            }

            // Use stripThinkingTags as fallback for any <think> tags still in content
            const { displayContent, thinkingContent: tagThinking } =
              this.stripThinkingTags(streamedContent);
            const combinedThinking = [streamedThinking, tagThinking]
              .filter(Boolean)
              .join("\n\n");

            // Only update currentResponse if target conversation is active
            if (this.activeConversationId === targetConversationId) {
              this.currentResponse = displayContent;
            }

            // Update the assistant message in the target conversation
            this.updateConversationMessage(
              targetConversationId,
              assistantMessage.id,
              (msg) => {
                msg.content = displayContent;
                msg.thinking = combinedThinking || undefined;
                msg.tokens = [...collectedTokens];
              },
            );
            this.syncActiveMessagesIfNeeded(targetConversationId);
            this.persistConversation(targetConversationId);
          }
        },
        {
          prefill_progress: (data) => {
            // TaggedModel wraps as {"PrefillProgressChunk": {...}}
            // model_dump_json() uses snake_case (by_alias defaults to False)
            const raw = data as Record<string, unknown>;
            const inner = (raw["PrefillProgressChunk"] ?? raw) as {
              processed_tokens: number;
              total_tokens: number;
            };
            this.prefillProgress = {
              processed: inner.processed_tokens,
              total: inner.total_tokens,
              startedAt: this.prefillProgress?.startedAt ?? performance.now(),
            };
          },
          generation_stats: (data) => {
            const stats = data as { generation_tps: number };

            if (stats.generation_tps > 0) {
              this.tps = stats.generation_tps;
              serverTpsReceived = true;
            }
          },
          cai_execution: (data) => {
            this.applyCaiExecutionEvent(
              targetConversationId,
              assistantMessage.id,
              data,
            );
          },
        },
      );

      // Clear prefill progress after stream ends
      this.prefillProgress = null;

      // Use server-side TPS if available, otherwise fall back to client-side
      if (!serverTpsReceived && firstTokenTime !== null && tokenCount > 1) {
        const totalGenerationTime = performance.now() - firstTokenTime;
        this.tps = (tokenCount / totalGenerationTime) * 1000;
      }

      // Final cleanup of the message (if conversation still exists)
      if (this.conversationExists(targetConversationId)) {
        const { displayContent, thinkingContent: tagThinking } =
          this.stripThinkingTags(streamedContent);
        const finalThinking = [streamedThinking, tagThinking]
          .filter(Boolean)
          .join("\n\n");
        this.updateConversationMessage(
          targetConversationId,
          assistantMessage.id,
          (msg) => {
            msg.content = displayContent;
            msg.thinking = finalThinking || undefined;
            msg.tokens = [...collectedTokens];
            // Store performance metrics on the message
            if (this.ttftMs !== null) {
              msg.ttftMs = this.ttftMs;
            }
            if (this.tps !== null) {
              msg.tps = this.tps;
            }
          },
        );
        this.syncActiveMessagesIfNeeded(targetConversationId);
        this.persistConversation(targetConversationId);
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        // User stopped generation — not an error
      } else {
        console.error("Error sending message:", error);
        this.maybeDispatchWalletAccessRequired(error);
        this.handleStreamingError(
          error,
          targetConversationId,
          assistantMessage.id,
          "Failed to get response",
        );
      }
    } finally {
      this.currentAbortController = null;
      this.prefillProgress = null;
      this.isLoading = false;
      void this.reportUpdateActivity("request-end");
      this.currentResponse = "";
      this.saveConversationsToStorage();
    }
  }

  stopGeneration(): void {
    this.currentAbortController?.abort();
    this.currentAbortController = null;
  }

  /**
   * Generate an image using the image generation API
   */
  async generateImage(prompt: string, modelId?: string): Promise<void> {
    if (!prompt.trim() || this.isLoading) return;

    if (!this.hasStartedChat) {
      this.startChat();
    }

    // Capture the target conversation ID at the start of the request
    const targetConversationId = this.activeConversationId;
    if (!targetConversationId) return;

    this.isLoading = true;
    void this.reportUpdateActivity("request-start");
    this.currentResponse = "";

    // Add user message directly to the target conversation
    const userMessage: Message = {
      id: generateUUID(),
      role: "user",
      content: prompt,
      timestamp: Date.now(),
      requestType: "image-generation",
    };

    const targetConversation = this.conversations.find(
      (c) => c.id === targetConversationId,
    );
    if (!targetConversation) {
      this.isLoading = false;
      void this.reportUpdateActivity("request-end");
      return;
    }
    targetConversation.messages.push(userMessage);

    // Create placeholder for assistant message directly in target conversation
    const assistantMessage = this.addMessageToConversation(
      targetConversationId,
      "assistant",
      tr("Generating image..."),
    );
    if (!assistantMessage) {
      this.isLoading = false;
      void this.reportUpdateActivity("request-end");
      return;
    }

    // Sync to this.messages if viewing the target conversation
    this.syncActiveMessagesIfNeeded(targetConversationId);
    this.saveConversationsToStorage();

    const abortController = new AbortController();
    this.currentAbortController = abortController;

    try {
      // Determine the model to use
      const model = this.getModelForRequest(modelId);
      if (!model) {
        throw new Error(
          tr("No model selected. Please select an image generation model."),
        );
      }

      // Build request body using image generation params
      const params = this.imageGenerationParams;
      const hasAdvancedParams =
        params.seed !== null ||
        params.numInferenceSteps !== null ||
        params.guidance !== null ||
        (params.negativePrompt !== null &&
          params.negativePrompt.trim() !== "") ||
        params.numSyncSteps !== null;

      const requestBody: Record<string, unknown> = {
        model,
        prompt,
        n: params.numImages,
        quality: params.quality,
        size: params.size,
        output_format: params.outputFormat,
        response_format: "b64_json",
        stream: params.stream,
        partial_images: params.partialImages,
      };

      if (hasAdvancedParams) {
        requestBody.advanced_params = {
          ...(params.seed !== null && { seed: params.seed }),
          ...(params.numInferenceSteps !== null && {
            num_inference_steps: params.numInferenceSteps,
          }),
          ...(params.guidance !== null && { guidance: params.guidance }),
          ...(params.negativePrompt !== null &&
            params.negativePrompt.trim() !== "" && {
              negative_prompt: params.negativePrompt,
            }),
          ...(params.numSyncSteps !== null && {
            num_sync_steps: params.numSyncSteps,
          }),
        };
      }

      const response = await fetch("/v1/images/generations", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
        signal: abortController.signal,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
          trf("API error: {status} - {error}", {
            status: response.status,
            error: errorText,
          }),
        );
      }

      // Streaming requires both stream=true AND partialImages > 0
      const isStreaming = params.stream && params.partialImages > 0;

      if (!isStreaming) {
        // Non-streaming: parse JSON response directly
        const jsonResponse = (await response.json()) as ImageApiResponse;
        const format = params.outputFormat || "png";
        const mimeType = `image/${format}`;

        const attachments: MessageAttachment[] = jsonResponse.data
          .filter((img) => img.b64_json)
          .map((img, index) => ({
            type: "generated-image" as const,
            name: `generated-image-${index + 1}.${format}`,
            preview: `data:${mimeType};base64,${img.b64_json}`,
            mimeType,
          }));

        this.updateConversationMessage(
          targetConversationId,
          assistantMessage.id,
          (msg) => {
            msg.content = "";
            msg.attachments = attachments;
          },
        );
        this.syncActiveMessagesIfNeeded(targetConversationId);
      } else {
        // Streaming mode: use SSE parser
        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error(tr("No response body"));
        }

        interface ImageGenerationChunk {
          data?: { b64_json?: string };
          format?: string;
          type?: "partial" | "final";
          image_index?: number;
          partial_index?: number;
          total_partials?: number;
        }

        const numImages = params.numImages;

        await this.parseSSEStream<ImageGenerationChunk>(
          reader,
          targetConversationId,
          (parsed) => {
            const imageData = parsed.data?.b64_json;

            if (imageData) {
              const format = parsed.format || "png";
              const mimeType = `image/${format}`;
              const imageIndex = parsed.image_index ?? 0;

              if (parsed.type === "partial") {
                // Update with partial image and progress
                const partialNum = (parsed.partial_index ?? 0) + 1;
                const totalPartials = parsed.total_partials ?? 3;
                const progressText =
                  numImages > 1
                    ? trf("Generating image {index}/{total}... {partial}/{partials}", {
                        index: imageIndex + 1,
                        total: numImages,
                        partial: partialNum,
                        partials: totalPartials,
                      })
                    : trf("Generating... {partial}/{partials}", {
                        partial: partialNum,
                        partials: totalPartials,
                      });

                const partialAttachment: MessageAttachment = {
                  type: "generated-image",
                  name: `generated-image.${format}`,
                  preview: `data:${mimeType};base64,${imageData}`,
                  mimeType,
                };

                this.updateConversationMessage(
                  targetConversationId,
                  assistantMessage.id,
                  (msg) => {
                    msg.content = progressText;
                    if (imageIndex === 0) {
                      // First image - safe to replace attachments with partial preview
                      msg.attachments = [partialAttachment];
                    } else {
                      // Subsequent images - keep existing finals, show partial at current position
                      const existingAttachments = msg.attachments || [];
                      // Keep only the completed final images (up to current imageIndex)
                      const finals = existingAttachments.slice(0, imageIndex);
                      msg.attachments = [...finals, partialAttachment];
                    }
                  },
                );
              } else if (parsed.type === "final") {
                // Final image - replace partial at this position
                const newAttachment: MessageAttachment = {
                  type: "generated-image",
                  name: `generated-image-${imageIndex + 1}.${format}`,
                  preview: `data:${mimeType};base64,${imageData}`,
                  mimeType,
                };

                this.updateConversationMessage(
                  targetConversationId,
                  assistantMessage.id,
                  (msg) => {
                    if (imageIndex === 0) {
                      // First final image - replace any partial preview
                      msg.attachments = [newAttachment];
                    } else {
                      // Subsequent images - keep previous finals, replace partial at current position
                      const existingAttachments = msg.attachments || [];
                      // Slice keeps indices 0 to imageIndex-1 (the previous final images)
                      const previousFinals = existingAttachments.slice(
                        0,
                        imageIndex,
                      );
                      msg.attachments = [...previousFinals, newAttachment];
                    }

                    // Update progress message for multiple images
                    if (numImages > 1 && imageIndex < numImages - 1) {
                      msg.content = trf("Generating image {index}/{total}...", {
                        index: imageIndex + 2,
                        total: numImages,
                      });
                    } else {
                      msg.content = "";
                    }
                  },
                );
              }

              this.syncActiveMessagesIfNeeded(targetConversationId);
            }
          },
        );
      }
    } catch (error) {
      if (abortController.signal.aborted) {
        this.updateConversationMessage(
          targetConversationId,
          assistantMessage.id,
          (msg) => {
            msg.content = tr("Cancelled");
            msg.attachments = [];
          },
        );
        this.syncActiveMessagesIfNeeded(targetConversationId);
      } else {
        console.error("Error generating image:", error);
        this.handleStreamingError(
          error,
          targetConversationId,
          assistantMessage.id,
          tr("Failed to generate image"),
        );
      }
    } finally {
      this.currentAbortController = null;
      this.isLoading = false;
      void this.reportUpdateActivity("request-end");
      this.saveConversationsToStorage();
    }
  }

  /**
   * Edit an image using the image edit API
   */
  async editImage(
    prompt: string,
    imageDataUrl: string,
    modelId?: string,
  ): Promise<void> {
    if (!prompt.trim() || !imageDataUrl || this.isLoading) return;

    if (!this.hasStartedChat) {
      this.startChat();
    }

    // Capture the target conversation ID at the start of the request
    const targetConversationId = this.activeConversationId;
    if (!targetConversationId) return;

    this.isLoading = true;
    void this.reportUpdateActivity("request-start");
    this.currentResponse = "";

    // Add user message directly to the target conversation
    const userMessage: Message = {
      id: generateUUID(),
      role: "user",
      content: prompt,
      timestamp: Date.now(),
      requestType: "image-editing",
      sourceImageDataUrl: imageDataUrl,
    };

    const targetConversation = this.conversations.find(
      (c) => c.id === targetConversationId,
    );
    if (!targetConversation) {
      this.isLoading = false;
      void this.reportUpdateActivity("request-end");
      return;
    }
    targetConversation.messages.push(userMessage);

    // Create placeholder for assistant message directly in target conversation
    const assistantMessage = this.addMessageToConversation(
      targetConversationId,
      "assistant",
      tr("Editing image..."),
    );
    if (!assistantMessage) {
      this.isLoading = false;
      void this.reportUpdateActivity("request-end");
      return;
    }

    // Sync to this.messages if viewing the target conversation
    this.syncActiveMessagesIfNeeded(targetConversationId);
    this.saveConversationsToStorage();

    // Clear editing state
    this.editingImage = null;

    const abortController = new AbortController();
    this.currentAbortController = abortController;

    try {
      // Determine the model to use
      const model = this.getModelForRequest(modelId);
      if (!model) {
        throw new Error(
          tr("No model selected. Please select an image generation model."),
        );
      }

      // Convert base64 data URL to blob
      const response = await fetch(imageDataUrl);
      const imageBlob = await response.blob();

      // Build FormData request
      const formData = new FormData();
      formData.append("model", model);
      formData.append("prompt", prompt);
      formData.append("image", imageBlob, "image.png");

      // Add params from image generation params
      const params = this.imageGenerationParams;
      formData.append("quality", params.quality);
      formData.append("size", params.size);
      formData.append("output_format", params.outputFormat);
      formData.append("response_format", "b64_json");
      formData.append("stream", params.stream ? "1" : "0");
      formData.append("partial_images", params.partialImages.toString());
      formData.append("input_fidelity", params.inputFidelity);

      // Advanced params
      const hasAdvancedParams =
        params.seed !== null ||
        params.numInferenceSteps !== null ||
        params.guidance !== null ||
        (params.negativePrompt !== null &&
          params.negativePrompt.trim() !== "") ||
        params.numSyncSteps !== null;

      if (hasAdvancedParams) {
        formData.append(
          "advanced_params",
          JSON.stringify({
            ...(params.seed !== null && { seed: params.seed }),
            ...(params.numInferenceSteps !== null && {
              num_inference_steps: params.numInferenceSteps,
            }),
            ...(params.guidance !== null && { guidance: params.guidance }),
            ...(params.negativePrompt !== null &&
              params.negativePrompt.trim() !== "" && {
                negative_prompt: params.negativePrompt,
              }),
            ...(params.numSyncSteps !== null && {
              num_sync_steps: params.numSyncSteps,
            }),
          }),
        );
      }

      const apiResponse = await fetch("/v1/images/edits", {
        method: "POST",
        body: formData,
        signal: abortController.signal,
      });

      if (!apiResponse.ok) {
        const errorText = await apiResponse.text();
        throw new Error(
          trf("API error: {status} - {error}", {
            status: apiResponse.status,
            error: errorText,
          }),
        );
      }

      // Streaming requires both stream=true AND partialImages > 0
      const isStreaming = params.stream && params.partialImages > 0;

      if (!isStreaming) {
        // Non-streaming: parse JSON response directly
        const jsonResponse = (await apiResponse.json()) as ImageApiResponse;
        const format = params.outputFormat || "png";
        const mimeType = `image/${format}`;
        const attachments: MessageAttachment[] = jsonResponse.data
          .filter((img) => img.b64_json)
          .map((img) => ({
            type: "generated-image" as const,
            name: `edited-image.${format}`,
            preview: `data:${mimeType};base64,${img.b64_json}`,
            mimeType,
          }));

        this.updateConversationMessage(
          targetConversationId,
          assistantMessage.id,
          (msg) => {
            msg.content = "";
            msg.attachments = attachments;
          },
        );
        this.syncActiveMessagesIfNeeded(targetConversationId);
      } else {
        // Streaming mode: use SSE parser
        const reader = apiResponse.body?.getReader();
        if (!reader) {
          throw new Error(tr("No response body"));
        }

        interface ImageEditChunk {
          data?: { b64_json?: string };
          format?: string;
          type?: "partial" | "final";
          partial_index?: number;
          total_partials?: number;
        }

        await this.parseSSEStream<ImageEditChunk>(
          reader,
          targetConversationId,
          (parsed) => {
            const imageData = parsed.data?.b64_json;

            if (imageData) {
              const format = parsed.format || "png";
              const mimeType = `image/${format}`;
              if (parsed.type === "partial") {
                // Update with partial image and progress
                const partialNum = (parsed.partial_index ?? 0) + 1;
                const totalPartials = parsed.total_partials ?? 3;
                this.updateConversationMessage(
                  targetConversationId,
                  assistantMessage.id,
                  (msg) => {
                    msg.content = trf("Editing... {partial}/{partials}", {
                      partial: partialNum,
                      partials: totalPartials,
                    });
                    msg.attachments = [
                      {
                        type: "generated-image",
                        name: `edited-image.${format}`,
                        preview: `data:${mimeType};base64,${imageData}`,
                        mimeType,
                      },
                    ];
                  },
                );
              } else if (parsed.type === "final") {
                // Final image
                this.updateConversationMessage(
                  targetConversationId,
                  assistantMessage.id,
                  (msg) => {
                    msg.content = "";
                    msg.attachments = [
                      {
                        type: "generated-image",
                        name: `edited-image.${format}`,
                        preview: `data:${mimeType};base64,${imageData}`,
                        mimeType,
                      },
                    ];
                  },
                );
              }
              this.syncActiveMessagesIfNeeded(targetConversationId);
            }
          },
        );
      }
    } catch (error) {
      if (abortController.signal.aborted) {
        this.updateConversationMessage(
          targetConversationId,
          assistantMessage.id,
          (msg) => {
            msg.content = tr("Cancelled");
            msg.attachments = [];
          },
        );
        this.syncActiveMessagesIfNeeded(targetConversationId);
      } else {
        console.error("Error editing image:", error);
        this.handleStreamingError(
          error,
          targetConversationId,
          assistantMessage.id,
          tr("Failed to edit image"),
        );
      }
    } finally {
      this.currentAbortController = null;
      this.isLoading = false;
      void this.reportUpdateActivity("request-end");
      this.saveConversationsToStorage();
    }
  }

  /**
   * Clear current chat and go back to welcome state
   */
  clearChat() {
    this.activeConversationId = null;
    this.messages = [];
    this.hasStartedChat = true;
    this.isTopologyMinimized = true;
    this.currentResponse = "";
    // Clear performance stats
    this.ttftMs = null;
    this.tps = null;
  }

  /**
   * Get the active conversation
   */
  getActiveConversation(): Conversation | null {
    if (!this.activeConversationId) return null;
    return (
      this.conversations.find((c) => c.id === this.activeConversationId) || null
    );
  }

  /**
   * Update the thinking preference for the active conversation
   */
  setConversationThinking(enabled: boolean) {
    this.thinkingEnabled = enabled;
    const conv = this.getActiveConversation();
    if (conv) {
      conv.enableThinking = enabled;
      this.saveConversationsToStorage();
    }
  }

  /**
   * Start a download on a specific node
   */
  async startDownload(nodeId: string, shardMetadata: object): Promise<void> {
    try {
      const response = await fetch("/download/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          targetNodeId: nodeId,
          shardMetadata: shardMetadata,
        }),
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
          `Failed to start download: ${response.status} - ${errorText}`,
        );
      }
    } catch (error) {
      console.error("Error starting download:", error);
      throw error;
    }
  }

  /**
   * Cancel/pause an active download on a specific node
   */
  async cancelDownload(nodeId: string, modelId: string): Promise<void> {
    try {
      const response = await fetch("/download/cancel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          targetNodeId: nodeId,
          modelId: modelId,
        }),
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
          `Failed to cancel download: ${response.status} - ${errorText}`,
        );
      }
    } catch (error) {
      console.error("Error cancelling download:", error);
      throw error;
    }
  }

  /**
   * Delete a downloaded model from a specific node
   */
  async deleteDownload(nodeId: string, modelId: string): Promise<void> {
    try {
      const response = await fetch(
        `/download/${encodeURIComponent(nodeId)}/${encodeURIComponent(modelId)}`,
        {
          method: "DELETE",
        },
      );
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
          `Failed to delete download: ${response.status} - ${errorText}`,
        );
      }
    } catch (error) {
      console.error("Error deleting download:", error);
      throw error;
    }
  }

  /**
   * List all available traces
   */
  async listTraces(): Promise<TraceListResponse> {
    const response = await fetch("/v1/traces");
    if (!response.ok) {
      throw new Error(`Failed to list traces: ${response.status}`);
    }
    return (await response.json()) as TraceListResponse;
  }

  /**
   * Check if a trace exists for a given task ID
   */
  async checkTraceExists(taskId: string): Promise<boolean> {
    try {
      const response = await fetch(`/v1/traces/${encodeURIComponent(taskId)}`);
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Get computed statistics for a task's trace
   */
  async fetchTraceStats(taskId: string): Promise<TraceStatsResponse> {
    const response = await fetch(
      `/v1/traces/${encodeURIComponent(taskId)}/stats`,
    );
    if (!response.ok) {
      throw new Error(`Failed to fetch trace stats: ${response.status}`);
    }
    return (await response.json()) as TraceStatsResponse;
  }

  /**
   * Delete traces by task IDs
   */
  async deleteTraces(
    taskIds: string[],
  ): Promise<{ deleted: string[]; notFound: string[] }> {
    const response = await fetch("/v1/traces/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ taskIds }),
    });
    if (!response.ok) {
      throw new Error(`Failed to delete traces: ${response.status}`);
    }
    return await response.json();
  }

  /**
   * Get the URL for the raw trace file (for Perfetto)
   */
  getTraceRawUrl(taskId: string): string {
    return `/v1/traces/${encodeURIComponent(taskId)}/raw`;
  }
}

export const appStore = new AppStore();

// Reactive exports
export const hasStartedChat = () => appStore.hasStartedChat;
export const messages = () => appStore.messages;
export const currentResponse = () => appStore.currentResponse;
export const isLoading = () => appStore.isLoading;
export const ttftMs = () => appStore.ttftMs;
export const tps = () => appStore.tps;
export const totalTokens = () => appStore.totalTokens;
export const prefillProgress = () => appStore.prefillProgress;
export const topologyData = () => appStore.topologyData;
export const instances = () => appStore.instances;
export const runners = () => appStore.runners;
export const downloads = () => appStore.downloads;
export const nodeDisk = () => appStore.nodeDisk;
export const placementPreviews = () => appStore.placementPreviews;
export const selectedPreviewModelId = () => appStore.selectedPreviewModelId;
export const isLoadingPreviews = () => appStore.isLoadingPreviews;
export const lastUpdate = () => appStore.lastUpdate;
export const isTopologyMinimized = () => appStore.isTopologyMinimized;
export const selectedChatModel = () => appStore.selectedChatModel;
export const thinkingEnabled = () => appStore.thinkingEnabled;
export const debugMode = () => appStore.getDebugMode();
export const chatSidebarVisible = () => appStore.getChatSidebarVisible();

// Actions
export const stopGeneration = () => appStore.stopGeneration();
export const startChat = () => appStore.startChat();
export const sendMessage = (
  content: string,
  files?: {
    id: string;
    name: string;
    type: string;
    textContent?: string;
    preview?: string;
    pageImages?: string[];
  }[],
  enableThinking?: boolean | null,
) => appStore.sendMessage(content, files, enableThinking);
export const generateImage = (prompt: string, modelId?: string) =>
  appStore.generateImage(prompt, modelId);
export const editImage = (
  prompt: string,
  imageDataUrl: string,
  modelId?: string,
) => appStore.editImage(prompt, imageDataUrl, modelId);
export const editingImage = () => appStore.editingImage;
export const setEditingImage = (imageDataUrl: string, sourceMessage: Message) =>
  appStore.setEditingImage(imageDataUrl, sourceMessage);
export const clearEditingImage = () => appStore.clearEditingImage();
export const clearChat = () => appStore.clearChat();
export const setSelectedChatModel = (modelId: string) =>
  appStore.setSelectedModel(modelId);
export const selectPreviewModel = (modelId: string | null) =>
  appStore.selectPreviewModel(modelId);
export const deleteMessage = (messageId: string) =>
  appStore.deleteMessage(messageId);
export const editMessage = (messageId: string, newContent: string) =>
  appStore.editMessage(messageId, newContent);
export const editAndRegenerate = (messageId: string, newContent: string) =>
  appStore.editAndRegenerate(messageId, newContent);
export const regenerateLastResponse = () => appStore.regenerateLastResponse();
export const regenerateFromToken = (messageId: string, tokenIndex: number) =>
  appStore.regenerateFromToken(messageId, tokenIndex);

// Conversation actions
export const conversations = () => appStore.conversations;
export const activeConversationId = () => appStore.activeConversationId;
export const createConversation = (name?: string) =>
  appStore.createConversation(name);
export const loadConversation = (id: string) => appStore.loadConversation(id);
export const deleteConversation = (id: string) =>
  appStore.deleteConversation(id);
export const deleteAllConversations = () => appStore.deleteAllConversations();
export const renameConversation = (id: string, name: string) =>
  appStore.renameConversation(id, name);
export const getActiveConversation = () => appStore.getActiveConversation();
export const setConversationThinking = (enabled: boolean) =>
  appStore.setConversationThinking(enabled);

// Sidebar actions
export const isSidebarOpen = () => appStore.isSidebarOpen;
export const toggleSidebar = () => appStore.toggleSidebar();
export const toggleDebugMode = () => appStore.toggleDebugMode();
export const setDebugMode = (enabled: boolean) =>
  appStore.setDebugMode(enabled);
export const toggleChatSidebarVisible = () =>
  appStore.toggleChatSidebarVisible();
export const setChatSidebarVisible = (visible: boolean) =>
  appStore.setChatSidebarVisible(visible);

// Mobile sidebar state
export const mobileChatSidebarOpen = () => appStore.mobileChatSidebarOpen;
export const toggleMobileChatSidebar = () => appStore.toggleMobileChatSidebar();
export const setMobileChatSidebarOpen = (open: boolean) =>
  appStore.setMobileChatSidebarOpen(open);
export const mobileRightSidebarOpen = () => appStore.mobileRightSidebarOpen;
export const toggleMobileRightSidebar = () =>
  appStore.toggleMobileRightSidebar();
export const setMobileRightSidebarOpen = (open: boolean) =>
  appStore.setMobileRightSidebarOpen(open);

export const refreshState = () => appStore.fetchState();

// Connection status
export const isConnected = () => appStore.isConnected;

// Node identities (for OS version mismatch detection)
export const nodeIdentities = () => appStore.nodeIdentities;

// Thunderbolt & RDMA status
export const nodeThunderbolt = () => appStore.nodeThunderbolt;
export const nodeRdmaCtl = () => appStore.nodeRdmaCtl;
export const thunderboltBridgeCycles = () => appStore.thunderboltBridgeCycles;
export const nodeThunderboltBridge = () => appStore.nodeThunderboltBridge;
export const caiSummary = () => appStore.caiSummary;
export const networkSummary = () => appStore.networkSummary;
export const refreshCaiSummary = () => appStore.fetchCaiSummary();
export const cancelCaiUpdate = () => appStore.cancelCaiUpdate();
export const checkCaiUpdate = () => appStore.checkCaiUpdate();
export const applyCaiUpdate = () => appStore.applyCaiUpdate();

// Image generation params
export const imageGenerationParams = () => appStore.getImageGenerationParams();
export const setImageGenerationParams = (
  params: Partial<ImageGenerationParams>,
) => appStore.setImageGenerationParams(params);
export const resetImageGenerationParams = () =>
  appStore.resetImageGenerationParams();

// Download actions
export const startDownload = (nodeId: string, shardMetadata: object) =>
  appStore.startDownload(nodeId, shardMetadata);
export const cancelDownload = (nodeId: string, modelId: string) =>
  appStore.cancelDownload(nodeId, modelId);
export const deleteDownload = (nodeId: string, modelId: string) =>
  appStore.deleteDownload(nodeId, modelId);

// Trace actions
export const listTraces = () => appStore.listTraces();
export const checkTraceExists = (taskId: string) =>
  appStore.checkTraceExists(taskId);
export const fetchTraceStats = (taskId: string) =>
  appStore.fetchTraceStats(taskId);
export const getTraceRawUrl = (taskId: string) =>
  appStore.getTraceRawUrl(taskId);
export const deleteTraces = (taskIds: string[]) =>
  appStore.deleteTraces(taskIds);
