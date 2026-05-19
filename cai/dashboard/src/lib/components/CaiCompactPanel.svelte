<!--
SPDX-FileCopyrightText: 2025 cai Technologies Ltd
SPDX-FileCopyrightText: 2026 CAI contributors
SPDX-License-Identifier: Apache-2.0
-->
<script lang="ts">
  import { refreshCaiSummary, type CaiSummary } from "$lib/stores/app.svelte";
  import { t } from "$lib/stores/i18n.svelte";
  import CaiCoinIcon from "./CaiCoinIcon.svelte";

  type PanelTab = "wallet" | "history";
  type AuthMode = "open" | "create" | "restore";

  interface Props {
    summary: CaiSummary | null;
    requestedAuthMode?: AuthMode | null;
    walletFocusRequestId?: number;
  }

  interface JournalHistoryItem extends Record<string, unknown> {
    entryId?: string | null;
    source?: string | null;
    eventType?: string | null;
    createdAt?: string | null;
    counterpartyAddress?: string | null;
    amountCoins?: string | null;
    balanceAfterCoins?: string | null;
    blockHeight?: number | null;
    note?: string | null;
  }

  interface ChainRouteAudit {
    directSocketLinkCount?: number | null;
    directBidirectionalLinkCount?: number | null;
    overlayLinkCount?: number | null;
    relayHopsUsed?: boolean | null;
    checkedRelayRouteCount?: number | null;
    activeRelayTransitNodeIds?: string[];
  }

  interface ChainProofAudit {
    executed?: boolean | null;
    verified?: boolean | null;
    error?: string | null;
    stageCount?: number | null;
    processedStageCount?: number | null;
  }

  interface ChainHistoryAudit {
    requesterNodeId?: string | null;
    coordinatorNodeId?: string | null;
    executorNodeIds?: string[];
    executorCount?: number | null;
    route?: ChainRouteAudit | null;
    proof?: ChainProofAudit | null;
  }

  interface SafetyWarning {
    code?: string | null;
    severity?: string | null;
    title?: string | null;
    message?: string | null;
  }

  const hiddenSafetyWarningCodes = new Set([
    "mainnet_alpha",
    "single_validator_guarded_alpha",
  ]);

  let {
    summary,
    requestedAuthMode = null,
    walletFocusRequestId = 0,
  }: Props = $props();

  let activeTab = $state<PanelTab>("wallet");
  let authMode = $state<AuthMode>("open");
  let selectedWallet = $state("");
  let unlockPassword = $state("");
  let createName = $state("");
  let createPassword = $state("");
  let restoreName = $state("");
  let restorePassword = $state("");
  let restoreSeedPhrase = $state("");
  let sendTo = $state("");
  let sendAmount = $state("");
  let transactionOpen = $state(false);
  let nodeModesOpen = $state(false);
  let actionBusy = $state(false);
  let actionMessage = $state("");
  let actionError = $state("");
  let createdSeedPhrase = $state("");
  let journalItems = $state<Array<JournalHistoryItem>>([]);
  let historyLoading = $state<Record<string, boolean>>({
    journal: false,
  });
  let historyHasMore = $state<Record<string, boolean>>({
    journal: true,
  });
  let historyOwnerKey = "";
  let lastWalletFocusRequestId = $state(0);
  let historyCache: {
    journal: Array<JournalHistoryItem>;
  } = {
    journal: [],
  };

  const walletOptions = $derived(summary?.wallets ?? []);
  const currencyCode = $derived(
    String(summary?.currency?.code || "CAICN").trim() || "CAICN",
  );
  const workerRuntimeQueue = $derived(
    summary?.worker?.runtimeQueue ?? summary?.worker?.runtime_queue ?? null,
  );
  const safetyWarnings = $derived(
    ((summary?.safety?.warnings ?? []) as SafetyWarning[]).filter(
      (warning) =>
        (warning?.code || warning?.title || warning?.message) &&
        !hiddenSafetyWarningCodes.has(String(warning?.code ?? "").trim()),
    ),
  );
  const activeWalletSelector = $derived(
    walletOptions.find((wallet) => wallet.active)?.selector ?? "",
  );
  const hasWallets = $derived(walletOptions.length > 0);
  const isWalletLoggedIn = $derived(
    Boolean(summary?.wallet?.has_active_wallet && summary?.wallet?.unlocked),
  );

  $effect(() => {
    if (!selectedWallet) {
      if (activeWalletSelector) {
        selectedWallet = activeWalletSelector;
      } else if (walletOptions.length === 1) {
        selectedWallet = walletOptions[0].selector;
      }
    }
  });

  $effect(() => {
    if (!hasWallets && authMode === "open") {
      authMode = "create";
    }
    if (!isWalletLoggedIn && activeTab === "history") {
      activeTab = "wallet";
    }
  });

  $effect(() => {
    if (!walletFocusRequestId) return;
    if (walletFocusRequestId === lastWalletFocusRequestId) return;
    lastWalletFocusRequestId = walletFocusRequestId;
    activeTab = "wallet";
    if (requestedAuthMode === "restore") {
      authMode = "restore";
      return;
    }
    if (requestedAuthMode === "open" && hasWallets) {
      authMode = "open";
      return;
    }
    authMode = hasWallets ? "open" : "create";
  });

  $effect(() => {
    const ownerKey = activeWalletSelector || "__no_wallet__";
    const freshJournal = [...(summary?.history?.journal ?? [])] as Array<JournalHistoryItem>;

    if (ownerKey !== historyOwnerKey) {
      historyOwnerKey = ownerKey;
      historyCache = {
        journal: freshJournal,
      };
      journalItems = freshJournal;
      historyHasMore = {
        journal: freshJournal.length > 0,
      };
      return;
    }

    const mergedJournal = mergeUnique(
      historyCache.journal,
      freshJournal,
      (item) => String(item.entryId ?? ""),
    );
    historyCache = {
      journal: mergedJournal,
    };
    journalItems = mergedJournal;
  });

  function shortText(text: string | null | undefined, limit = 56): string {
    if (!text) return "-";
    return text.length > limit ? `${text.slice(0, limit - 3)}...` : text;
  }

  function shortWhen(value: string | null | undefined): string {
    if (!value) return "-";
    const match = value.match(/T(\d{2}:\d{2})/);
    return match ? match[1] : shortText(value, 16);
  }

  function friendlyEventName(value: string | null | undefined): string {
    if (!value) return t("history.event");
    return value.replaceAll("_", " ");
  }

  function formatCurrencyAmount(value: string | null | undefined): string {
    const amount = value && value.trim() ? value : "0.00000000";
    return `${amount} ${currencyCode}`;
  }

  function routeModeLabel(value: string | null | undefined): string {
    if (value === "multi_worker_direct") return t("route.multiWorkerDirect");
    if (value === "multi_worker_relay") return t("route.multiWorkerRelay");
    if (value === "single_worker") return t("route.singleWorker");
    if (value === "multi_worker_disconnected") {
      return t("route.multiWorkerDisconnected");
    }
    if (value === "multi_worker_overlay_only") {
      return t("route.multiWorkerOverlayOnly");
    }
    return t("route.unknown");
  }

  function routeModeClass(
    mode: string | null | undefined,
    bottleneck: boolean | null | undefined,
  ): string {
    if (bottleneck) return "border-amber-300/30 bg-amber-400/10 text-amber-200";
    if (mode === "multi_worker_direct") return "border-cyan-300/30 bg-cyan-400/10 text-cyan-200";
    if (mode === "multi_worker_relay") return "border-sky-300/30 bg-sky-400/10 text-sky-200";
    if (mode === "single_worker") return "border-white/15 bg-white/5 text-white/60";
    if (mode === "multi_worker_disconnected" || mode === "multi_worker_overlay_only") {
      return "border-red-400/30 bg-red-500/10 text-red-200";
    }
    return "border-cai-medium-gray/30 bg-cai-black/30 text-white/55";
  }

  function compactBool(value: boolean | null | undefined): string {
    if (value === true) return t("value.yes");
    if (value === false) return t("value.no");
    return t("value.unknown");
  }

  function compactNumber(value: number | null | undefined): string {
    return typeof value === "number" && Number.isFinite(value) ? String(value) : "?";
  }

  function compactNodeList(nodeIds: string[] | null | undefined, limit = 2): string {
    if (!Array.isArray(nodeIds) || nodeIds.length === 0) return "-";
    const shown = nodeIds.slice(0, limit).map((nodeId) => shortText(nodeId, 10));
    const extra = nodeIds.length > limit ? `+${nodeIds.length - limit}` : "";
    return [...shown, extra].filter(Boolean).join(",");
  }

  function chainAuditLine(audit: ChainHistoryAudit | null | undefined): string {
    const chain = t("history.chain");
    if (!audit) return `${chain} -`;
    const requester = shortText(audit.requesterNodeId, 10);
    const coordinator = shortText(audit.coordinatorNodeId, 10);
    const executors = compactNodeList(audit.executorNodeIds);
    if (coordinator !== "-" && coordinator !== requester) {
      return `${chain} ${requester} > ${coordinator} > ${executors}`;
    }
    return `${chain} ${requester} > ${executors}`;
  }

  function routeAuditLine(route: ChainRouteAudit | null | undefined): string {
    if (!route) return `${t("history.route")} -`;
    const direct = compactNumber(route.directSocketLinkCount);
    const bidi = compactNumber(route.directBidirectionalLinkCount);
    const relay = route.relayHopsUsed ? t("value.yes") : t("value.no");
    const relayRoutes = compactNumber(route.checkedRelayRouteCount);
    return `${t("history.direct")} ${direct}/${bidi} ${t("history.relay")} ${relay}/${relayRoutes}`;
  }

  function proofStatusLabel(proof: ChainProofAudit | null | undefined): string {
    if (!proof) return t("proof.unknown");
    if (proof.error) return t("proof.error");
    if (proof.executed && proof.verified) return t("proof.verified");
    if (proof.executed) return t("proof.unverified");
    if (proof.executed === false) return t("proof.missing");
    return t("proof.unknown");
  }

  function proofStatusClass(proof: ChainProofAudit | null | undefined): string {
    if (proof?.error) return "border-red-300/30 bg-red-500/10 text-red-200";
    if (proof?.executed && proof?.verified) {
      return "border-emerald-300/25 bg-emerald-400/10 text-emerald-200";
    }
    if (proof?.executed) return "border-amber-300/30 bg-amber-400/10 text-amber-200";
    return "border-cai-medium-gray/25 bg-cai-black/30 text-white/45";
  }

  function attemptStatusLine(job: Record<string, unknown>): string {
    const status = job.executionAttemptStatus as
      | {
          attempt?: number | null;
          maxAttempts?: number | null;
          status?: string | null;
          phase?: string | null;
          message?: string | null;
        }
      | null
      | undefined;
    const count =
      typeof job.executionAttemptCount === "number"
        ? job.executionAttemptCount
        : Array.isArray(job.executionAttempts)
          ? job.executionAttempts.length
          : 0;
    if (!status && count <= 1) return "";
    const attempt = status?.attempt ?? count;
    const maxAttempts = status?.maxAttempts ?? Math.max(count, attempt || 1);
    const label = shortText(status?.status || "attempt", 14);
    const phase = status?.phase
      ? ` ${shortText(String(status.phase).replaceAll("_", " "), 22)}`
      : "";
    const reason = status?.message ? ` ${shortText(status.message, 42)}` : "";
    return `${t("history.attempt")} ${attempt}/${maxAttempts} ${label}${phase}${reason}`;
  }

  function attemptStatusClass(job: Record<string, unknown>): string {
    const status = job.executionAttemptStatus as { status?: string | null } | null;
    const value = String(status?.status ?? "").trim();
    if (value === "retrying") return "border-amber-300/25 bg-amber-400/10 text-amber-200";
    if (value === "failed") return "border-red-300/25 bg-red-500/10 text-red-200";
    return "border-cyan-300/25 bg-cyan-400/10 text-cyan-100";
  }

  function workerQueueLine(): string {
    const queue = workerRuntimeQueue;
    if (!queue) return `${t("worker.queue")} -`;
    const received = compactNumber(queue.receivedCount);
    const processing = compactNumber(queue.processingCount);
    const processed = compactNumber(queue.processedCount);
    const failed = compactNumber((queue.failedCount ?? 0) + (queue.timedOutCount ?? 0));
    const current = queue.currentBatch?.batch?.batchId;
    const tail = current ? ` | ${shortText(current, 18)}` : queue.lastError ? ` | ${shortText(queue.lastError, 28)}` : "";
    return `${t("worker.queue")} r${received} p${processing} ok${processed} ${t("worker.fail")}${failed}${tail}`;
  }

  function mergeUnique<T extends Record<string, unknown>>(
    current: Array<T>,
    incoming: Array<T>,
    getKey: (item: T) => string,
  ): Array<T> {
    if (incoming.length === 0) {
      return current;
    }
    const merged = [...incoming, ...current];
    const seen = new Set<string>();
    const result: Array<T> = [];
    for (const item of merged) {
      const key = getKey(item);
      if (!key || seen.has(key)) {
        continue;
      }
      seen.add(key);
      result.push(item);
    }
    return result;
  }

  function handleHistoryScroll(event: Event): void {
    const target = event.currentTarget;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    const remaining = target.scrollHeight - target.scrollTop - target.clientHeight;
    if (remaining <= 56) {
      void loadMoreHistory();
    }
  }

  async function requestJson(
    url: string,
    payload: Record<string, unknown> = {},
  ): Promise<Record<string, unknown> | null> {
    actionBusy = true;
    actionError = "";
    actionMessage = "";
    createdSeedPhrase = "";
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
      const data = (await response.json().catch(() => ({}))) as Record<string, unknown>;
      if (!response.ok) {
        throw new Error(
          String(data?.detail || data?.error || `${t("error.requestFailed")}: ${response.status}`),
        );
      }
      actionMessage = typeof data?.message === "string" ? data.message : t("action.done");
      if (typeof data?.seed_phrase === "string") {
        createdSeedPhrase = data.seed_phrase;
      }
      await refreshCaiSummary();
      return data;
    } catch (error) {
      actionError = error instanceof Error ? error.message : String(error);
      return null;
    } finally {
      actionBusy = false;
    }
  }

  async function handleCreateWallet(): Promise<void> {
    const result = await requestJson("/v1/cai/wallet/create", {
      name: createName,
      password: createPassword,
    });
    if (!result) {
      return;
    }
    createName = "";
    createPassword = "";
    activeTab = "wallet";
  }

  async function handleRestoreWallet(): Promise<void> {
    const result = await requestJson("/v1/cai/wallet/restore", {
      name: restoreName,
      password: restorePassword,
      seed_phrase: restoreSeedPhrase,
    });
    if (!result) {
      return;
    }
    restoreName = "";
    restorePassword = "";
    restoreSeedPhrase = "";
    activeTab = "wallet";
  }

  async function handleOpenExistingWallet(): Promise<void> {
    const opened = await requestJson("/v1/cai/wallet/select", {
      selector: selectedWallet,
    });
    if (!opened) {
      return;
    }
    const unlocked = await requestJson("/v1/cai/wallet/unlock", {
      password: unlockPassword,
      wallet: selectedWallet,
    });
    if (!unlocked) {
      return;
    }
    unlockPassword = "";
    activeTab = "wallet";
  }

  async function handleLockWallet(): Promise<void> {
    await requestJson("/v1/cai/wallet/lock", {});
  }

  async function handleLogoutWallet(): Promise<void> {
    const loggedOut = await requestJson("/v1/cai/wallet/logout", {});
    if (!loggedOut) {
      return;
    }
    selectedWallet = "";
    unlockPassword = "";
    authMode = hasWallets ? "open" : "create";
    activeTab = "wallet";
  }

  async function handleSend(): Promise<void> {
    const sent = await requestJson("/v1/cai/wallet/send", {
      to: sendTo,
      amount: sendAmount,
    });
    if (!sent) {
      return;
    }
    sendTo = "";
    sendAmount = "";
  }

  async function handleToggleValidator(enabled: boolean): Promise<void> {
    await requestJson("/v1/cai/node/validator", { enabled });
  }

  async function handleConfirmValidatorStaticIp(confirmed: boolean): Promise<void> {
    await requestJson("/v1/cai/node/validator/static-ip", { confirmed });
  }

  async function handleToggleWorker(enabled: boolean): Promise<void> {
    await requestJson("/v1/cai/node/worker", { enabled });
  }

  async function handleCompleteValidatorUnbond(): Promise<void> {
    await requestJson("/v1/cai/node/validator/unbond-complete", {});
  }

  async function handleClearValidatorJail(): Promise<void> {
    await requestJson("/v1/cai/node/validator/unjail", {});
  }

  function validatorState(): string {
    return (
      summary?.validator?.validator_state ??
      (summary?.validator?.validator_enabled ? "bonded" : "unbonded")
    );
  }

  function isActionTimeReady(value: string | null | undefined): boolean {
    if (!value) return false;
    const timestamp = Date.parse(value);
    if (Number.isNaN(timestamp)) return false;
    return timestamp <= Date.now();
  }

  function validatorToneClass(): string {
    const state = validatorState();
    if (state === "bonded") return "bg-green-500/15 text-green-400";
    if (state === "unbonding") return "bg-amber-500/15 text-amber-300";
    if (state === "jailed") return "bg-red-500/15 text-red-300";
    return "bg-white/8 text-white/50";
  }

  function validatorStateLabel(): string {
    const state = validatorState();
    if (state === "bonded") return t("validator.stateBonded");
    if (state === "unbonding") return t("validator.stateUnbonding");
    if (state === "jailed") return t("validator.stateJailed");
    return t("validator.stateUnbonded");
  }

  function validatorSubtext(): string {
    const state = validatorState();
    if (state === "bonded") {
      return summary?.validator?.validator_can_attest
        ? t("validator.ready")
        : (summary?.validator?.validator_attestation_note ?? t("validator.bonded"));
    }
    if (state === "unbonding") {
      const availableAt = summary?.validator?.validator_unbonding_available_at;
      return availableAt
        ? `${t("validator.unbondUnlock")} ${shortWhen(availableAt)}`
        : t("validator.bondPending");
    }
    if (state === "jailed") {
      const availableAt = summary?.validator?.validator_unjail_available_at;
      return availableAt
        ? `${t("validator.cooldownUntil")} ${shortWhen(availableAt)}`
        : t("validator.jailed");
    }
    return "";
  }

  function validatorPenaltySummary(): string {
    const validator = summary?.validator;
    if (!validator) return "";
    const pendingAttestation = Number(validator.penalty_case_pending_attestation_count ?? 0);
    const pending = Number(validator.penalty_case_pending_count ?? 0);
    const applied = Number(validator.penalty_case_applied_count ?? 0);
    const finalized = Number(validator.penalty_case_finalized_count ?? 0);
    const total = Number(validator.penalty_case_count ?? 0);

    if (pendingAttestation > 0) {
      return `${pendingAttestation} ${t("validator.penaltyCasesWaitingCommittee")}`;
    }
    if (total <= 0) {
      return "";
    }
    return `${t("validator.penaltyCases")} ${applied}/${total} ${t("validator.applied")}${finalized > 0 ? `, ${finalized} ${t("validator.finalized")}` : ""}${pending > 0 ? `, ${pending} ${t("validator.pending")}` : ""}`;
  }

  function validatorActionLabel(): string {
    const state = validatorState();
    if (state === "bonded") return t("action.disable");
    if (state === "unbonding") return t("action.complete");
    if (state === "jailed") return t("action.unjail");
    return t("action.enable");
  }

  function validatorActionDisabled(): boolean {
    const state = validatorState();
    if (actionBusy) return true;
    if (state === "bonded") return false;
    if (state === "unbonding") {
      return !isActionTimeReady(summary?.validator?.validator_unbonding_available_at);
    }
    if (state === "jailed") {
      return !isActionTimeReady(summary?.validator?.validator_unjail_available_at);
    }
    return summary?.validator?.validator_can_enable === false;
  }

  async function handlePrimaryValidatorAction(): Promise<void> {
    const state = validatorState();
    if (state === "bonded") {
      await handleToggleValidator(false);
      return;
    }
    if (state === "unbonding") {
      await handleCompleteValidatorUnbond();
      return;
    }
    if (state === "jailed") {
      await handleClearValidatorJail();
      return;
    }
    await handleToggleValidator(true);
  }

  function workerToggleBlocked(): boolean {
    const state = validatorState();
    return state === "bonded" || state === "unbonding";
  }

  async function loadMoreHistory(): Promise<void> {
    if (historyLoading.journal || !historyHasMore.journal) {
      return;
    }
    historyLoading = { ...historyLoading, journal: true };
    actionError = "";
    try {
      const offset = journalItems.length;
      const response = await fetch(
        `/v1/cai/history?section=journal&offset=${offset}&limit=10`,
      );
      const data = (await response.json().catch(() => ({}))) as Record<string, unknown>;
      if (!response.ok) {
        throw new Error(
          String(data?.detail || data?.error || `${t("error.requestFailed")}: ${response.status}`),
        );
      }
      const items = Array.isArray(data?.items) ? (data.items as Array<JournalHistoryItem>) : [];
      const merged = mergeUnique(
        historyCache.journal,
        items,
        (item) => String(item.entryId ?? ""),
      );
      historyCache = { ...historyCache, journal: merged };
      journalItems = merged;
      historyHasMore = { ...historyHasMore, journal: Boolean(data?.hasMore) };
    } catch (error) {
      actionError = error instanceof Error ? error.message : String(error);
    } finally {
      historyLoading = { ...historyLoading, journal: false };
    }
  }

  const walletButtonBase =
    "inline-flex items-center justify-center gap-2 rounded-md border px-2.5 py-1.5 font-mono uppercase tracking-[0.16em] transition-all duration-200 ease-out disabled:opacity-40 disabled:cursor-not-allowed enabled:hover:-translate-y-[1px] enabled:hover:scale-[1.01] enabled:active:translate-y-0 enabled:active:scale-[0.99]";

  function walletButtonClass(
    tone:
      | "neutral"
      | "yellow"
      | "green"
      | "blue"
      | "red"
      | "amber",
    options: {
      full?: boolean;
      compact?: boolean;
    } = {},
  ): string {
    const width = options.full ? "w-full" : "";
    const density = options.compact
      ? "text-[9px] min-h-[30px]"
      : "text-[10px] min-h-[34px]";

    const tones: Record<string, string> = {
      neutral:
        "border-white/10 bg-gradient-to-b from-white/10 to-white/5 text-white/80 enabled:hover:border-white/20 enabled:hover:bg-white/10 enabled:hover:text-white enabled:hover:shadow-[0_10px_24px_rgba(255,255,255,0.08)]",
      yellow:
        "border-cai-yellow/25 bg-gradient-to-b from-cai-yellow/18 to-cai-yellow/8 text-cai-yellow enabled:hover:border-cai-yellow/45 enabled:hover:shadow-[0_10px_28px_rgba(125,211,252,0.18)]",
      green:
        "border-green-400/25 bg-gradient-to-b from-green-400/18 to-green-500/8 text-green-300 enabled:hover:border-green-300/45 enabled:hover:shadow-[0_10px_28px_rgba(74,222,128,0.18)]",
      blue:
        "border-blue-400/25 bg-gradient-to-b from-blue-400/18 to-blue-500/8 text-blue-300 enabled:hover:border-blue-300/45 enabled:hover:shadow-[0_10px_28px_rgba(96,165,250,0.18)]",
      red:
        "border-red-400/25 bg-gradient-to-b from-red-400/18 to-red-500/8 text-red-300 enabled:hover:border-red-300/45 enabled:hover:shadow-[0_10px_28px_rgba(248,113,113,0.18)]",
      amber:
        "border-amber-400/25 bg-gradient-to-b from-amber-400/18 to-amber-500/8 text-amber-300 enabled:hover:border-amber-300/45 enabled:hover:shadow-[0_10px_28px_rgba(251,191,36,0.18)]",
    };

    return [walletButtonBase, density, width, tones[tone]].filter(Boolean).join(" ");
  }

  function walletTabClass(active: boolean, fill = false): string {
    return [
      walletButtonBase,
      fill ? "flex-1 min-h-[34px] text-[10px]" : "min-h-[30px] text-[10px]",
      active
        ? "border-cai-yellow/30 bg-gradient-to-b from-cai-yellow/18 to-cai-yellow/8 text-cai-yellow shadow-[0_8px_24px_rgba(125,211,252,0.12)]"
        : "border-transparent bg-transparent text-white/55 enabled:hover:border-white/10 enabled:hover:bg-white/6 enabled:hover:text-white/85",
    ].join(" ");
  }

  function safetyWarningClass(warning: SafetyWarning): string {
    const critical = String(warning.severity || "").toLowerCase() === "critical";
    return critical
      ? "border-red-400/25 bg-red-500/8 text-red-100"
      : "border-amber-400/25 bg-amber-400/8 text-amber-100";
  }

</script>

<div class="p-4 border-b border-cai-yellow/10 bg-cai-black/10">
  <div class="flex items-center gap-2 mb-3">
    <div
      class={`w-2 h-2 rounded-full shadow-[0_0_8px_currentColor] ${
        summary?.available ? "bg-green-400 text-green-400" : "bg-sky-300 text-sky-300"
      }`}
    ></div>
    <h3 class="text-xs text-cai-yellow font-mono tracking-[0.2em] uppercase">
      {t("wallet.title")}
    </h3>
    <div class="flex-1 h-px bg-gradient-to-r from-cai-yellow/30 to-transparent"></div>
  </div>

  {#if !summary}
    <p class="text-[11px] text-white/50 font-mono">{t("wallet.loading")}</p>
  {:else if !summary.available}
    <div class="space-y-1">
      <div class="text-xs text-sky-200 font-mono tracking-wider">{t("wallet.unavailable")}</div>
      <p class="text-[11px] text-white/50 leading-relaxed break-words">
        {summary.error || t("wallet.endpointNotReady")}
      </p>
    </div>
  {:else}
    <div class="space-y-3">
      {#if safetyWarnings.length > 0}
        <div class="space-y-1">
          {#each safetyWarnings.slice(0, 2) as warning}
            <div class={`rounded border px-2.5 py-2 ${safetyWarningClass(warning)}`}>
              <div class="text-[10px] font-mono uppercase tracking-[0.16em]">
                {warning.title || warning.code || "Network warning"}
              </div>
              {#if warning.message}
                <div class="mt-1 text-[10px] leading-relaxed opacity-80">
                  {warning.message}
                </div>
              {/if}
            </div>
          {/each}
        </div>
      {/if}

      <div class="flex gap-1 border border-cai-medium-gray/30 bg-cai-black/20 p-1 rounded">
        <button
          type="button"
          class={walletTabClass(activeTab === "wallet")}
          onclick={() => (activeTab = "wallet")}
        >
          {t("wallet.title")}
        </button>
        {#if isWalletLoggedIn}
          <button
            type="button"
            class={walletTabClass(activeTab === "history")}
            onclick={() => (activeTab = "history")}
          >
            {t("wallet.tabHistory")}
          </button>
        {/if}
      </div>

      {#if activeTab === "wallet"}
        {#if isWalletLoggedIn}
          <div class="mx-auto w-full max-w-xl space-y-3">
            {#if createdSeedPhrase}
              <div class="border border-cai-yellow/25 bg-cai-yellow/5 p-3 space-y-2">
                <div class="text-[10px] text-cai-yellow font-mono uppercase tracking-wider">
                  {t("wallet.saveSeed")}
                </div>
                <div class="text-[11px] text-white/70 leading-relaxed">
                  {t("wallet.saveSeedHelp")}
                </div>
                <div class="text-[11px] text-white/85 leading-relaxed break-words">
                  {createdSeedPhrase}
                </div>
              </div>
            {/if}

            <div class="border border-cai-medium-gray/30 bg-cai-black/30 p-3 space-y-3">
              <div class="flex items-start justify-between gap-3">
                <div class="min-w-0">
                  <div class="text-[10px] text-white/40 font-mono uppercase tracking-wider">{t("wallet.current")}</div>
                  <div class="text-sm text-cai-light-gray font-mono mt-1 truncate">
                    {summary.wallet?.wallet_name || t("wallet.noWallet")}
                  </div>
                  <div class="mt-2 flex items-center gap-2">
                    <CaiCoinIcon
                      decorative={false}
                      label={currencyCode}
                      class="h-7 w-7 border border-cai-yellow/20 shadow-[0_0_18px_rgba(125,211,252,0.18)]"
                    />
                    <div class="min-w-0">
                      <div class="text-xl text-cai-yellow font-mono leading-none">
                        {summary.wallet?.balance_coins || "0.00000000"}
                      </div>
                      <div class="mt-1 text-[11px] text-cai-yellow/70 font-mono uppercase tracking-[0.14em]">
                        {currencyCode}
                      </div>
                    </div>
                  </div>
                  {#if summary.wallet?.balance_source || summary.wallet?.local_cached_balance_coins}
                    <div class="mt-1 text-[10px] text-white/40 font-mono uppercase tracking-wider">
                      {t("wallet.balanceSource")} {summary.wallet?.balance_source ?? "local"}{summary.wallet?.local_cached_balance_coins ? ` | ${t("wallet.cache")} ${formatCurrencyAmount(summary.wallet.local_cached_balance_coins)}` : ""}
                    </div>
                  {/if}
                </div>
                <span class="px-1.5 py-0.5 text-[10px] font-mono uppercase rounded bg-cai-yellow/15 text-cai-yellow">
                  {t("wallet.unlocked")}
                </span>
              </div>

              {#if summary.wallet?.address}
                <div class="border border-cai-medium-gray/20 bg-cai-black/20 p-2">
                  <div class="text-[10px] text-white/40 font-mono uppercase tracking-wider">{t("wallet.address")}</div>
                  <div class="mt-1 text-[11px] text-white/75 font-mono break-all">
                    {summary.wallet.address}
                  </div>
                </div>
              {/if}

              <div class="flex flex-wrap gap-2">
                <button
                  type="button"
                  class={walletButtonClass("neutral")}
                  onclick={handleLockWallet}
                  disabled={actionBusy}
                >
                  {t("wallet.lock")}
                </button>
                <button
                  type="button"
                  class={walletButtonClass("red")}
                  onclick={handleLogoutWallet}
                  disabled={actionBusy}
                >
                  {t("wallet.exit")}
                </button>
              </div>
            </div>

            <div class="border border-cai-medium-gray/30 bg-cai-black/30 p-3 space-y-2">
              <button
                type="button"
                class="w-full flex items-center justify-between gap-3 text-left"
                onclick={() => (transactionOpen = !transactionOpen)}
              >
                <div class="text-[10px] text-white/40 font-mono uppercase tracking-wider">{t("wallet.transaction")}</div>
                <div class="text-[10px] text-white/45 font-mono uppercase">{transactionOpen ? t("action.hide") : t("action.show")}</div>
              </button>

              {#if transactionOpen}
                <div class="space-y-3 pt-1">
                  <input bind:value={sendTo} placeholder={t("wallet.recipientPlaceholder")} class="w-full bg-cai-black/60 border border-cai-medium-gray/30 rounded px-2 py-1.5 text-xs text-cai-light-gray" />
                  <input bind:value={sendAmount} placeholder={`${t("wallet.amountPlaceholder")} ${currencyCode}`} class="w-full bg-cai-black/60 border border-cai-medium-gray/30 rounded px-2 py-1.5 text-xs text-cai-light-gray" />
                  <button
                    type="button"
                    class={walletButtonClass("blue", { compact: true })}
                    onclick={handleSend}
                    disabled={!sendTo || !sendAmount || actionBusy}
                  >
                    {t("action.send")}
                  </button>
                </div>
              {/if}
            </div>

            <div class="border border-cai-medium-gray/30 bg-cai-black/30 p-3 space-y-2">
              <button
                type="button"
                class="w-full flex items-center justify-between gap-3 text-left"
                onclick={() => (nodeModesOpen = !nodeModesOpen)}
              >
                <div class="text-[10px] text-white/40 font-mono uppercase tracking-wider">{t("node.modes")}</div>
                <div class="text-[10px] text-white/45 font-mono uppercase">{nodeModesOpen ? t("action.hide") : t("action.show")}</div>
              </button>

              {#if nodeModesOpen}
                <div class="space-y-2 pt-1">
                  <div class="border border-cai-medium-gray/20 bg-cai-black/20 p-2 flex items-start gap-2">
                    <div class="min-w-0 flex-1">
                      <div class="flex items-center gap-2 min-w-0">
                        <div class="text-[10px] text-white/40 font-mono uppercase tracking-wider">{t("node.validator")}</div>
                        <span class={`px-1.5 py-0.5 text-[10px] font-mono uppercase rounded ${validatorToneClass()}`}>
                          {validatorStateLabel()}
                        </span>
                      </div>
                      {#if validatorSubtext()}
                        <div class="mt-1 text-[10px] text-white/55 leading-relaxed">
                          {validatorSubtext()}
                        </div>
                      {/if}
                      {#if validatorState() === "jailed" && summary.validator?.validator_jail_reason}
                        <div class="mt-1 text-[10px] text-red-300/75 leading-relaxed">
                          {shortText(summary.validator.validator_jail_reason, 64)}
                        </div>
                      {/if}
                      {#if validatorPenaltySummary()}
                        <div class={`mt-1 text-[10px] leading-relaxed ${
                          (summary.validator?.penalty_case_pending_attestation_count ?? 0) > 0
                            ? "text-amber-300/80"
                            : "text-white/45"
                        }`}>
                          {validatorPenaltySummary()}
                        </div>
                      {/if}
                      {#if validatorState() === "unbonded" && !summary.validator?.validator_static_ip_confirmed}
                        <button
                          type="button"
                          class={`mt-2 ${walletButtonClass("neutral", { compact: true })}`}
                          onclick={() => handleConfirmValidatorStaticIp(true)}
                          disabled={actionBusy}
                        >
                          {t("node.confirmStaticIp")}
                        </button>
                      {/if}
                    </div>
                    <button
                      type="button"
                      class={`min-w-[84px] ${walletButtonClass(
                        validatorState() === "bonded"
                          ? "neutral"
                          : validatorState() === "unbonding"
                            ? "amber"
                            : validatorState() === "jailed"
                              ? "red"
                              : "green",
                        { compact: true },
                      )} ${
                        actionBusy ? "animate-pulse" : ""
                      }`}
                      onclick={handlePrimaryValidatorAction}
                      disabled={validatorActionDisabled()}
                    >
                      {actionBusy ? t("action.wait") : validatorActionLabel()}
                    </button>
                  </div>

                  <div class="border border-cai-medium-gray/20 bg-cai-black/20 p-2 flex items-center gap-2">
                    <div class="min-w-0 flex-1">
                      <div class="flex items-center gap-2 min-w-0">
                        <div class="text-[10px] text-white/40 font-mono uppercase tracking-wider">{t("node.worker")}</div>
                        <span class={`px-1.5 py-0.5 text-[10px] font-mono uppercase rounded ${
                          summary.worker?.worker_enabled
                            ? "bg-blue-500/15 text-blue-300"
                            : "bg-white/8 text-white/50"
                        }`}>
                          {summary.worker?.worker_enabled ? t("status.on") : t("status.off")}
                        </span>
                      </div>
                      {#if workerRuntimeQueue}
                        <div class="mt-1 text-[10px] text-white/45 font-mono truncate">
                          {workerQueueLine()}
                        </div>
                      {/if}
                    </div>
                    <button
                      type="button"
                      class={`min-w-[84px] ${walletButtonClass(
                        summary.worker?.worker_enabled
                          ? "neutral"
                          : "blue",
                        { compact: true },
                      )}`}
                      onclick={() => handleToggleWorker(!summary.worker?.worker_enabled)}
                      disabled={
                        actionBusy ||
                        (!summary.worker?.worker_enabled && workerToggleBlocked())
                      }
                    >
                      {actionBusy ? t("action.wait") : summary.worker?.worker_enabled ? t("action.disable") : t("action.enable")}
                    </button>
                  </div>
                </div>
              {/if}
            </div>
          </div>
        {:else}
          <div class="flex justify-center">
            <div class="w-full max-w-md border border-cai-medium-gray/30 bg-cai-black/30 p-4 space-y-4">
              <div class="text-center space-y-1">
                <div class="text-[10px] text-white/40 font-mono uppercase tracking-[0.2em]">{t("auth.walletAccess")}</div>
                <div class="text-sm text-cai-light-gray">{t("auth.walletAccessHelp")}</div>
              </div>

              <div class="flex gap-1 border border-cai-medium-gray/30 bg-cai-black/20 p-1 rounded">
                {#if hasWallets}
                  <button
                    type="button"
                    class={walletTabClass(authMode === "open", true)}
                    onclick={() => (authMode = "open")}
                  >
                    {t("action.open")}
                  </button>
                {/if}
                <button
                  type="button"
                  class={walletTabClass(authMode === "create", true)}
                  onclick={() => (authMode = "create")}
                >
                  {t("action.create")}
                </button>
                <button
                  type="button"
                  class={walletTabClass(authMode === "restore", true)}
                  onclick={() => (authMode = "restore")}
                >
                  {t("action.restore")}
                </button>
              </div>

              {#if authMode === "open"}
                <div class="space-y-3">
                  {#if hasWallets}
                    <select bind:value={selectedWallet} class="w-full bg-cai-black/60 border border-cai-medium-gray/30 rounded px-2 py-1.5 text-xs text-cai-light-gray">
                      <option value="">{t("auth.selectWallet")}</option>
                      {#each walletOptions as wallet}
                        <option value={wallet.selector}>{wallet.name} - {wallet.address}</option>
                      {/each}
                    </select>
                    <input bind:value={unlockPassword} type="password" placeholder={t("auth.password")} class="w-full bg-cai-black/60 border border-cai-medium-gray/30 rounded px-2 py-1.5 text-xs text-cai-light-gray" />
                    <button
                      type="button"
                      class={walletButtonClass("yellow", { full: true })}
                      onclick={handleOpenExistingWallet}
                      disabled={!selectedWallet || !unlockPassword || actionBusy}
                    >
                      {t("auth.openWallet")}
                    </button>
                  {:else}
                    <div class="text-[11px] text-white/50 leading-relaxed">
                      {t("auth.noLocalWallets")}
                    </div>
                  {/if}
                </div>
              {/if}

              {#if authMode === "create"}
                <div class="space-y-3">
                  <input bind:value={createName} placeholder={t("auth.walletName")} class="w-full bg-cai-black/60 border border-cai-medium-gray/30 rounded px-2 py-1.5 text-xs text-cai-light-gray" />
                  <input bind:value={createPassword} type="password" placeholder={t("auth.password")} class="w-full bg-cai-black/60 border border-cai-medium-gray/30 rounded px-2 py-1.5 text-xs text-cai-light-gray" />
                  <button
                    type="button"
                    class={walletButtonClass("green", { full: true })}
                    onclick={handleCreateWallet}
                    disabled={!createName || !createPassword || actionBusy}
                  >
                    {t("auth.createWallet")}
                  </button>
                  {#if createdSeedPhrase}
                    <div class="border border-cai-yellow/20 bg-cai-black/40 p-2 space-y-1">
                      <div class="text-[10px] text-cai-yellow font-mono uppercase tracking-wider">{t("auth.seedPhrase")}</div>
                      <div class="text-[11px] text-white/80 leading-relaxed break-words">{createdSeedPhrase}</div>
                    </div>
                  {/if}
                </div>
              {/if}

              {#if authMode === "restore"}
                <div class="space-y-3">
                  <input bind:value={restoreName} placeholder={t("auth.walletName")} class="w-full bg-cai-black/60 border border-cai-medium-gray/30 rounded px-2 py-1.5 text-xs text-cai-light-gray" />
                  <input bind:value={restorePassword} type="password" placeholder={t("auth.password")} class="w-full bg-cai-black/60 border border-cai-medium-gray/30 rounded px-2 py-1.5 text-xs text-cai-light-gray" />
                  <textarea bind:value={restoreSeedPhrase} rows="4" placeholder={t("auth.seedPhrase")} class="w-full bg-cai-black/60 border border-cai-medium-gray/30 rounded px-2 py-1.5 text-xs text-cai-light-gray resize-y"></textarea>
                  <button
                    type="button"
                    class={walletButtonClass("yellow", { full: true })}
                    onclick={handleRestoreWallet}
                    disabled={!restoreName || !restorePassword || !restoreSeedPhrase || actionBusy}
                  >
                    {t("auth.restoreWallet")}
                  </button>
                </div>
              {/if}
            </div>
          </div>
        {/if}
      {/if}

      {#if activeTab === "history"}
        <div class="space-y-3">
          <div class="border border-cai-medium-gray/30 bg-cai-black/30 p-2">
            <div class="flex items-center justify-between gap-3 mb-2">
              <div class="text-[10px] text-white/40 font-mono uppercase tracking-wider">{t("history.jobs")}</div>
              <div class="text-[10px] text-white/35 font-mono">{summary?.history?.jobs?.length ?? 0}</div>
            </div>
            {#if !summary?.history?.jobs || summary.history.jobs.length === 0}
              <div class="text-[11px] text-white/45 font-mono">{t("history.noJobs")}</div>
            {:else}
              <div class="space-y-2">
                {#each summary.history.jobs.slice(0, 4) as job}
                  <div class="border border-cai-medium-gray/20 bg-cai-black/20 p-2">
                    <div class="flex items-center justify-between gap-2 text-[10px] font-mono text-white/40 uppercase">
                      <span>{shortText(job.status, 20)}</span>
                      <span>{shortWhen(job.createdAt)}</span>
                    </div>
                    <div class="mt-1 text-[11px] text-cai-light-gray font-mono break-all">
                      {shortText(job.modelId, 48)}
                    </div>
                    <div class="mt-2 flex flex-wrap items-center gap-1.5 text-[10px] font-mono">
                      <span class={`rounded-full border px-2 py-0.5 ${routeModeClass(job.transportMode, job.relayBottleneckRisk)}`}>
                        {routeModeLabel(job.transportMode)}
                      </span>
                      <span class="rounded-full border border-cai-medium-gray/25 bg-cai-black/30 px-2 py-0.5 text-white/45">
                        {t("history.nodes")} {job.participantCount ?? "?"}
                      </span>
                      <span class="rounded-full border border-cai-medium-gray/25 bg-cai-black/30 px-2 py-0.5 text-white/45">
                        {t("history.decentralized")} {compactBool(job.decentralizedExecution)}
                      </span>
                      {#if job.relayBottleneckRisk}
                        <span class="rounded-full border border-amber-300/25 bg-amber-400/10 px-2 py-0.5 text-amber-200">
                          {t("history.bottleneckRisk")}
                        </span>
                      {/if}
                      {#if attemptStatusLine(job)}
                        <span class={`rounded-full border px-2 py-0.5 ${attemptStatusClass(job)}`}>
                          {attemptStatusLine(job)}
                        </span>
                      {/if}
                      {#if job.decentralizedChainAudit}
                        <span class="rounded-full border border-cai-medium-gray/25 bg-cai-black/30 px-2 py-0.5 text-white/50">
                          {chainAuditLine(job.decentralizedChainAudit)}
                        </span>
                        <span class="rounded-full border border-cyan-300/25 bg-cyan-400/10 px-2 py-0.5 text-cyan-100">
                          {t("history.exec")} {compactNumber(job.decentralizedChainAudit.executorCount)}
                        </span>
                        <span class="rounded-full border border-cai-medium-gray/25 bg-cai-black/30 px-2 py-0.5 text-white/45">
                          {routeAuditLine(job.decentralizedChainAudit.route)}
                        </span>
                        <span class={`rounded-full border px-2 py-0.5 ${proofStatusClass(job.decentralizedChainAudit.proof)}`}>
                          {proofStatusLabel(job.decentralizedChainAudit.proof)}
                        </span>
                        <span class="rounded-full border border-cai-medium-gray/25 bg-cai-black/30 px-2 py-0.5 text-white/45">
                          {t("history.stages")} {compactNumber(job.decentralizedChainAudit.proof?.processedStageCount)}/{compactNumber(job.decentralizedChainAudit.proof?.stageCount)}
                        </span>
                        <span class="rounded-full border border-cai-medium-gray/25 bg-cai-black/30 px-2 py-0.5 text-white/45">
                          {t("history.tokens")} {compactNumber(job.decentralizedChainAudit.tokens?.promptTokens)}/{compactNumber(job.decentralizedChainAudit.tokens?.completionTokens)}
                        </span>
                        <span class="inline-flex items-center gap-1 rounded-full border border-cai-yellow/20 bg-cai-yellow/8 px-2 py-0.5 text-cai-yellow/85">
                          <CaiCoinIcon class="h-3 w-3 border border-cai-yellow/15" />
                          <span>{t("history.rewardAtoms")} {compactNumber(job.decentralizedChainAudit.reward?.workerPayoutTotalAtomic)}</span>
                        </span>
                      {/if}
                    </div>
                    {#if job.outputText}
                      <div class="mt-1 text-[10px] text-white/55 leading-relaxed">
                        {shortText(job.outputText, 90)}
                      </div>
                    {/if}
                    {#if job.lastError}
                      <div class="mt-1 text-[10px] text-red-300/80 leading-relaxed">
                        {shortText(job.lastError, 110)}
                      </div>
                    {/if}
                  </div>
                {/each}
              </div>
            {/if}
          </div>

          <div class="border border-cai-medium-gray/30 bg-cai-black/30 p-2">
            <div class="flex items-center justify-between gap-3 mb-2">
              <div class="text-[10px] text-white/40 font-mono uppercase tracking-wider">{t("history.transactions")}</div>
              <div class="text-[10px] text-white/35 font-mono">{journalItems.length}</div>
            </div>
            {#if journalItems.length === 0}
              <div class="text-[11px] text-white/45 font-mono">{t("history.noWallet")}</div>
            {:else}
              <div class="max-h-44 overflow-y-auto space-y-2 pr-1" onscroll={handleHistoryScroll}>
                {#each journalItems as entry}
                  <div class="border border-cai-medium-gray/20 bg-cai-black/20 p-2">
                    <div class="flex items-center justify-between gap-2 text-[10px] font-mono text-white/40 uppercase">
                      <span>{friendlyEventName(entry.eventType)}</span>
                      <span>{shortWhen(entry.createdAt)}</span>
                    </div>
                    <div class="mt-1 flex items-center gap-1.5 text-[11px] text-cai-light-gray font-mono">
                      <CaiCoinIcon class="h-3.5 w-3.5 border border-cai-yellow/15" />
                      <span>{formatCurrencyAmount(entry.amountCoins)}</span>
                    </div>
                    {#if entry.source || entry.balanceAfterCoins}
                      <div class="mt-1 text-[10px] text-white/35 font-mono uppercase tracking-wider">
                        {entry.source ?? "local"}{entry.balanceAfterCoins ? ` | balance ${formatCurrencyAmount(entry.balanceAfterCoins)}` : ""}
                      </div>
                    {/if}
                    {#if entry.counterpartyAddress}
                      <div class="mt-1 text-[10px] text-white/40 font-mono break-all">
                        {shortText(entry.counterpartyAddress, 28)}
                      </div>
                    {/if}
                    {#if entry.note}
                      <div class="mt-1 text-[10px] text-white/55 leading-relaxed">
                        {shortText(entry.note, 90)}
                      </div>
                    {/if}
                  </div>
                {/each}
              </div>
            {/if}
          </div>
        </div>
      {/if}

      {#if actionMessage}
        <div class="border border-green-500/20 bg-green-500/5 p-2 text-[11px] text-green-300 leading-relaxed break-words">
          {actionMessage}
        </div>
      {/if}
      {#if actionError}
        <div class="border border-red-500/20 bg-red-500/5 p-2 text-[11px] text-red-300 leading-relaxed break-words">
          {actionError}
        </div>
      {/if}
    </div>
  {/if}
</div>
