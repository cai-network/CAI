<!--
SPDX-FileCopyrightText: 2025 cai Technologies Ltd
SPDX-FileCopyrightText: 2026 CAI contributors
SPDX-License-Identifier: Apache-2.0
-->
<script lang="ts">
  import { tr } from "$lib/stores/i18n.svelte";

  import type {
    CaiModelShardInventoryEntry,
    DownloadProgress,
    NodeInfo,
    PlacementPreview,
  } from "$lib/stores/app.svelte";

  interface Props {
    model: {
      id: string;
      name?: string;
      storage_size_megabytes?: number;
      gguf_architecture?: string;
      shard_compatibility?: string;
      layer_range_supported?: boolean;
      model_package_manifest_url?: string | null;
      model_package_catalog_id?: string | null;
      model_package_version?: string | null;
      layer_range_probe_report?: string | null;
      layer_range_equivalence_probe_report?: string | null;
      shard_compatibility_reason?: string;
    };
    isLaunching?: boolean;
    downloadStatus?: {
      isDownloading: boolean;
      progress: DownloadProgress | null;
      perNode?: Array<{
        nodeId: string;
        nodeName: string;
        status: "completed" | "partial" | "pending" | "downloading";
        percentage: number;
        progress: DownloadProgress | null;
        statusMessage?: string | null;
      }>;
    } | null;
    nodes?: Record<string, NodeInfo>;
    sharding?: "Pipeline" | "Tensor";
    runtime?: "MlxRing" | "MlxJaccl" | "LlamaCpp";
    onLaunch?: () => void;
    tags?: string[];
    apiPreview?: PlacementPreview | null;
    modelIdOverride?: string | null;
    modelPackageCache?: CaiModelShardInventoryEntry | null;
  }

  let {
    model,
    isLaunching = false,
    downloadStatus = null,
    nodes = {},
    sharding = "Pipeline",
    runtime = "MlxRing",
    onLaunch,
    tags = [],
    apiPreview = null,
    modelIdOverride = null,
    modelPackageCache = null,
  }: Props = $props();

  // Estimate memory requirements from model name
  // Uses regex with word boundaries to avoid false matches like '4bit' matching '4b'
  function estimateMemoryGB(modelId: string, modelName?: string): number {
    // Check both ID and name for quantization info
    const combined = `${modelId} ${modelName || ""}`.toLowerCase();

    // Detect quantization level - affects memory by roughly 2x between levels
    const is4bit =
      combined.includes("4bit") ||
      combined.includes("4-bit") ||
      combined.includes(":4bit");
    const is8bit =
      combined.includes("8bit") ||
      combined.includes("8-bit") ||
      combined.includes(":8bit");
    // 4-bit = 0.5 bytes/param, 8-bit = 1 byte/param, fp16 = 2 bytes/param
    const quantMultiplier = is4bit ? 0.5 : is8bit ? 1 : 2;
    const id = modelId.toLowerCase();

    // Known large models that don't follow the standard naming pattern
    // DeepSeek V3 has 685B parameters
    if (id.includes("deepseek-v3")) {
      return Math.round(685 * quantMultiplier);
    }
    // DeepSeek V2 has 236B parameters
    if (id.includes("deepseek-v2")) {
      return Math.round(236 * quantMultiplier);
    }
    // Llama 4 Scout/Maverick are large models
    if (id.includes("llama-4")) {
      return Math.round(400 * quantMultiplier);
    }

    // Match parameter counts with word boundaries (e.g., "70b" but not "4bit")
    const paramMatch = id.match(/(\d+(?:\.\d+)?)\s*b(?![a-z])/i);
    if (paramMatch) {
      const params = parseFloat(paramMatch[1]);
      return Math.max(4, Math.round(params * quantMultiplier));
    }

    // Fallback patterns for explicit size markers (assume fp16 baseline, adjust for quant)
    if (id.includes("405b") || id.includes("400b"))
      return Math.round(405 * quantMultiplier);
    if (id.includes("180b")) return Math.round(180 * quantMultiplier);
    if (id.includes("141b") || id.includes("140b"))
      return Math.round(140 * quantMultiplier);
    if (id.includes("123b") || id.includes("120b"))
      return Math.round(123 * quantMultiplier);
    if (id.includes("72b") || id.includes("70b"))
      return Math.round(70 * quantMultiplier);
    if (id.includes("67b") || id.includes("65b"))
      return Math.round(65 * quantMultiplier);
    if (
      id.includes("35b") ||
      id.includes("34b") ||
      id.includes("32b") ||
      id.includes("30b")
    )
      return Math.round(32 * quantMultiplier);
    if (id.includes("27b") || id.includes("26b") || id.includes("22b"))
      return Math.round(24 * quantMultiplier);
    if (id.includes("14b") || id.includes("13b") || id.includes("15b"))
      return Math.round(14 * quantMultiplier);
    if (id.includes("8b") || id.includes("9b") || id.includes("7b"))
      return Math.round(8 * quantMultiplier);
    if (id.includes("3b") || id.includes("3.8b"))
      return Math.round(4 * quantMultiplier);
    if (
      id.includes("2b") ||
      id.includes("1b") ||
      id.includes("1.5b") ||
      id.includes("0.5b")
    )
      return Math.round(2 * quantMultiplier);

    return 16; // Default fallback
  }

  function formatBytes(bytes: number, decimals = 1): string {
    if (!bytes || bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return (
      parseFloat((bytes / Math.pow(k, i)).toFixed(decimals)) + " " + sizes[i]
    );
  }

  function formatSpeed(bps: number): string {
    if (!bps || bps <= 0) return "0 B/s";
    return formatBytes(bps) + "/s";
  }

  function shardModeLabel(
    compatibility?: string,
    layerRangeSupported?: boolean,
  ): string {
    if (compatibility === "layer_range_supported" || layerRangeSupported) {
      return "Layer-range";
    }
    if (compatibility === "unsupported_for_sharding") {
      return "Local only";
    }
    if (compatibility === "low_latency_rpc_cell") {
      return "RPC cell";
    }
    if (compatibility === "full_model_local") {
      return "Full model";
    }
    return "GGUF";
  }

  function shardModeClass(
    compatibility?: string,
    layerRangeSupported?: boolean,
  ): string {
    if (compatibility === "layer_range_supported" || layerRangeSupported) {
      return "bg-green-500/15 text-green-300 border-green-500/25";
    }
    if (compatibility === "unsupported_for_sharding") {
      return "bg-cai-medium-gray/30 text-cai-light-gray border-cai-medium-gray/40";
    }
    return "bg-blue-500/15 text-blue-300 border-blue-500/25";
  }

  const perNode = $derived(downloadStatus?.perNode ?? []);
  let expandedNodes = $state<Set<string>>(new Set());

  function toggleNodeDetails(nodeId: string): void {
    const next = new Set(expandedNodes);
    if (next.has(nodeId)) {
      next.delete(nodeId);
    } else {
      next.add(nodeId);
    }
    expandedNodes = next;
  }

  // Use actual storage_size_megabytes from API if available, otherwise fall back to estimate
  const estimatedMemory = $derived(
    model.storage_size_megabytes
      ? Math.round(model.storage_size_megabytes / 1024)
      : estimateMemoryGB(model.id, model.name),
  );

  const huggingFaceModelId = $derived(modelIdOverride ?? model.id);
  const ggufShardCompatibility = $derived(model.shard_compatibility ?? "");
  const showGgufShardMode = $derived(
    runtime === "LlamaCpp" && ggufShardCompatibility.length > 0,
  );
  const hasModelPackageManifest = $derived(
    Boolean(model.model_package_manifest_url || model.model_package_catalog_id),
  );
  const packageCacheSummary = $derived(
    modelPackageCache?.chunkCache ?? modelPackageCache ?? null,
  );
  const hasPackageCacheSummary = $derived(
    Boolean(
      packageCacheSummary &&
        typeof packageCacheSummary.totalChunkCount === "number" &&
        packageCacheSummary.totalChunkCount > 0,
    ),
  );
  const packageCacheLabel = $derived.by(() => {
    if (!packageCacheSummary) return "";
    const cached = Math.max(0, Number(packageCacheSummary.cachedChunkCount ?? 0));
    const total = Math.max(0, Number(packageCacheSummary.totalChunkCount ?? 0));
    if (packageCacheSummary.fullCacheReady) {
      return tr("Seed cache ready");
    }
    if (total > 0) {
      return `${tr("Chunks")} ${cached}/${total}`;
    }
    return tr("Chunk cache");
  });

  const nodeList = $derived(() => {
    const ids = new Set(Object.keys(nodes));
    for (const nodeId of Object.keys(apiPreview?.memory_delta_by_node ?? {})) {
      ids.add(nodeId);
    }
    return [...ids];
  });

  const canFit = $derived(apiPreview ? apiPreview.error === null : false);
  const placementError = $derived(apiPreview?.error ?? null);
  const nodeCount = $derived(nodeList().length);
</script>

<div class="relative group">
  <!-- Corner accents -->
  <div
    class="absolute -top-px -left-px w-2 h-2 border-l border-t {canFit
      ? 'border-cai-yellow/30 group-hover:border-cai-yellow/60'
      : 'border-red-500/30'} transition-colors"
  ></div>
  <div
    class="absolute -top-px -right-px w-2 h-2 border-r border-t {canFit
      ? 'border-cai-yellow/30 group-hover:border-cai-yellow/60'
      : 'border-red-500/30'} transition-colors"
  ></div>
  <div
    class="absolute -bottom-px -left-px w-2 h-2 border-l border-b {canFit
      ? 'border-cai-yellow/30 group-hover:border-cai-yellow/60'
      : 'border-red-500/30'} transition-colors"
  ></div>
  <div
    class="absolute -bottom-px -right-px w-2 h-2 border-r border-b {canFit
      ? 'border-cai-yellow/30 group-hover:border-cai-yellow/60'
      : 'border-red-500/30'} transition-colors"
  ></div>

  <div
    class="bg-cai-dark-gray/60 border {canFit
      ? 'border-cai-yellow/20 group-hover:border-cai-yellow/40'
      : 'border-red-500/20'} p-3 transition-all duration-200 group-hover:shadow-[0_0_15px_rgba(125,211,252,0.1)]"
  >
    <!-- Model Name & Memory Required -->
    <div class="flex items-start justify-between gap-2 mb-2">
      <div class="flex-1 min-w-0">
        <div class="flex items-center gap-2">
          <div
            class="text-cai-yellow text-xs font-mono tracking-wide truncate"
            title={model.name || model.id}
          >
            {model.name || model.id}
          </div>
          {#if huggingFaceModelId}
            <a
              class="shrink-0 text-white/60 hover:text-cai-yellow transition-colors"
              href={`https://huggingface.co/${huggingFaceModelId}`}
              target="_blank"
              rel="noreferrer noopener"
              aria-label={tr("View model on Hugging Face")}
            >
              <svg
                class="w-3.5 h-3.5"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              >
                <path d="M14 3h7v7" />
                <path d="M10 14l11-11" />
                <path
                  d="M21 14v6a1 1 0 0 1-1 1h-16a1 1 0 0 1-1-1v-16a1 1 0 0 1 1-1h6"
                />
              </svg>
            </a>
          {/if}
          {#if tags.length > 0}
            <div class="flex gap-1 flex-shrink-0">
              {#each tags as tag}
                <span
                  class="px-1.5 py-0.5 text-xs font-mono tracking-wider uppercase rounded {tag ===
                  'FASTEST'
                    ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                    : 'bg-purple-500/20 text-purple-400 border border-purple-500/30'}"
                >
                  {tag}
                </span>
              {/each}
            </div>
          {/if}
        </div>
        {#if model.name && model.name !== model.id}
          <div
            class="text-xs text-cai-light-gray font-mono truncate mt-0.5"
            title={model.id}
          >
            {model.id}
          </div>
        {/if}
      </div>
      <div class="flex-shrink-0 text-right">
        <div
          class="text-xs font-mono {canFit
            ? 'text-cai-yellow'
            : 'text-red-400'}"
        >
          {estimatedMemory}GB
        </div>
      </div>
    </div>

    <!-- Configuration Badge -->
    <div class="flex items-center gap-1.5 mb-2">
      <span
        class="px-1.5 py-0.5 text-xs font-mono tracking-wider uppercase bg-cai-medium-gray/30 text-cai-light-gray border border-cai-medium-gray/40"
        title={sharding === "Pipeline"
          ? tr("Pipeline: splits model into sequential stages across devices. Lower network overhead.")
          : tr("Tensor: splits each layer across devices. Best with high-bandwidth connections (Thunderbolt).")}
      >
        {sharding}
      </span>
      <span
        class="px-1.5 py-0.5 text-xs font-mono tracking-wider uppercase bg-cai-medium-gray/30 text-cai-light-gray border border-cai-medium-gray/40"
        title={runtime === "LlamaCpp"
          ? tr("llama.cpp runtime for GGUF models over standard TCP/IP networking.")
          : runtime === "MlxRing"
          ? tr("Ring: standard networking. Works over any connection (Wi-Fi, Ethernet, Thunderbolt).")
          : tr("RDMA: direct memory access over Thunderbolt. Significantly faster for multi-device inference.")}
      >
        {runtime === "LlamaCpp"
          ? "GGUF / llama.cpp"
          : runtime === "MlxRing"
          ? "MLX Ring"
          : runtime === "MlxJaccl"
            ? "MLX RDMA"
            : runtime}
      </span>
      {#if showGgufShardMode}
        <span
          class="px-1.5 py-0.5 text-xs font-mono tracking-wider uppercase border {shardModeClass(
            ggufShardCompatibility,
            model.layer_range_supported,
          )}"
          title={model.shard_compatibility_reason ||
            model.layer_range_equivalence_probe_report ||
            model.layer_range_probe_report ||
            ggufShardCompatibility}
        >
          {shardModeLabel(ggufShardCompatibility, model.layer_range_supported)}
        </span>
      {/if}
      {#if hasModelPackageManifest}
        <span
          class="px-1.5 py-0.5 text-xs font-mono tracking-wider uppercase bg-cai-yellow/10 text-cai-yellow border border-cai-yellow/25"
          title={tr("This model publishes a CAI chunk manifest, so workers can prepare only the assigned GGUF chunks.")}
        >
          {tr("Chunk manifest")}
        </span>
      {/if}
      {#if hasPackageCacheSummary}
        <span
          class="px-1.5 py-0.5 text-xs font-mono tracking-wider uppercase border {packageCacheSummary?.fullCacheReady
            ? 'bg-green-500/15 text-green-300 border-green-500/25'
            : 'bg-blue-500/15 text-blue-300 border-blue-500/25'}"
          title={tr("Local cache coverage for this CAI model package. A ready seed cache can serve chunks to other executors.")}
        >
          {packageCacheLabel}
        </span>
      {/if}
    </div>

    <!-- Download Status (per-node) -->
    {#if perNode.length > 0}
      <div class="mb-2 space-y-1">
        <div
          class="text-[10px] font-mono text-white/20 tracking-widest uppercase"
        >
          {tr("Download progress")}
        </div>
        {#each perNode as node}
          <div class="flex items-center gap-2 text-xs font-mono">
            <span class="text-white/40 w-20 truncate" title={node.nodeId}
              >{node.nodeName}</span
            >
            <div
              class="flex-1 h-1 bg-cai-medium-gray/30 rounded overflow-hidden"
            >
              <div
                class="h-full transition-all duration-300 {node.status ===
                'downloading'
                  ? 'bg-blue-500/70'
                  : node.status === 'completed'
                    ? 'bg-cai-yellow/40'
                    : 'bg-white/20'}"
                style="width: {node.percentage}%"
              ></div>
            </div>
            <span
              class="text-right {node.status === 'completed'
                ? 'text-cai-yellow/60'
                : node.status === 'downloading'
                  ? 'text-blue-400/60'
                  : 'text-white/30'}"
            >
              {#if node.status === "downloading" && node.progress}
                {Math.round(node.percentage)}% {formatSpeed(
                  node.progress.speed,
                )}
              {:else if node.statusMessage}
                {tr(node.statusMessage)}
              {:else}
                {node.percentage > 0 ? `${Math.round(node.percentage)}%` : "0%"}
              {/if}
            </span>
          </div>
        {/each}
      </div>
    {/if}

    <!-- Launch Button -->
    <button
      onclick={onLaunch}
      disabled={isLaunching || !canFit}
      class="w-full py-2 text-sm font-mono tracking-wider uppercase border transition-all duration-200
				{isLaunching
        ? 'bg-transparent text-cai-yellow border-cai-yellow/50 cursor-wait'
        : !canFit
          ? 'bg-red-500/10 text-red-400/70 border-red-500/30 cursor-not-allowed'
          : 'bg-transparent text-cai-light-gray border-cai-light-gray/40 hover:text-cai-yellow hover:border-cai-yellow/50 cursor-pointer'}"
    >
      {#if isLaunching}
        <span class="flex items-center justify-center gap-1.5">
          <span
            class="w-2 h-2 border border-cai-yellow border-t-transparent rounded-full animate-spin"
          ></span>
          {tr("Launching...")}
        </span>
      {:else if !canFit}
        {tr("Insufficient memory")}
      {:else}
        {tr("Launch")}
      {/if}
    </button>
  </div>
</div>
