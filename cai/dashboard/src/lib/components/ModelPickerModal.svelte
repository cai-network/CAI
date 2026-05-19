<!--
SPDX-FileCopyrightText: 2025 cai Technologies Ltd
SPDX-FileCopyrightText: 2026 CAI contributors
SPDX-License-Identifier: Apache-2.0
-->
<script lang="ts">
  import { tick } from "svelte";
  import { fade, fly } from "svelte/transition";
  import { cubicOut } from "svelte/easing";
  import FamilySidebar from "./FamilySidebar.svelte";
  import FamilyLogos from "./FamilyLogos.svelte";
  import ModelPickerGroup from "./ModelPickerGroup.svelte";
  import ModelFilterPopover from "./ModelFilterPopover.svelte";
  import HuggingFaceResultItem from "./HuggingFaceResultItem.svelte";
  import { getNodesWithModelDownloaded } from "$lib/utils/downloads";
  import { getRecentEntries } from "$lib/stores/recents.svelte";
  import { addToast } from "$lib/stores/toast.svelte";
  import { tr, trf } from "$lib/stores/i18n.svelte";

  interface ModelInfo {
    id: string;
    name?: string;
    storage_size_megabytes?: number;
    base_model?: string;
    quantization?: string;
    supports_tensor?: boolean;
    capabilities?: string[];
    family?: string;
    is_custom?: boolean;
    tasks?: string[];
    hugging_face_id?: string;
    inference_backend?: string;
  }

  interface ModelGroup {
    id: string;
    name: string;
    capabilities: string[];
    family: string;
    variants: ModelInfo[];
    smallestVariant: ModelInfo;
    hasMultipleVariants: boolean;
  }

  interface FilterState {
    capabilities: string[];
    sizeRange: { min: number; max: number } | null;
    downloadedOnly: boolean;
    readyOnly: boolean;
  }

  interface HuggingFaceModel {
    id: string;
    author: string;
    downloads: number;
    likes: number;
    last_modified: string;
    tags: string[];
  }

  type ModelFitStatus = "fits_now" | "fits_cluster_capacity" | "too_large";

  export type InstanceStatus = {
    status: string;
    statusClass: string;
  };

  type ModelPickerModalProps = {
    isOpen: boolean;
    initialFamily?: string | null;
    models: ModelInfo[];
    selectedModelId: string | null;
    favorites: Set<string>;
    recentModelIds?: string[];
    hasRecents?: boolean;
    existingModelIds: Set<string>;
    canModelFit: (modelId: string) => boolean;
    getModelFitStatus: (modelId: string) => ModelFitStatus;
    onSelect: (modelId: string) => void;
    onClose: () => void;
    onToggleFavorite: (baseModelId: string) => void;
    onAddModel: (modelId: string) => Promise<void>;
    onAddLocalModel: (localPath: string, modelId?: string) => Promise<void>;
    onDeleteModel: (modelId: string) => Promise<void>;
    totalMemoryGB: number;
    usedMemoryGB: number;
    downloadsData?: Record<string, unknown[]>;
    topologyNodes?: Record<
      string,
      {
        friendly_name?: string;
        system_info?: { model_id?: string };
        macmon_info?: { memory?: { ram_total?: number } };
      }
    >;
    instanceStatuses?: Record<string, InstanceStatus>;
    allowCustomModelFlow?: boolean;
  };

  let {
    isOpen,
    initialFamily = null,
    models,
    selectedModelId,
    favorites,
    recentModelIds = [],
    hasRecents: hasRecentsTab = false,
    existingModelIds,
    canModelFit,
    getModelFitStatus,
    onSelect,
    onClose,
    onToggleFavorite,
    onAddModel,
    onAddLocalModel,
    onDeleteModel,
    totalMemoryGB,
    usedMemoryGB,
    downloadsData,
    topologyNodes,
    instanceStatuses = {},
    allowCustomModelFlow = false,
  }: ModelPickerModalProps = $props();

  // Local state
  let searchQuery = $state("");
  let selectedFamily = $state<string | null>(null);
  let expandedGroups = $state<Set<string>>(new Set());
  let showFilters = $state(false);
  let filters = $state<FilterState>({
    capabilities: [],
    sizeRange: null,
    downloadedOnly: false,
    readyOnly: false,
  });
  let infoGroup = $state<ModelGroup | null>(null);

  // Download availability per model group
  type DownloadAvailability = {
    available: boolean;
    nodeNames: string[];
    nodeIds: string[];
  };

  function getNodeName(nodeId: string): string {
    const node = topologyNodes?.[nodeId];
    return (
      node?.friendly_name || node?.system_info?.model_id || nodeId.slice(0, 8)
    );
  }

  const modelDownloadAvailability = $derived.by(() => {
    const result = new Map<string, DownloadAvailability>();
    if (!downloadsData || !topologyNodes) return result;

    for (const model of models) {
      const nodeIds = getNodesWithModelDownloaded(downloadsData, model.id);
      if (nodeIds.length === 0) continue;

      // Sum total RAM across nodes that have the model
      let totalRamBytes = 0;
      for (const nodeId of nodeIds) {
        const ramTotal = topologyNodes[nodeId]?.macmon_info?.memory?.ram_total;
        if (typeof ramTotal === "number") totalRamBytes += ramTotal;
      }

      const modelSizeBytes = (model.storage_size_megabytes || 0) * 1024 * 1024;
      result.set(model.id, {
        available: modelSizeBytes > 0 && totalRamBytes >= modelSizeBytes,
        nodeNames: nodeIds.map(getNodeName),
        nodeIds,
      });
    }
    return result;
  });

  const catalogModels = $derived.by((): ModelInfo[] => {
    return models.filter(
      (model) =>
        model.inference_backend === "llama_cpp" ||
        model.is_custom ||
        String(model.id || "")
          .toLowerCase()
          .includes("gguf"),
    );
  });

  // Aggregate download availability per group (available if ANY variant is available)
  function getGroupDownloadAvailability(
    group: ModelGroup,
  ): DownloadAvailability | undefined {
    for (const variant of group.variants) {
      const avail = modelDownloadAvailability.get(variant.id);
      if (avail && avail.nodeIds.length > 0) return avail;
    }
    return undefined;
  }

  // Get per-variant download map for a group
  function getVariantDownloadMap(
    group: ModelGroup,
  ): Map<string, DownloadAvailability> {
    const map = new Map<string, DownloadAvailability>();
    for (const variant of group.variants) {
      const avail = modelDownloadAvailability.get(variant.id);
      if (avail && avail.nodeIds.length > 0) map.set(variant.id, avail);
    }
    return map;
  }

  // HuggingFace Hub state
  let hfSearchQuery = $state("");
  let hfSearchResults = $state<HuggingFaceModel[]>([]);
  let hfTrendingModels = $state<HuggingFaceModel[]>([]);
  let hfFamilyBrowseResults = $state<Record<string, HuggingFaceModel[]>>({});
  let hfIsSearching = $state(false);
  let hfIsLoadingTrending = $state(false);
  let hfIsLoadingFamily = $state(false);
  let addingModelId = $state<string | null>(null);
  let hfSearchDebounceTimer: ReturnType<typeof setTimeout> | null = null;
  let manualModelId = $state("");
  let localModelPath = $state("");
  let localModelId = $state("");
  const HF_TRENDING_LIMIT = 100;
  const HF_SEARCH_LIMIT = 60;
  const INLINE_SEARCH_LIMIT = 30;
  const DISCOVER_FAMILY_ORDER = [
    "qwen",
    "qwen-abliterated",
    "gemma",
    "glm",
    "deepseek",
    "gpt-oss",
    "llama",
    "nemotron",
    "minimax",
    "kimi",
    "step",
  ] as const;
  const FAMILY_DISCOVER_QUERY: Record<string, string> = {
    qwen: "Qwen GGUF",
    "qwen-abliterated": "Qwen abliterated GGUF",
    gemma: "Gemma GGUF",
    glm: "GLM GGUF",
    deepseek: "DeepSeek GGUF",
    "gpt-oss": "gpt-oss GGUF",
    llama: "Llama GGUF",
    nemotron: "Nemotron GGUF",
    minimax: "MiniMax GGUF",
    kimi: "Kimi GGUF",
    step: "Step-3.5-Flash GGUF",
    "qwen-image": "Qwen-Image GGUF",
  };
  const FAMILY_DISCOVER_LABELS: Record<string, string> = {
    qwen: "Qwen",
    "qwen-abliterated": "Qwen Ablit",
    gemma: "Google",
    glm: "GLM",
    deepseek: "DeepSeek",
    "gpt-oss": "OpenAI",
    llama: "Meta",
    nemotron: "NVIDIA",
    minimax: "MiniMax",
    kimi: "Kimi",
    step: "Step",
    "qwen-image": "Qwen Img",
  };

  function normalizeFamilyName(family: string | null | undefined): string {
    return (family || "").trim().toLowerCase();
  }

  function inferHfFamily(model: HuggingFaceModel): string | null {
    const id = model.id.toLowerCase();
    const tags = model.tags.map((tag) => tag.toLowerCase());
    const match = (needle: string) =>
      id.includes(needle) || tags.some((tag) => tag.includes(needle));

    if (match("qwen-image")) return "qwen-image";
    if (match("qwen") && (match("abliterated") || match("abliteration")))
      return "qwen-abliterated";
    if (match("qwen")) return "qwen";
    if (match("gemma")) return "gemma";
    if (match("glm")) return "glm";
    if (match("deepseek")) return "deepseek";
    if (match("gpt-oss")) return "gpt-oss";
    if (match("llama") || match("meta-llama")) return "llama";
    if (match("nemotron")) return "nemotron";
    if (match("minimax")) return "minimax";
    if (match("kimi")) return "kimi";
    if (match("step-3.5-flash") || match("stepfun") || match("step35"))
      return "step";
    return null;
  }

  function isHfFamilyValue(family: string | null): boolean {
    return family === "huggingface" || family?.startsWith("hf:") === true;
  }

  function isHfSelection(family: string | null): boolean {
    return (
      allowCustomModelFlow &&
      isHfFamilyValue(family)
    );
  }

  function getSelectedHfFamily(family: string | null): string | null {
    return family?.startsWith("hf:") ? family.slice(3) : null;
  }

  function getHfFamilyLabel(family: string): string {
    return (
      FAMILY_DISCOVER_LABELS[family] ||
      family.charAt(0).toUpperCase() + family.slice(1)
    );
  }

  $effect(() => {
    if (!isOpen) return;
    selectedFamily =
      allowCustomModelFlow || !isHfFamilyValue(initialFamily)
        ? initialFamily
        : null;
  });
  let addModelError = $state<string | null>(null);
  let justAddedModelId = $state<string | null>(null);
  let justAddedTimer: ReturnType<typeof setTimeout> | null = null;

  // Inline HuggingFace search in main search bar
  let mainSearchHfResults = $state<HuggingFaceModel[]>([]);
  let mainSearchHfLoading = $state(false);
  let mainSearchDebounceTimer: ReturnType<typeof setTimeout> | null = null;

  // Reset transient state when modal opens, but preserve tab selection
  $effect(() => {
    if (isOpen) {
      searchQuery = "";
      expandedGroups = new Set();
      showFilters = false;
      manualModelId = "";
      localModelPath = "";
      localModelId = "";
      addModelError = null;
      justAddedModelId = null;
      if (justAddedTimer) {
        clearTimeout(justAddedTimer);
        justAddedTimer = null;
      }
    }
  });

  // Fetch trending models when HuggingFace is selected
  $effect(() => {
    if (
      allowCustomModelFlow &&
      selectedFamily === "huggingface" &&
      hfTrendingModels.length === 0 &&
      !hfIsLoadingTrending
    ) {
      fetchTrendingModels();
    }
  });

  $effect(() => {
    const selectedHfFamily = getSelectedHfFamily(selectedFamily);
    if (
      !allowCustomModelFlow ||
      !selectedHfFamily ||
      hfSearchQuery.trim().length >= 2 ||
      selectedHfFamily in hfFamilyBrowseResults ||
      hfIsLoadingFamily
    ) {
      return;
    }
    fetchFamilyModels(selectedHfFamily);
  });

  // Inline HuggingFace search alongside local results so GGUF Hub matches do not
  // disappear just because one local family happens to match the query.
  $effect(() => {
    const query = searchQuery.trim();

    if (mainSearchDebounceTimer) {
      clearTimeout(mainSearchDebounceTimer);
      mainSearchDebounceTimer = null;
    }

    if (
      !allowCustomModelFlow ||
      isHfSelection(selectedFamily) ||
      selectedFamily === "recents" ||
      selectedFamily === "favorites" ||
      query.length < 2
    ) {
      mainSearchHfResults = [];
      mainSearchHfLoading = false;
      return;
    }

    mainSearchHfLoading = true;
    mainSearchDebounceTimer = setTimeout(async () => {
      try {
        const response = await fetch(
          `/models/search?query=${encodeURIComponent(query)}&limit=${INLINE_SEARCH_LIMIT}`,
        );
        if (response.ok) {
          const results: HuggingFaceModel[] = await response.json();
          mainSearchHfResults = results.filter(
            (r) => !existingModelIds.has(r.id),
          );
        } else {
          mainSearchHfResults = [];
        }
      } catch {
        mainSearchHfResults = [];
      } finally {
        mainSearchHfLoading = false;
      }
    }, 500);
  });

  async function fetchTrendingModels() {
    hfIsLoadingTrending = true;
    try {
      const response = await fetch(
        `/models/search?query=&limit=${HF_TRENDING_LIMIT}`,
      );
      if (response.ok) {
        hfTrendingModels = await response.json();
      }
    } catch (error) {
      console.error("Failed to fetch trending models:", error);
    } finally {
      hfIsLoadingTrending = false;
    }
  }

  function getFamilyQueryPrefix(family: string | null): string {
    if (!family) return "";
    return FAMILY_DISCOVER_QUERY[family] || family;
  }

  async function fetchFamilyModels(family: string) {
    hfIsLoadingFamily = true;
    try {
      const response = await fetch(
        `/models/search?query=${encodeURIComponent(getFamilyQueryPrefix(family))}&limit=${HF_TRENDING_LIMIT}`,
      );
      if (response.ok) {
        const results: HuggingFaceModel[] = await response.json();
        hfFamilyBrowseResults = {
          ...hfFamilyBrowseResults,
          [family]: results,
        };
      }
    } catch (error) {
      console.error(`Failed to fetch ${family} models:`, error);
    } finally {
      hfIsLoadingFamily = false;
    }
  }

  async function searchHuggingFace(query: string, family: string | null = null) {
    if (query.length < 2) {
      hfSearchResults = [];
      return;
    }

    hfIsSearching = true;
    try {
      const prefixedQuery = family
        ? `${getFamilyQueryPrefix(family)} ${query}`.trim()
        : query;
      const response = await fetch(
        `/models/search?query=${encodeURIComponent(prefixedQuery)}&limit=${HF_SEARCH_LIMIT}`,
      );
      if (response.ok) {
        hfSearchResults = await response.json();
      } else {
        hfSearchResults = [];
      }
    } catch (error) {
      console.error("Failed to search models:", error);
      hfSearchResults = [];
    } finally {
      hfIsSearching = false;
    }
  }

  function handleHfSearchInput(query: string) {
    hfSearchQuery = query;
    addModelError = null;
    const selectedHfFamily = getSelectedHfFamily(selectedFamily);

    if (hfSearchDebounceTimer) {
      clearTimeout(hfSearchDebounceTimer);
    }

    if (query.length >= 2) {
      hfSearchDebounceTimer = setTimeout(() => {
        searchHuggingFace(query, selectedHfFamily);
      }, 300);
    } else {
      hfSearchResults = [];
    }
  }

  async function handleAddModel(modelId: string) {
    addingModelId = modelId;
    addModelError = null;
    try {
      await onAddModel(modelId);
      // Success: show toast, switch to All Models, highlight the model
      const shortName = modelId.split("/").pop() || modelId;
      addToast({
        type: "success",
        message: trf("Added {model}", { model: shortName }),
      });
      justAddedModelId = modelId;
      selectedFamily = null;
      searchQuery = "";
      // Scroll to the newly added model after DOM update
      await tick();
      const el = document.querySelector(
        `[data-model-ids~="${CSS.escape(modelId)}"]`,
      );
      el?.scrollIntoView({ behavior: "smooth", block: "center" });
      // Clear highlight after 4 seconds
      if (justAddedTimer) clearTimeout(justAddedTimer);
      justAddedTimer = setTimeout(() => {
        justAddedModelId = null;
        justAddedTimer = null;
      }, 4000);
    } catch (error) {
      addModelError =
        error instanceof Error ? error.message : tr("Failed to add model");
    } finally {
      addingModelId = null;
    }
  }

  async function handleAddManualModel() {
    if (!manualModelId.trim()) return;
    await handleAddModel(manualModelId.trim());
    if (!addModelError) {
      manualModelId = "";
    }
  }

  async function handleAddLocalModel() {
    if (!localModelPath.trim()) return;
    addingModelId = localModelPath.trim();
    addModelError = null;
    try {
      await onAddLocalModel(
        localModelPath.trim(),
        localModelId.trim() || undefined,
      );
      const shortName =
        localModelId.trim() ||
        localModelPath
          .trim()
          .split(/[\\/]/)
          .filter(Boolean)
          .at(-1) ||
        tr("local model");
      addToast({
        type: "success",
        message: trf("Added {model}", { model: shortName }),
      });
      manualModelId = "";
      localModelPath = "";
      localModelId = "";
      searchQuery = "";
      selectedFamily = null;
    } catch (error) {
      addModelError =
        error instanceof Error ? error.message : tr("Failed to add local model");
    } finally {
      addingModelId = null;
    }
  }

  function handleSelectHfModel(modelId: string) {
    onSelect(modelId);
    onClose();
  }

  // Models to display in HuggingFace view
  const hfDisplayModels = $derived.by((): HuggingFaceModel[] => {
    const selectedHfFamily = getSelectedHfFamily(selectedFamily);
    const baseResults =
      hfSearchQuery.length >= 2
        ? hfSearchResults
        : selectedHfFamily
          ? hfFamilyBrowseResults[selectedHfFamily] || []
          : hfTrendingModels;
    if (!selectedHfFamily) {
      return baseResults;
    }
    return baseResults.filter(
      (model) => inferHfFamily(model) === selectedHfFamily,
    );
  });

  // Group models by base_model
  const groupedModels = $derived.by((): ModelGroup[] => {
    const groups = new Map<string, ModelGroup>();

    for (const model of catalogModels) {
      const groupId = model.base_model || model.id;
      const groupName = model.base_model || model.name || model.id;

      if (!groups.has(groupId)) {
        groups.set(groupId, {
          id: groupId,
          name: groupName,
          capabilities: model.capabilities || ["text"],
          family: model.family || "",
          variants: [],
          smallestVariant: model,
          hasMultipleVariants: false,
        });
      }

      const group = groups.get(groupId)!;
      group.variants.push(model);

      // Track smallest variant
      if (
        (model.storage_size_megabytes || 0) <
        (group.smallestVariant.storage_size_megabytes || Infinity)
      ) {
        group.smallestVariant = model;
      }

      // Update capabilities if not set
      if (
        group.capabilities.length <= 1 &&
        model.capabilities &&
        model.capabilities.length > 1
      ) {
        group.capabilities = model.capabilities;
      }
      if (!group.family && model.family) {
        group.family = model.family;
      }
    }

    // Sort variants within each group by size
    for (const group of groups.values()) {
      group.variants.sort(
        (a, b) =>
          (a.storage_size_megabytes || 0) - (b.storage_size_megabytes || 0),
      );
      group.hasMultipleVariants = group.variants.length > 1;
    }

    // Convert to array and sort by smallest variant size (biggest first)
    return Array.from(groups.values()).sort((a, b) => {
      return (
        (b.smallestVariant.storage_size_megabytes || 0) -
        (a.smallestVariant.storage_size_megabytes || 0)
      );
    });
  });

  // Get unique families
  const uniqueFamilies = $derived.by((): string[] => {
    const families = new Set<string>();
    for (const group of groupedModels) {
      if (group.family) {
        families.add(normalizeFamilyName(group.family));
      }
    }
    const familyOrder = [
      "kimi",
      "qwen",
      "glm",
      "minimax",
      "deepseek",
      "gpt-oss",
      "llama",
      "gemma",
      "flux",
      "qwen-image",
      "nemotron",
    ];
    return Array.from(families).sort((a, b) => {
      const aIdx = familyOrder.indexOf(a);
      const bIdx = familyOrder.indexOf(b);
      if (aIdx === -1 && bIdx === -1) return a.localeCompare(b);
      if (aIdx === -1) return 1;
      if (bIdx === -1) return -1;
      return aIdx - bIdx;
    });
  });

  const sidebarFamilies = $derived.by((): string[] => {
    const merged = [...uniqueFamilies];
    if (!allowCustomModelFlow) {
      return merged;
    }
    const pinnedDiscoverFamilies = ["hf:qwen-abliterated"];
    for (const family of pinnedDiscoverFamilies) {
      if (!merged.includes(family)) {
        merged.push(family);
      }
    }
    return merged;
  });

  const hfBrowseFamilies = $derived.by((): string[] => {
    const discovered = new Set<string>();
    if (!allowCustomModelFlow) {
      return [];
    }

    for (const family of DISCOVER_FAMILY_ORDER) {
      discovered.add(family);
    }

    for (const model of hfTrendingModels) {
      const family = inferHfFamily(model);
      if (family) discovered.add(family);
    }

    for (const model of hfSearchResults) {
      const family = inferHfFamily(model);
      if (family) discovered.add(family);
    }

    return Array.from(discovered).sort((a, b) => {
      const aIdx = DISCOVER_FAMILY_ORDER.indexOf(
        a as (typeof DISCOVER_FAMILY_ORDER)[number],
      );
      const bIdx = DISCOVER_FAMILY_ORDER.indexOf(
        b as (typeof DISCOVER_FAMILY_ORDER)[number],
      );
      if (aIdx === -1 && bIdx === -1) return a.localeCompare(b);
      if (aIdx === -1) return 1;
      if (bIdx === -1) return -1;
      return aIdx - bIdx;
    });
  });

  // Filter models based on search, family, and filters
  const filteredGroups = $derived.by((): ModelGroup[] => {
    let result: ModelGroup[] = [...groupedModels];

    // Filter by family
    if (selectedFamily === "favorites") {
      result = result.filter((g) => favorites.has(g.id));
    } else if (
      selectedFamily &&
      !isHfSelection(selectedFamily) &&
      selectedFamily !== "recents"
    ) {
      result = result.filter(
        (g) => normalizeFamilyName(g.family) === normalizeFamilyName(selectedFamily),
      );
    }

    // Filter by search query
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase().trim();
      result = result.filter(
        (g) =>
          g.name.toLowerCase().includes(query) ||
          g.variants.some(
            (v) =>
              v.id.toLowerCase().includes(query) ||
              (v.name || "").toLowerCase().includes(query),
          ),
      );
    }

    // Filter by capabilities
    if (filters.capabilities.length > 0) {
      result = result.filter((g) =>
        filters.capabilities.every((cap) => g.capabilities.includes(cap)),
      );
    }

    // Filter by size range
    if (filters.sizeRange) {
      const { min, max } = filters.sizeRange;
      result = result.filter((g) => {
        const sizeGB = (g.smallestVariant.storage_size_megabytes || 0) / 1024;
        return sizeGB >= min && sizeGB <= max;
      });
    }

    // Filter to downloaded models only
    if (filters.downloadedOnly) {
      result = result.filter((g) =>
        g.variants.some((v) => {
          const avail = modelDownloadAvailability.get(v.id);
          return avail && avail.nodeIds.length > 0;
        }),
      );
    }

    // Filter to ready/running models only
    if (filters.readyOnly) {
      result = result.filter((g) =>
        g.variants.some((v) => {
          const s = instanceStatuses[v.id];
          return s && s.statusClass === "ready";
        }),
      );
    }

    // Sort: fits-now first, then fits-cluster-capacity, then too-large
    result.sort((a, b) => {
      const getGroupFitRank = (group: ModelGroup): number => {
        let hasClusterCapacityOnly = false;
        for (const variant of group.variants) {
          const fitStatus = getModelFitStatus(variant.id);
          if (fitStatus === "fits_now") return 0;
          if (fitStatus === "fits_cluster_capacity") {
            hasClusterCapacityOnly = true;
          }
        }
        return hasClusterCapacityOnly ? 1 : 2;
      };

      const aRank = getGroupFitRank(a);
      const bRank = getGroupFitRank(b);
      if (aRank !== bRank) return aRank - bRank;

      return (
        (b.smallestVariant.storage_size_megabytes || 0) -
        (a.smallestVariant.storage_size_megabytes || 0)
      );
    });

    return result;
  });

  // Check if any favorites exist
  const hasFavorites = $derived(favorites.size > 0);

  // Timestamp lookup for recent models
  const recentTimestamps = $derived(
    new Map(getRecentEntries().map((e) => [e.modelId, e.launchedAt])),
  );

  // Recent models: single-variant ModelGroups in launch order
  const recentGroups = $derived.by((): ModelGroup[] => {
    if (!recentModelIds || recentModelIds.length === 0) return [];
    const result: ModelGroup[] = [];
    for (const id of recentModelIds) {
      const model = models.find((m) => m.id === id);
      if (model) {
        result.push({
          id: model.base_model || model.id,
          name: model.name || model.id,
          capabilities: model.capabilities || ["text"],
          family: model.family || "",
          variants: [model],
          smallestVariant: model,
          hasMultipleVariants: false,
        });
      }
    }
    return result;
  });

  // Filtered recent groups (apply search query)
  const filteredRecentGroups = $derived.by((): ModelGroup[] => {
    if (!searchQuery.trim()) return recentGroups;
    const query = searchQuery.toLowerCase().trim();
    return recentGroups.filter(
      (g) =>
        g.name.toLowerCase().includes(query) ||
        g.variants.some(
          (v) =>
            v.id.toLowerCase().includes(query) ||
            (v.name || "").toLowerCase().includes(query) ||
            (v.quantization || "").toLowerCase().includes(query),
        ),
    );
  });

  // Split filtered groups into recommended (fits_now) and others for visual separation
  const recommendedGroups = $derived(
    filteredGroups.filter((g) =>
      g.variants.some((v) => getModelFitStatus(v.id) === "fits_now"),
    ),
  );
  const otherGroups = $derived(
    filteredGroups.filter(
      (g) => !g.variants.some((v) => getModelFitStatus(v.id) === "fits_now"),
    ),
  );
  const visibleVariantCount = $derived(
    filteredGroups.reduce((acc, group) => acc + group.variants.length, 0),
  );
  const recommendedVariantCount = $derived(
    recommendedGroups.reduce((acc, group) => acc + group.variants.length, 0),
  );
  const otherVariantCount = $derived(
    otherGroups.reduce((acc, group) => acc + group.variants.length, 0),
  );

  function toggleGroupExpanded(groupId: string) {
    const next = new Set(expandedGroups);
    if (next.has(groupId)) {
      next.delete(groupId);
    } else {
      next.add(groupId);
    }
    expandedGroups = next;
  }

  function handleSelect(modelId: string) {
    onSelect(modelId);
    onClose();
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === "Escape") {
      onClose();
    }
  }

  function handleFiltersChange(newFilters: FilterState) {
    filters = newFilters;
  }

  function clearFilters() {
    filters = {
      capabilities: [],
      sizeRange: null,
      downloadedOnly: false,
      readyOnly: false,
    };
  }

  const hasActiveFilters = $derived(
    filters.capabilities.length > 0 ||
      filters.sizeRange !== null ||
      filters.downloadedOnly ||
      filters.readyOnly,
  );
</script>

<svelte:window onkeydown={handleKeydown} />

{#if isOpen}
  <!-- Backdrop -->
  <div
    class="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm"
    transition:fade={{ duration: 200 }}
    onclick={onClose}
    role="presentation"
  ></div>

  <!-- Modal -->
  <div
    class="fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[min(90vw,600px)] h-[min(80vh,700px)] bg-cai-dark-gray border border-cai-yellow/10 rounded-lg shadow-2xl overflow-hidden flex flex-col"
    transition:fly={{ y: 20, duration: 300, easing: cubicOut }}
    role="dialog"
    aria-modal="true"
    aria-label={tr("Select a model")}
  >
    <!-- Header with search -->
    <div
      class="flex items-center gap-2 p-3 border-b border-cai-yellow/10 bg-cai-medium-gray/30"
    >
      {#if isHfSelection(selectedFamily)}
        <!-- HuggingFace search -->
        <svg
          class="w-5 h-5 text-orange-400/60 flex-shrink-0"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
        >
          <circle cx="11" cy="11" r="8" />
          <path d="M21 21l-4.35-4.35" />
        </svg>
        <input
          type="search"
          class="flex-1 bg-transparent border-none outline-none text-sm font-mono text-white placeholder-white/40"
          placeholder={tr("Search supported GGUF models...")}
          value={hfSearchQuery}
          oninput={(e) => handleHfSearchInput(e.currentTarget.value)}
        />
        {#if hfIsSearching}
          <div class="flex-shrink-0">
            <span
              class="w-4 h-4 border-2 border-orange-400 border-t-transparent rounded-full animate-spin block"
            ></span>
          </div>
        {/if}
      {:else}
        <!-- Normal model search -->
        <svg
          class="w-5 h-5 text-white/40 flex-shrink-0"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
        >
          <circle cx="11" cy="11" r="8" />
          <path d="M21 21l-4.35-4.35" />
        </svg>
        <input
          type="search"
          class="flex-1 bg-transparent border-none outline-none text-sm font-mono text-white placeholder-white/40"
          placeholder={tr("Search models...")}
          bind:value={searchQuery}
        />
        <!-- Cluster memory -->
        <span
          class="text-xs font-mono flex-shrink-0"
          title={tr("Cluster memory usage")}
          ><span class="text-cai-yellow">{Math.round(usedMemoryGB)}GB</span
          ><span class="text-white/40">/{Math.round(totalMemoryGB)}GB</span
          ></span
        >
        <!-- Filter button -->
        <div class="relative filter-toggle">
          <button
            type="button"
            class="p-1.5 rounded hover:bg-white/10 transition-colors {hasActiveFilters
              ? 'text-cai-yellow'
              : 'text-white/50'}"
            onclick={() => (showFilters = !showFilters)}
            title={tr("Filter by capability or size")}
          >
            <svg class="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
              <path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z" />
            </svg>
          </button>
          {#if showFilters}
            <ModelFilterPopover
              {filters}
              onChange={handleFiltersChange}
              onClear={clearFilters}
              onClose={() => (showFilters = false)}
            />
          {/if}
        </div>
      {/if}
      <!-- Close button -->
      <button
        type="button"
        class="p-1.5 rounded hover:bg-white/10 transition-colors text-white/50 hover:text-white/70"
        onclick={onClose}
        title={tr("Close model picker")}
      >
        <svg class="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
          <path
            d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z"
          />
        </svg>
      </button>
    </div>

    <!-- Body -->
    <div class="flex flex-1 overflow-hidden">
      <!-- Family sidebar -->
      <FamilySidebar
        families={sidebarFamilies}
        selectedFamily={selectedFamily}
        {hasFavorites}
        hasRecents={hasRecentsTab}
        {allowCustomModelFlow}
        onSelect={(family) => (selectedFamily = family)}
      />

      <!-- Model list -->
      <div class="flex-1 overflow-y-auto scrollbar-hide flex flex-col">
        {#if isHfSelection(selectedFamily)}
          <!-- HuggingFace Hub view -->
          <div class="flex-1 flex flex-col min-h-0">
            <!-- Section header -->
            <div
              class="sticky top-0 z-10 px-3 py-2 bg-cai-dark-gray/95 border-b border-cai-yellow/10"
            >
              <div class="flex flex-col gap-2">
                <span class="text-xs font-mono text-white/40">
                  {#if hfSearchQuery.length >= 2}
                    {#if getSelectedHfFamily(selectedFamily)}
                      {trf("{family} supported GGUF matches for \"{query}\"", {
                        family: getHfFamilyLabel(
                          getSelectedHfFamily(selectedFamily) || "",
                        ),
                        query: hfSearchQuery,
                      })}
                    {:else}
                      {trf("Search results for \"{query}\"", {
                        query: hfSearchQuery,
                      })}
                    {/if}
                  {:else}
                    {#if getSelectedHfFamily(selectedFamily)}
                      {trf("Trending {family} supported GGUF models", {
                        family: getHfFamilyLabel(
                          getSelectedHfFamily(selectedFamily) || "",
                        ),
                      })}
                    {:else}
                      {tr("Trending supported GGUF models")}
                    {/if}
                  {/if}
                </span>

                <div class="flex flex-wrap gap-1.5">
                  <button
                    type="button"
                    class="inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-mono uppercase tracking-[0.14em] transition-colors {selectedFamily ===
                    'huggingface'
                      ? 'border-orange-400/60 bg-orange-500/12 text-orange-300'
                      : 'border-white/10 bg-white/4 text-white/45 hover:border-white/20 hover:text-white/70'}"
                    onclick={() => (selectedFamily = "huggingface")}
                  >
                    <span>{tr("All")}</span>
                  </button>
                  {#each hfBrowseFamilies as family}
                    <button
                      type="button"
                      class="inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-mono uppercase tracking-[0.14em] transition-colors {getSelectedHfFamily(selectedFamily) ===
                      family
                        ? 'border-orange-400/60 bg-orange-500/12 text-orange-300'
                        : 'border-white/10 bg-white/4 text-white/45 hover:border-white/20 hover:text-white/70'}"
                      onclick={() => (selectedFamily = `hf:${family}`)}
                    >
                      <FamilyLogos family={family} class="w-3.5 h-3.5" />
                      <span>{getHfFamilyLabel(family)}</span>
                    </button>
                  {/each}
                </div>
              </div>
            </div>

            <!-- Results list -->
            <div class="flex-1 overflow-y-auto scrollbar-hide">
              {#if ((selectedFamily === "huggingface" && hfIsLoadingTrending && hfTrendingModels.length === 0) || (getSelectedHfFamily(selectedFamily) && hfIsLoadingFamily && hfDisplayModels.length === 0))}
                <div
                  class="flex items-center justify-center py-12 text-white/40"
                >
                  <span
                    class="w-5 h-5 border-2 border-orange-400 border-t-transparent rounded-full animate-spin mr-2"
                  ></span>
                  <span class="font-mono text-sm"
                    >{tr("Loading models...")}</span
                  >
                </div>
              {:else if hfDisplayModels.length === 0}
                <div
                  class="flex flex-col items-center justify-center py-12 text-white/40"
                >
                  <svg
                    class="w-10 h-10 mb-2"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                  >
                    <path
                      d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 13.5c-.83 0-1.5-.67-1.5-1.5s.67-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5zm4 0c-.83 0-1.5-.67-1.5-1.5s.67-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5zm2-4.5H8c0-2.21 1.79-4 4-4s4 1.79 4 4z"
                    />
                  </svg>
                  <p class="font-mono text-sm">{tr("No models found")}</p>
                  {#if hfSearchQuery}
                    <p class="font-mono text-xs mt-1">
                      {tr("Try a different search term")}
                    </p>
                  {/if}
                </div>
              {:else}
                {#each hfDisplayModels as model}
                  <HuggingFaceResultItem
                    {model}
                    isAdded={existingModelIds.has(model.id)}
                    isAdding={addingModelId === model.id}
                    onAdd={() => handleAddModel(model.id)}
                    onSelect={() => handleSelectHfModel(model.id)}
                    downloadedOnNodes={downloadsData
                      ? getNodesWithModelDownloaded(
                          downloadsData,
                          model.id,
                        ).map(getNodeName)
                      : []}
                  />
                {/each}
              {/if}
            </div>

            <!-- Manual input footer -->
            <div
              class="sticky bottom-0 border-t border-cai-yellow/10 bg-cai-dark-gray p-3"
            >
              {#if addModelError}
                <div
                  class="bg-red-500/10 border border-red-500/30 rounded px-3 py-2 mb-2"
                >
                  <p class="text-red-400 text-xs font-mono break-words">
                    {addModelError}
                  </p>
                </div>
              {/if}
              <div class="space-y-3">
                <div>
                  <div class="mb-1 text-[10px] font-mono uppercase tracking-[0.18em] text-white/35">
                    {tr("Add By Model ID")}
                  </div>
                  <div class="flex gap-2">
                    <input
                      type="text"
                      class="flex-1 bg-cai-black/60 border border-cai-yellow/30 rounded px-3 py-1.5 text-xs font-mono text-white placeholder-white/30 focus:outline-none focus:border-cai-yellow/50"
                      placeholder={tr("Paste Hugging Face model ID...")}
                      bind:value={manualModelId}
                      onkeydown={(e) => {
                        if (e.key === "Enter") handleAddManualModel();
                      }}
                    />
                    <button
                      type="button"
                      onclick={handleAddManualModel}
                      disabled={!manualModelId.trim() || addingModelId !== null}
                      class="px-3 py-1.5 text-xs font-mono tracking-wider uppercase bg-orange-500/10 text-orange-400 border border-orange-400/30 hover:bg-orange-500/20 transition-colors rounded disabled:opacity-50 disabled:cursor-not-allowed"
                     >
                      {tr("Add")}
                    </button>
                  </div>
                </div>

                <div class="border-t border-white/8 pt-3">
                  <div class="mb-1 text-[10px] font-mono uppercase tracking-[0.18em] text-white/35">
                    {tr("Add From Local Folder")}
                  </div>
                  <div class="space-y-2">
                    <input
                      type="text"
                      class="w-full bg-cai-black/60 border border-cai-yellow/30 rounded px-3 py-1.5 text-xs font-mono text-white placeholder-white/30 focus:outline-none focus:border-cai-yellow/50"
                      placeholder={tr("Folder path, for example D:\\Models\\MyModel or /mnt/d/Models/MyModel")}
                      bind:value={localModelPath}
                      onkeydown={(e) => {
                        if (e.key === "Enter") handleAddLocalModel();
                      }}
                    />
                    <div class="flex gap-2">
                      <input
                        type="text"
                        class="flex-1 bg-cai-black/60 border border-white/10 rounded px-3 py-1.5 text-xs font-mono text-white placeholder-white/30 focus:outline-none focus:border-cai-yellow/40"
                        placeholder={tr("Optional custom model ID, for example local/my-model")}
                        bind:value={localModelId}
                        onkeydown={(e) => {
                          if (e.key === "Enter") handleAddLocalModel();
                        }}
                      />
                      <button
                        type="button"
                        onclick={handleAddLocalModel}
                        disabled={!localModelPath.trim() || addingModelId !== null}
                        class="px-3 py-1.5 text-xs font-mono tracking-wider uppercase bg-cai-yellow/10 text-cai-yellow border border-cai-yellow/30 hover:bg-cai-yellow/20 transition-colors rounded disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {tr("Add Local")}
                      </button>
                    </div>
                    <p class="text-[10px] font-mono text-white/30">
                      {tr("We will register this folder as a custom model on this node.")}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        {:else if selectedFamily === "recents"}
          <!-- Recent models view -->
          {#if filteredRecentGroups.length === 0}
            <div
              class="flex flex-col items-center justify-center h-full text-white/40 p-8"
            >
              <svg
                class="w-12 h-12 mb-3"
                viewBox="0 0 24 24"
                fill="currentColor"
              >
                <path
                  d="M13 3a9 9 0 0 0-9 9H1l3.89 3.89.07.14L9 12H6c0-3.87 3.13-7 7-7s7 3.13 7 7-3.13 7-7 7c-1.93 0-3.68-.79-4.94-2.06l-1.42 1.42A8.954 8.954 0 0 0 13 21a9 9 0 0 0 0-18zm-1 5v5l4.28 2.54.72-1.21-3.5-2.08V8H12z"
                />
              </svg>
              <p class="font-mono text-sm">
                {searchQuery
                  ? tr("No matching recent models")
                  : tr("No recently launched models")}
              </p>
            </div>
          {:else}
            {#each filteredRecentGroups as group}
              <ModelPickerGroup
                {group}
                isExpanded={expandedGroups.has(group.id)}
                isFavorite={favorites.has(group.id)}
                isHighlighted={justAddedModelId !== null &&
                  group.variants.some((v) => v.id === justAddedModelId)}
                {selectedModelId}
                {canModelFit}
                {getModelFitStatus}
                onToggleExpand={() => toggleGroupExpanded(group.id)}
                onSelectModel={handleSelect}
                {onToggleFavorite}
                onShowInfo={(g) => (infoGroup = g)}
                downloadStatusMap={getVariantDownloadMap(group)}
                launchedAt={recentTimestamps.get(group.variants[0]?.id ?? "")}
                {instanceStatuses}
              />
            {/each}
          {/if}
        {:else if filteredGroups.length === 0}
          <div
            class="flex flex-col items-center justify-center h-full text-white/40 p-8"
          >
            <svg class="w-12 h-12 mb-3" viewBox="0 0 24 24" fill="currentColor">
              <path
                d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"
              />
            </svg>
            <p class="font-mono text-sm">{tr("No models found")}</p>
            {#if hasActiveFilters || searchQuery}
              <button
                type="button"
                class="mt-2 text-xs text-cai-yellow hover:underline"
                onclick={() => {
                  searchQuery = "";
                  clearFilters();
                }}
              >
                {tr("Clear filters")}
              </button>
            {/if}
          </div>
        {:else}
          <div
            class="sticky top-0 z-10 px-3 py-2 bg-cai-dark-gray/95 border-b border-cai-yellow/10 backdrop-blur-sm"
          >
            <div class="flex items-center justify-between gap-3 text-xs font-mono">
              <span class="text-white/55">
                {trf("{families} model families / {variants} variants", {
                  families: filteredGroups.length,
                  variants: visibleVariantCount,
                })}
              </span>
              {#if recommendedGroups.length > 0 && otherGroups.length > 0 && !searchQuery.trim()}
                <span class="text-white/35">
                  {trf("{ready} ready now, {more} more below", {
                    ready: recommendedGroups.length,
                    more: otherGroups.length,
                  })}
                </span>
              {/if}
            </div>
          </div>
          <!-- Recommended for your cluster -->
          {#if recommendedGroups.length > 0 && otherGroups.length > 0 && !searchQuery.trim()}
            <div
              class="sticky top-0 z-10 flex items-center gap-2 px-3 py-2 bg-green-950/60 border-b border-green-500/20 backdrop-blur-sm"
            >
              <svg
                class="w-3.5 h-3.5 text-green-400 flex-shrink-0"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                stroke-width="2"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <span
                class="text-xs font-mono text-green-400 tracking-wider uppercase"
                >{tr("Recommended for your cluster")}</span
              >
              <span class="text-xs font-mono text-green-400/50"
                >- {trf("{count} variants fit in available memory", {
                  count: recommendedVariantCount,
                })}</span
              >
            </div>
          {/if}
          {#each recommendedGroups as group}
            <ModelPickerGroup
              {group}
              isExpanded={expandedGroups.has(group.id)}
              isFavorite={favorites.has(group.id)}
              isHighlighted={justAddedModelId !== null &&
                group.variants.some((v) => v.id === justAddedModelId)}
              {selectedModelId}
              {canModelFit}
              {getModelFitStatus}
              onToggleExpand={() => toggleGroupExpanded(group.id)}
              onSelectModel={handleSelect}
              {onToggleFavorite}
              onShowInfo={(g) => (infoGroup = g)}
              downloadStatusMap={getVariantDownloadMap(group)}
              {instanceStatuses}
            />
          {/each}
          <!-- Other models -->
          {#if otherGroups.length > 0 && recommendedGroups.length > 0 && !searchQuery.trim()}
            <div
              class="sticky top-0 z-10 flex items-center gap-2 px-3 py-2 bg-cai-dark-gray/80 border-y border-cai-medium-gray/20 backdrop-blur-sm"
            >
              <span
                class="text-xs font-mono text-white/40 tracking-wider uppercase"
                >{tr("Other models")}</span
              >
              <span class="text-xs font-mono text-white/30"
                >- {trf("{count} more variants in the catalog", {
                  count: otherVariantCount,
                })}</span
              >
            </div>
          {/if}
          {#each otherGroups as group}
            <ModelPickerGroup
              {group}
              isExpanded={expandedGroups.has(group.id)}
              isFavorite={favorites.has(group.id)}
              isHighlighted={justAddedModelId !== null &&
                group.variants.some((v) => v.id === justAddedModelId)}
              {selectedModelId}
              {canModelFit}
              {getModelFitStatus}
              onToggleExpand={() => toggleGroupExpanded(group.id)}
              onSelectModel={handleSelect}
              {onToggleFavorite}
              onShowInfo={(g) => (infoGroup = g)}
              downloadStatusMap={getVariantDownloadMap(group)}
              {instanceStatuses}
            />
          {/each}
          <!-- Inline HuggingFace search results (shown when no local results match) -->
          {#if allowCustomModelFlow && searchQuery.trim().length >= 2 && !isHfSelection(selectedFamily) && selectedFamily !== "recents" && selectedFamily !== "favorites"}
            {#if mainSearchHfLoading}
              <div
                class="flex items-center gap-2 px-3 py-2 border-t border-cyan-300/20 bg-cyan-950/20"
              >
                <span
                  class="w-4 h-4 border-2 border-cyan-300 border-t-transparent rounded-full animate-spin"
                ></span>
                <span class="text-xs font-mono text-cyan-200/70"
                  >{tr("Searching HuggingFace...")}</span
                >
              </div>
            {:else if mainSearchHfResults.length > 0}
              <div
                class="sticky top-0 z-10 flex items-center gap-2 px-3 py-2 bg-cyan-950/30 border-y border-cyan-300/20 backdrop-blur-sm"
              >
                <span
                  class="text-xs font-mono text-cyan-200 tracking-wider uppercase"
                  >{tr("More Supported GGUF Models From HuggingFace")}</span
                >
              </div>
              {#each mainSearchHfResults as model}
                <HuggingFaceResultItem
                  {model}
                  isAdded={existingModelIds.has(model.id)}
                  isAdding={addingModelId === model.id}
                  onAdd={() => handleAddModel(model.id)}
                  onSelect={() => handleSelectHfModel(model.id)}
                  downloadedOnNodes={downloadsData
                    ? getNodesWithModelDownloaded(downloadsData, model.id).map(
                        getNodeName,
                      )
                    : []}
                />
              {/each}
              <button
                type="button"
                class="w-full px-3 py-2 text-xs font-mono text-cyan-200/70 hover:text-cyan-100 hover:bg-cyan-400/10 transition-colors text-center"
                onclick={() => {
                  hfSearchQuery = searchQuery;
                  searchHuggingFace(searchQuery);
                  selectedFamily = "huggingface";
                }}
              >
                {tr("See all results on Hub")}
              </button>
            {/if}
          {/if}
        {/if}
      </div>
    </div>

    <!-- Footer with active filters indicator -->
    {#if hasActiveFilters}
      <div
        class="flex items-center gap-2 px-3 py-2 border-t border-cai-yellow/10 bg-cai-medium-gray/20 text-xs font-mono text-white/50"
      >
        <span>{tr("Filters:")}</span>
        {#each filters.capabilities as cap}
          <span class="px-1.5 py-0.5 bg-cai-yellow/20 text-cai-yellow rounded"
            >{cap}</span
          >
        {/each}
        {#if filters.downloadedOnly}
          <span class="px-1.5 py-0.5 bg-green-500/20 text-green-400 rounded"
            >{tr("Downloaded")}</span
          >
        {/if}
        {#if filters.readyOnly}
          <span class="px-1.5 py-0.5 bg-green-500/20 text-green-400 rounded"
            >{tr("Ready")}</span
          >
        {/if}
        {#if filters.sizeRange}
          <span class="px-1.5 py-0.5 bg-cai-yellow/20 text-cai-yellow rounded">
            {filters.sizeRange.min}GB - {filters.sizeRange.max}GB
          </span>
        {/if}
        <button
          type="button"
          class="ml-auto text-white/40 hover:text-white/60"
          onclick={clearFilters}
        >
          {tr("Clear all")}
        </button>
      </div>
    {/if}
  </div>

  <!-- Info modal -->
  {#if infoGroup}
    <div
      class="fixed inset-0 z-[60] bg-black/60"
      transition:fade={{ duration: 150 }}
      onclick={() => (infoGroup = null)}
      role="presentation"
    ></div>
    <div
      class="fixed z-[60] top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[min(80vw,400px)] bg-cai-dark-gray border border-cai-yellow/10 rounded-lg shadow-2xl p-4"
      transition:fly={{ y: 10, duration: 200, easing: cubicOut }}
      role="dialog"
      aria-modal="true"
    >
      <div class="flex items-start justify-between mb-3">
        <h3 class="font-mono text-lg text-white">{infoGroup.name}</h3>
        <button
          type="button"
          class="p-1 rounded hover:bg-white/10 transition-colors text-white/50"
          onclick={() => (infoGroup = null)}
          title={tr("Close model details")}
          aria-label={tr("Close info dialog")}
        >
          <svg class="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
            <path
              d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z"
            />
          </svg>
        </button>
      </div>
      <div class="space-y-2 text-xs font-mono">
        <div class="flex items-center gap-2">
          <span class="text-white/40">{tr("Family:")}</span>
          <span class="text-white/70">{infoGroup.family || "Unknown"}</span>
        </div>
        <div class="flex items-center gap-2">
          <span class="text-white/40">{tr("Capabilities:")}</span>
          <span class="text-white/70">{infoGroup.capabilities.join(", ")}</span>
        </div>
        <div class="flex items-center gap-2">
          <span class="text-white/40">{tr("Variants:")}</span>
          <span class="text-white/70">{infoGroup.variants.length}</span>
        </div>
        {#if infoGroup.variants.length > 0}
          <div class="mt-3 pt-3 border-t border-cai-yellow/10">
            <span class="text-white/40">{tr("Available quantizations:")}</span>
            <div class="flex flex-wrap gap-1 mt-1">
              {#each infoGroup.variants as variant}
                <span
                  class="px-1.5 py-0.5 bg-white/10 text-white/60 rounded text-[10px]"
                >
                  {variant.quantization || tr("default")} ({Math.round(
                    (variant.storage_size_megabytes || 0) / 1024,
                  )}GB)
                </span>
              {/each}
            </div>
          </div>
        {/if}
        {#if getGroupDownloadAvailability(infoGroup)?.nodeNames?.length}
          {@const infoDownload = getGroupDownloadAvailability(infoGroup)}
          {#if infoDownload}
            <div class="mt-3 pt-3 border-t border-cai-yellow/10">
              <div class="flex items-center gap-2 mb-1">
                <svg
                  class="w-3.5 h-3.5"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  stroke-width="2"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                >
                  <path
                    class="text-white/40"
                    d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"
                  />
                  <path class="text-green-400" d="m9 13 2 2 4-4" />
                </svg>
                <span class="text-white/40">{tr("Downloaded on:")}</span>
              </div>
              <div class="flex flex-wrap gap-1 mt-1">
                {#each infoDownload.nodeNames as nodeName}
                  <span
                    class="px-1.5 py-0.5 bg-green-500/10 text-green-400/80 border border-green-500/20 rounded text-[10px]"
                  >
                    {nodeName}
                  </span>
                {/each}
              </div>
            </div>
          {/if}
        {/if}
      </div>
    </div>
  {/if}
{/if}

