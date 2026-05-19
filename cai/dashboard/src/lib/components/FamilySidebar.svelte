<!--
SPDX-FileCopyrightText: 2025 cai Technologies Ltd
SPDX-FileCopyrightText: 2026 CAI contributors
SPDX-License-Identifier: Apache-2.0
-->
<script lang="ts">
  import FamilyLogos from "./FamilyLogos.svelte";
  import { tr } from "$lib/stores/i18n.svelte";

  type FamilySidebarProps = {
    families: string[];
    selectedFamily: string | null;
    hasFavorites: boolean;
    hasRecents: boolean;
    allowCustomModelFlow?: boolean;
    onSelect: (family: string | null) => void;
  };

  let {
    families,
    selectedFamily,
    hasFavorites,
    hasRecents,
    allowCustomModelFlow = false,
    onSelect,
  }: FamilySidebarProps = $props();

  function normalizeFamily(family: string): string {
    return family.replace(/^hf:/, "");
  }

  // Family display names
  const familyNames: Record<string, string> = {
    favorites: "Favorites",
    recents: "Recent",
    huggingface: "Add",
    llama: "Meta",
    qwen: "Qwen",
    "qwen-abliterated": "Qwen Ablit",
    deepseek: "DeepSeek",
    "gpt-oss": "OpenAI",
    glm: "GLM",
    minimax: "MiniMax",
    kimi: "Kimi",
    flux: "FLUX",
    "qwen-image": "Qwen Img",
    nemotron: "NVIDIA",
    gemma: "Google",
  };

  function getFamilyName(family: string): string {
    const normalized = normalizeFamily(family);
    return (
      (familyNames[normalized] ? tr(familyNames[normalized]) : null) ||
      normalized.charAt(0).toUpperCase() + normalized.slice(1)
    );
  }

  function isDiscoverFamily(family: string): boolean {
    return family.startsWith("hf:");
  }
</script>

<div
  class="flex flex-col gap-1 py-2 px-1 border-r border-cai-yellow/10 bg-cai-medium-gray/30 min-w-[80px] sm:min-w-[72px] overflow-y-auto scrollbar-hide"
>
  <!-- All models (no filter) -->
  <button
    type="button"
    onclick={() => onSelect(null)}
    class="group flex items-center justify-center px-3 py-2.5 rounded transition-all duration-200 cursor-pointer min-h-[44px] sm:min-h-0 {selectedFamily ===
    null
      ? 'bg-cai-yellow/20 border-l-2 border-cai-yellow'
      : 'hover:bg-white/5 border-l-2 border-transparent'}"
    title={tr("All models")}
  >
    <span
      class="text-[12px] font-mono font-medium {selectedFamily === null
        ? 'text-cai-yellow'
        : 'text-white/40 group-hover:text-white/60'}">{tr("All")}</span
    >
  </button>

  <!-- Favorites (only show if has favorites) -->
  {#if hasFavorites}
    <button
      type="button"
      onclick={() => onSelect("favorites")}
      class="group flex flex-col items-center justify-center p-2 rounded transition-all duration-200 cursor-pointer {selectedFamily ===
      'favorites'
        ? 'bg-cai-yellow/20 border-l-2 border-cai-yellow'
        : 'hover:bg-white/5 border-l-2 border-transparent'}"
      title={tr("Show favorited models")}
    >
      <FamilyLogos
        family="favorites"
        class={selectedFamily === "favorites"
          ? "text-amber-400"
          : "text-white/50 group-hover:text-amber-400/70"}
      />
      <span
        class="text-[11px] font-mono mt-0.5 {selectedFamily === 'favorites'
          ? 'text-amber-400'
          : 'text-white/40 group-hover:text-white/60'}">{tr("Faves")}</span
      >
    </button>
  {/if}

  <!-- Recent (only show if has recent models) -->
  {#if hasRecents}
    <button
      type="button"
      onclick={() => onSelect("recents")}
      class="group flex flex-col items-center justify-center p-2 rounded transition-all duration-200 cursor-pointer {selectedFamily ===
      'recents'
        ? 'bg-cai-yellow/20 border-l-2 border-cai-yellow'
        : 'hover:bg-white/5 border-l-2 border-transparent'}"
      title={tr("Recently launched models")}
    >
      <FamilyLogos
        family="recents"
        class={selectedFamily === "recents"
          ? "text-cai-yellow"
          : "text-white/50 group-hover:text-white/70"}
      />
      <span
        class="text-[11px] font-mono mt-0.5 {selectedFamily === 'recents'
          ? 'text-cai-yellow'
          : 'text-white/40 group-hover:text-white/60'}">{tr("Recent")}</span
      >
    </button>
  {/if}

  {#if allowCustomModelFlow}
    <!-- HuggingFace Hub -->
    <button
      type="button"
      onclick={() => onSelect("huggingface")}
      class="group flex flex-col items-center justify-center p-2 rounded transition-all duration-200 cursor-pointer {selectedFamily ===
      'huggingface'
        ? 'bg-orange-500/20 border-l-2 border-orange-400'
        : 'hover:bg-white/5 border-l-2 border-transparent'}"
        title={tr("Add models from Hugging Face or a local folder")}
    >
      <FamilyLogos
        family="huggingface"
        class={selectedFamily === "huggingface"
          ? "text-orange-400"
          : "text-white/50 group-hover:text-orange-400/70"}
      />
      <span
        class="text-[11px] font-mono mt-0.5 {selectedFamily === 'huggingface'
          ? 'text-orange-400'
          : 'text-white/40 group-hover:text-white/60'}">{tr("Add")}</span
      >
    </button>

    <div class="h-px bg-cai-yellow/10 my-1"></div>
  {/if}

  <!-- Model families -->
  {#each families as family, index}
    {#if index > 0 && isDiscoverFamily(family) && !isDiscoverFamily(families[index - 1])}
      <div class="h-px bg-cai-yellow/10 my-1"></div>
    {/if}
    <button
      type="button"
      onclick={() => onSelect(family)}
      class="group flex flex-col items-center justify-center p-2 rounded transition-all duration-200 cursor-pointer {selectedFamily ===
      family
        ? 'bg-cai-yellow/20 border-l-2 border-cai-yellow'
        : 'hover:bg-white/5 border-l-2 border-transparent'}"
      title={getFamilyName(family)}
    >
      <FamilyLogos
        {family}
        class={selectedFamily === family
          ? "text-cai-yellow"
          : "text-white/50 group-hover:text-white/70"}
      />
      <span
        class="text-[11px] font-mono mt-0.5 truncate max-w-full {selectedFamily ===
        family
          ? 'text-cai-yellow'
          : 'text-white/40 group-hover:text-white/60'}"
      >
        {getFamilyName(family)}
      </span>
    </button>
  {/each}
</div>
