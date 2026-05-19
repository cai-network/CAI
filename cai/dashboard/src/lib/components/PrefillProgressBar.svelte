<!--
SPDX-FileCopyrightText: 2025 cai Technologies Ltd
SPDX-FileCopyrightText: 2026 CAI contributors
SPDX-License-Identifier: Apache-2.0
-->
<script lang="ts">
  import type { PrefillProgress } from "$lib/stores/app.svelte";
  import { tr, trf } from "$lib/stores/i18n.svelte";

  interface Props {
    progress: PrefillProgress;
    class?: string;
  }

  let { progress, class: className = "" }: Props = $props();

  const percentage = $derived(
    progress.total > 0
      ? Math.round((progress.processed / progress.total) * 100)
      : 0,
  );

  const etaText = $derived.by(() => {
    if (progress.processed <= 0 || progress.total <= 0) return null;
    const elapsedMs = performance.now() - progress.startedAt;
    if (elapsedMs < 200) return null; // need a minimum sample window
    const tokensPerMs = progress.processed / elapsedMs;
    const remainingTokens = progress.total - progress.processed;
    const remainingMs = remainingTokens / tokensPerMs;
    const remainingSec = Math.ceil(remainingMs / 1000);
    if (remainingSec <= 0) return null;
    if (remainingSec < 60) {
      return trf("~{seconds}s remaining", { seconds: remainingSec });
    }
    const mins = Math.floor(remainingSec / 60);
    const secs = remainingSec % 60;
    return trf("~{minutes}m {seconds}s remaining", {
      minutes: mins,
      seconds: secs,
    });
  });

  function formatTokenCount(count: number | undefined): string {
    if (count == null) return "0";
    if (count >= 1000) {
      return `${(count / 1000).toFixed(1)}k`;
    }
    return count.toString();
  }
</script>

<div class="prefill-progress {className}">
  <div
    class="flex items-center justify-between text-xs text-cai-light-gray mb-1"
  >
    <span>{tr("Processing prompt")}</span>
    <span class="font-mono">
      {formatTokenCount(progress.processed)} / {formatTokenCount(
        progress.total,
      )} {tr("tokens")}
    </span>
  </div>
  <div class="h-1.5 bg-cai-black/60 rounded-full overflow-hidden">
    <div
      class="h-full bg-cai-yellow rounded-full transition-all duration-150 ease-out"
      style="width: {percentage}%"
    ></div>
  </div>
  <div
    class="flex items-center justify-between text-xs text-cai-light-gray/70 mt-0.5 font-mono"
  >
    <span>{etaText ?? ""}</span>
    <span>{percentage}%</span>
  </div>
</div>

<style>
  .prefill-progress {
    width: 100%;
  }
</style>
