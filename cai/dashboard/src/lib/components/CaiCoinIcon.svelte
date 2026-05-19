<!--
SPDX-FileCopyrightText: 2025 cai Technologies Ltd
SPDX-FileCopyrightText: 2026 CAI contributors
SPDX-License-Identifier: Apache-2.0
-->
<script lang="ts">
  type IconVariant = "coin" | "full";

  interface Props {
    class?: string;
    decorative?: boolean;
    label?: string;
    variant?: IconVariant;
  }

  let {
    class: className = "h-4 w-4",
    decorative = true,
    label = "CAI",
    variant = "coin",
  }: Props = $props();

  const altText = $derived(decorative ? "" : label);
  const src = $derived(variant === "full" ? "/cai-coin-full.png" : "/cai-coin.png");
  const shapeClass = $derived(
    variant === "full" ? "object-contain" : "rounded-full object-cover",
  );
</script>

<img
  {src}
  alt={altText}
  aria-hidden={decorative}
  decoding="async"
  loading="lazy"
  class={`inline-block ${shapeClass} ${className}`}
/>
