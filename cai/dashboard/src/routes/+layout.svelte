<!--
SPDX-FileCopyrightText: 2025 cai Technologies Ltd
SPDX-FileCopyrightText: 2026 CAI contributors
SPDX-License-Identifier: Apache-2.0
-->
<script lang="ts">
  import "../app.css";
  import ToastContainer from "$lib/components/ToastContainer.svelte";
  import ConnectionBanner from "$lib/components/ConnectionBanner.svelte";

  let { children } = $props();
</script>

<svelte:head>
  <title>CAI</title>
  <meta
    name="description"
    content="CAI - Distributed AI Network Dashboard"
  />
</svelte:head>

<div class="min-h-screen bg-background text-foreground">
  <ConnectionBanner />
  {@render children?.()}
  <ToastContainer />
</div>
