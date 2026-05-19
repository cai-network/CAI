<!--
SPDX-FileCopyrightText: 2025 cai Technologies Ltd
SPDX-FileCopyrightText: 2026 CAI contributors
SPDX-License-Identifier: Apache-2.0
-->
<script lang="ts">
  import { browser } from "$app/environment";
  import { fade } from "svelte/transition";
  import HeaderNav from "$lib/components/HeaderNav.svelte";
  import IntegrationCard from "$lib/components/IntegrationCard.svelte";
  import { instances, refreshState } from "$lib/stores/app.svelte";
  import { tr, trf } from "$lib/stores/i18n.svelte";
  import { onMount } from "svelte";

  const apiUrl = browser
    ? window.location.origin.replace("localhost", "127.0.0.1")
    : "http://127.0.0.1:52415";

  const instancesData = $derived(instances());

  let modelCapabilities = $state<Record<string, string[]>>({});
  let modelContextLengths = $state<Record<string, number>>({});

  const runningModels = $derived.by(() => {
    const models: string[] = [];
    for (const [, wrapper] of Object.entries(instancesData)) {
      if (wrapper && typeof wrapper === "object") {
        const values = Object.values(wrapper as Record<string, unknown>);
        if (values.length > 0) {
          const instance = values[0];
          if (instance && typeof instance === "object") {
            const inst = instance as {
              shardAssignments?: { modelId?: string };
            };
            const modelId = inst.shardAssignments?.modelId;
            if (modelId && !models.includes(modelId)) {
              models.push(modelId);
            }
          }
        }
      }
    }
    return models;
  });

  function estimateParamSize(modelId: string): number {
    const match = modelId.match(/(\d+(?:\.\d+)?)[Bb]/);
    return match ? parseFloat(match[1]) : 0;
  }

  const modelsBySize = $derived(
    [...runningModels].sort(
      (a, b) => estimateParamSize(b) - estimateParamSize(a),
    ),
  );

  const defaultTiers = $derived.by(() => {
    const n = modelsBySize.length;
    if (n === 0)
      return {
        opus: "your-model-id",
        sonnet: "your-model-id",
        haiku: "your-model-id",
      };
    if (n === 1)
      return {
        opus: modelsBySize[0],
        sonnet: modelsBySize[0],
        haiku: modelsBySize[0],
      };
    if (n === 2)
      return {
        opus: modelsBySize[0],
        sonnet: modelsBySize[1],
        haiku: modelsBySize[1],
      };
    return {
      opus: modelsBySize[0],
      sonnet: modelsBySize[Math.floor(n / 2)],
      haiku: modelsBySize[n - 1],
    };
  });

  let opusModel = $state("");
  let sonnetModel = $state("");
  let haikuModel = $state("");

  $effect(() => {
    opusModel = defaultTiers.opus;
    sonnetModel = defaultTiers.sonnet;
    haikuModel = defaultTiers.haiku;
  });

  let codexModel = $state("");
  let codexMcpPath = $state("/Users/username");
  let openClawModel = $state("");
  $effect(() => {
    const def = modelsBySize.length > 0 ? modelsBySize[0] : "your-model-id";
    codexModel = def;
    openClawModel = def;
  });

  const claudeShellCommand = $derived(
    [
      `ANTHROPIC_BASE_URL=${apiUrl} \\`,
      `ANTHROPIC_API_KEY=x \\`,
      `ANTHROPIC_DEFAULT_OPUS_MODEL=${opusModel} \\`,
      `ANTHROPIC_DEFAULT_SONNET_MODEL=${sonnetModel} \\`,
      `ANTHROPIC_DEFAULT_HAIKU_MODEL=${haikuModel} \\`,
      `API_TIMEOUT_MS=3000000 \\`,
      `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \\`,
      `claude`,
    ].join("\n"),
  );

  const claudeSettingsJson = $derived(
    JSON.stringify(
      {
        env: {
          ANTHROPIC_BASE_URL: apiUrl,
          ANTHROPIC_API_KEY: "x",
          ANTHROPIC_DEFAULT_OPUS_MODEL: opusModel,
          ANTHROPIC_DEFAULT_SONNET_MODEL: sonnetModel,
          ANTHROPIC_DEFAULT_HAIKU_MODEL: haikuModel,
          API_TIMEOUT_MS: "3000000",
          CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: "1",
        },
      },
      null,
      2,
    ),
  );

  const openCodeConfig = $derived.by(() => {
    const models: Record<string, Record<string, unknown>> = {};
    for (const modelId of runningModels) {
      const caps = modelCapabilities[modelId] || [];
      const ctxLen = modelContextLengths[modelId] || 0;
      const entry: Record<string, unknown> = { name: modelId };
      if (ctxLen > 0) {
        entry.limit = { context: ctxLen, output: Math.min(ctxLen, 16384) };
      }
      if (caps.includes("vision")) {
        entry.modalities = { input: ["text", "image"], output: ["text"] };
      }
      models[modelId] = entry;
    }
    if (Object.keys(models).length === 0) {
      models["your-model-id"] = { name: "your-model-name" };
    }
    const firstModel =
      runningModels.length > 0 ? runningModels[0] : "your-model-id";
    return JSON.stringify(
      {
        $schema: "https://opencode.ai/config.json",
        provider: {
          cai: {
            npm: "@ai-sdk/openai-compatible",
            name: "cai",
            options: {
              baseURL: `${apiUrl}/v1`,
              apiKey: "x",
            },
            models,
          },
        },
        model: `cai/${firstModel}`,
      },
      null,
      2,
    );
  });

  const codexShellCommand = $derived(`CAI_API_KEY=x npx @openai/codex`);

  const codexConfig = $derived(
    [
      `model = "${codexModel}"`,
      `model_provider = "cai"`,
      ``,
      `[model_providers.cai]`,
      `name = "cai"`,
      `base_url = "${apiUrl}/v1"`,
      `env_key = "CAI_API_KEY"`,
      ``,
      `[mcp_servers.filesystem]`,
      `command = "npx"`,
      `args = ["-y", "@modelcontextprotocol/server-filesystem", "${codexMcpPath}"]`,
    ].join("\n"),
  );

  const openClawConfig = $derived(
    JSON.stringify(
      {
        gateway: { mode: "local" },
        models: {
          providers: {
            cai: {
              baseUrl: `${apiUrl}/v1`,
              apiKey: "x",
              api: "openai-completions",
              models: [
                {
                  id: openClawModel,
                  name: "cai local",
                  input: (modelCapabilities[openClawModel] || []).includes(
                    "vision",
                  )
                    ? ["text", "image"]
                    : ["text"],
                },
              ],
            },
          },
        },
        agents: {
          defaults: {
            model: `cai/${openClawModel}`,
          },
        },
      },
      null,
      2,
    ),
  );

  const ollamaCommand = $derived(
    `OLLAMA_HOST=${apiUrl}/ollama ollama run ${modelsBySize.length > 0 ? modelsBySize[0] : "your-model-id"}`,
  );

  const openWebUiCommand = $derived(
    [
      `docker run -d -p 3000:8080 \\`,
      `  -e OLLAMA_BASE_URL=${apiUrl.replace("localhost", "host.docker.internal")}/ollama \\`,
      `  -v open-webui:/app/backend/data \\`,
      `  --name open-webui \\`,
      `  ghcr.io/open-webui/open-webui:main`,
    ].join("\n"),
  );

  const n8nDockerCommand = $derived(
    [
      `docker run -d -p 5678:5678 \\`,
      `  -v n8n_data:/home/node/.n8n \\`,
      `  --name n8n \\`,
      `  docker.n8n.io/n8nio/n8n`,
    ].join("\n"),
  );

  const n8nCredentialSteps = $derived(
    [
      tr('1. Go to Credentials -> Add Credential -> search "OpenAI API"'),
      tr("2. Set API Key to: x"),
      trf("3. Set Base URL to: {url}/v1", {
        url: apiUrl
          .replace("127.0.0.1", "host.docker.internal")
          .replace("localhost", "host.docker.internal"),
      }),
      tr("4. Save the credential"),
    ].join("\n"),
  );

  const n8nWorkflowSteps = $derived(
    [
      tr('1. Create a new workflow -> "Start from Scratch"'),
      tr('2. Add an "AI Agent" or "Basic LLM Chain" node'),
      tr('3. Inside it, add an "OpenAI Chat Model" sub-node'),
      tr("4. Select the OpenAI credential you just created"),
      trf('5. Set Model to "From list" and pick your model (e.g. {model})', {
        model: modelsBySize.length > 0 ? modelsBySize[0] : "your-model-id",
      }),
      tr(
        '6. Optionally toggle "Use Responses API", add Built-in Tools, or click "Add Option" for sampling settings',
      ),
      tr('7. Connect a "Chat Trigger" node for interactive chat'),
      tr('8. On the Chat Trigger, enable "Allow File Uploads" for vision'),
    ].join("\n"),
  );

  const firefoxConfig = $derived(
    [
      tr("1. Open about:config in Firefox"),
      tr("2. Set browser.ml.chat.enabled to true"),
      tr("3. Set browser.ml.chat.hideLocalhost to false"),
      trf("4. Set browser.ml.chat.provider to: {url}/", { url: apiUrl }),
    ].join("\n"),
  );

  const tabs = [
    "Claude Code",
    "OpenCode",
    "Codex",
    "OpenClaw",
    "Open WebUI",
    "n8n",
    "Firefox",
  ] as const;
  type Tab = (typeof tabs)[number];
  const stored = browser ? localStorage.getItem("cai-integrations-tab") : null;
  let activeTab = $state<Tab>(
    stored && tabs.includes(stored as Tab) ? (stored as Tab) : "Claude Code",
  );
  $effect(() => {
    if (browser) localStorage.setItem("cai-integrations-tab", activeTab);
  });

  const selectClass =
    "bg-black/30 border border-cai-light-gray/20 rounded px-2 py-1.5 text-white font-mono text-xs focus:border-cai-yellow/50 focus:outline-none appearance-none cursor-pointer";

  onMount(async () => {
    refreshState();
    try {
      const resp = await fetch("/v1/models");
      const data = (await resp.json()) as {
        data: { id: string; capabilities: string[]; context_length: number }[];
      };
      const caps: Record<string, string[]> = {};
      const ctxs: Record<string, number> = {};
      for (const model of data.data) {
        caps[model.id] = model.capabilities || [];
        if (model.context_length > 0) ctxs[model.id] = model.context_length;
      }
      modelCapabilities = caps;
      modelContextLengths = ctxs;
    } catch {
      /* ignore */
    }
  });
</script>

<div class="min-h-screen bg-cai-dark-gray flex flex-col">
  <HeaderNav showHome={true} />

  <main
    class="flex-1 max-w-3xl mx-auto w-full px-4 md:px-6 py-8"
    in:fade={{ duration: 200 }}
  >
    <div class="mb-8">
      <h1
        class="text-white text-xl md:text-2xl font-semibold tracking-wide mb-2"
      >
        {tr("Integrations")}
      </h1>
      <p class="text-cai-light-gray/60 text-sm">
        {tr("Connect external tools to your CAI cluster.")}
      </p>
    </div>

    <!-- Status -->
    <div class="mb-8">
      <span class="text-cai-light-gray/70 text-xs uppercase tracking-wider"
        >{tr("API Endpoint")}</span
      >
      <span class="text-white font-mono text-sm ml-2">{apiUrl}</span>
      {#if runningModels.length > 0}
        <div class="text-cai-light-gray/50 text-xs mt-2">
          {runningModels.length === 1
            ? tr("Running model:")
            : trf("Running models ({count}):", { count: runningModels.length })}
          <ul class="mt-1 space-y-0.5 list-none">
            {#each runningModels as model}
              <li class="text-cai-yellow font-mono">{model}</li>
            {/each}
          </ul>
        </div>
      {:else}
        <p class="text-cai-light-gray/40 text-xs mt-2 italic">
          {tr("No models currently running")}
        </p>
      {/if}
    </div>

    <!-- API Endpoints -->
    <div class="mb-8">
      <div
        class="flex flex-col sm:flex-row gap-3 text-xs font-mono text-cai-light-gray/70"
      >
        <div
          class="flex-1 bg-black/20 border border-cai-light-gray/10 rounded px-3 py-2"
        >
          <span class="text-cai-light-gray/40 text-[10px] uppercase block mb-1"
            >{tr("OpenAI-compatible")}</span
          >
          <span class="text-white/80">{apiUrl}/v1</span>
        </div>
        <div
          class="flex-1 bg-black/20 border border-cai-light-gray/10 rounded px-3 py-2"
        >
          <span class="text-cai-light-gray/40 text-[10px] uppercase block mb-1"
            >{tr("Claude-compatible")}</span
          >
          <span class="text-white/80">{apiUrl}</span>
        </div>
        <div
          class="flex-1 bg-black/20 border border-cai-light-gray/10 rounded px-3 py-2"
        >
          <span class="text-cai-light-gray/40 text-[10px] uppercase block mb-1"
            >{tr("Ollama-compatible")}</span
          >
          <span class="text-white/80">{apiUrl}/ollama</span>
        </div>
      </div>
    </div>

    <!-- Tabs -->
    <div
      class="flex flex-wrap gap-2 mb-6 border-b border-cai-light-gray/10 pb-3"
    >
      {#each tabs as tab}
        <button
          onclick={() => (activeTab = tab)}
          class="px-3 py-1.5 text-xs rounded-md transition-all cursor-pointer
            {activeTab === tab
            ? 'bg-cai-yellow/15 text-cai-yellow border border-cai-yellow/30'
            : 'text-cai-light-gray/60 hover:text-white/80 border border-transparent hover:border-cai-light-gray/20'}"
        >
          {tab}
        </button>
      {/each}
    </div>

    <!-- Tab Content -->
    <div class="space-y-4">
      {#if activeTab === "Claude Code"}
        {#if runningModels.length > 1}
          <div class="grid grid-cols-3 gap-3 text-xs">
            {#each [{ label: "Opus", bind: () => opusModel, set: (v: string) => (opusModel = v) }, { label: "Sonnet", bind: () => sonnetModel, set: (v: string) => (sonnetModel = v) }, { label: "Haiku", bind: () => haikuModel, set: (v: string) => (haikuModel = v) }] as tier}
              <div>
                <span
                  class="text-cai-light-gray/50 text-[10px] uppercase tracking-wider block mb-1"
                  >{tier.label}</span
                >
                <select
                  value={tier.bind()}
                  onchange={(e) =>
                    tier.set((e.target as HTMLSelectElement).value)}
                  class="w-full {selectClass}"
                >
                  {#each runningModels as model}
                    <option value={model}>{model.split("/").pop()}</option>
                  {/each}
                </select>
              </div>
            {/each}
          </div>
        {/if}
        <IntegrationCard
          title={tr("Shell Command")}
          subtitle={tr("Run in terminal")}
          description={tr(
            "Launch Claude Code with CAI as the backend. Paste this into your terminal.",
          )}
          config={claudeShellCommand}
          language="bash"
        />
        <IntegrationCard
          title={tr("Settings File")}
          subtitle="~/.claude/settings.json"
          description={tr(
            "Or add this to your Claude Code settings for persistent configuration.",
          )}
          config={claudeSettingsJson}
        />
      {:else if activeTab === "OpenCode"}
        <IntegrationCard
          title={tr("Config File")}
          subtitle="opencode.json"
          description={tr(
            "Add this to your project root or ~/.config/opencode/opencode.json for global config. Vision models automatically get image input modality.",
          )}
          config={openCodeConfig}
        />
      {:else if activeTab === "Codex"}
        <div class="flex gap-3 text-xs">
          {#if runningModels.length > 1}
            <div>
              <span
                class="text-cai-light-gray/50 text-[10px] uppercase tracking-wider block mb-1"
                >{tr("Model")}</span
              >
              <select bind:value={codexModel} class={selectClass}>
                {#each runningModels as model}
                  <option value={model}>{model.split("/").pop()}</option>
                {/each}
              </select>
            </div>
          {/if}
          <div class="flex-1">
            <span
              class="text-cai-light-gray/50 text-[10px] uppercase tracking-wider block mb-1"
              >{tr("MCP Filesystem Path")}</span
            >
            <input
              type="text"
              bind:value={codexMcpPath}
              class="w-full bg-black/30 border border-cai-light-gray/20 rounded px-2 py-1.5 text-white font-mono text-xs focus:border-cai-yellow/50 focus:outline-none"
            />
          </div>
        </div>
        <IntegrationCard
          title={tr("Config File")}
          subtitle="~/.codex/config.toml"
          description={tr(
            "Add this to your Codex CLI config so the model and provider persist.",
          )}
          config={codexConfig}
        />
        <IntegrationCard
          title={tr("Shell Command")}
          subtitle={tr("Run in terminal")}
          description={tr("Launch Codex with CAI as the backend.")}
          config={codexShellCommand}
          language="bash"
        />
      {:else if activeTab === "OpenClaw"}
        {#if runningModels.length > 1}
          <div class="text-xs">
            <span
              class="text-cai-light-gray/50 text-[10px] uppercase tracking-wider block mb-1"
              >{tr("Model")}</span
            >
            <select bind:value={openClawModel} class={selectClass}>
              {#each runningModels as model}
                <option value={model}>{model.split("/").pop()}</option>
              {/each}
            </select>
          </div>
        {/if}
        <IntegrationCard
          title={tr("Config File")}
          subtitle="~/.openclaw/openclaw.json"
          description={tr(
            "Add this to your OpenClaw config. If you haven't installed OpenClaw yet, run: npm install -g openclaw@latest",
          )}
          config={openClawConfig}
        />
        <IntegrationCard
          title={tr("Setup Commands")}
          subtitle={tr("Run in terminal")}
          description={tr(
            "After saving the config, run these commands to fix metadata and start the gateway.",
          )}
          config={`openclaw doctor --fix${(modelCapabilities[openClawModel] || []).includes("vision") ? `\nopenclaw models set-image cai/${openClawModel}` : ""}\nopenclaw gateway &\nopenclaw dashboard`}
          language="bash"
        />
      {:else if activeTab === "Open WebUI"}
        <IntegrationCard
          title={tr("1. Start Open WebUI")}
          subtitle={tr("Run in terminal")}
          description={tr("Run this to start Open WebUI.")}
          config={openWebUiCommand}
          language="bash"
        />
        <IntegrationCard
          title={tr("2. Open & Select Model")}
          subtitle="http://localhost:3000"
          description={trf(
            "Open http://localhost:3000 in your browser. Select the running model from the dropdown at the top: {models}",
            {
              models:
                runningModels.length > 0
                  ? runningModels.join(", ")
                  : tr("no models running"),
            },
          )}
          config={"open http://localhost:3000"}
          language="bash"
        />
        <IntegrationCard
          title={tr("Ollama CLI")}
          subtitle={tr("Run in terminal")}
          description={tr("Or use the Ollama CLI directly.")}
          config={ollamaCommand}
          language="bash"
        />
      {:else if activeTab === "n8n"}
        <IntegrationCard
          title={tr("1. Start n8n")}
          subtitle={tr("Run in terminal")}
          description={tr(
            "Start n8n with Docker. If you already have n8n running, skip this step.",
          )}
          config={n8nDockerCommand}
          language="bash"
        />
        <IntegrationCard
          title={tr("2. Open n8n")}
          subtitle="http://localhost:5678"
          description={tr(
            "Open n8n in your browser. If this is your first time, complete the setup and select 'Start from Scratch' when prompted.",
          )}
          config={"open http://localhost:5678"}
          language="bash"
        />
        <IntegrationCard
          title={tr("3. Add OpenAI Credential")}
          subtitle={tr("n8n UI -> Credentials")}
          description={tr("Create an OpenAI credential pointing at your CAI cluster.")}
          config={n8nCredentialSteps}
        />
        <IntegrationCard
          title={tr("4. Build a Workflow")}
          subtitle={tr("n8n UI -> Workflows")}
          description={tr("Create a workflow that uses your CAI-powered model.")}
          config={n8nWorkflowSteps}
        />
      {:else if activeTab === "Firefox"}
        <IntegrationCard
          title={tr("Firefox AI Chatbot")}
          subtitle="about:config"
          description={tr(
            "Use the CAI dashboard as Firefox's built-in AI chatbot. Requires Firefox 130+.",
          )}
          config={firefoxConfig}
        />
      {/if}
    </div>
  </main>
</div>
