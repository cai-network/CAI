<!--
SPDX-FileCopyrightText: 2025 cai Technologies Ltd
SPDX-FileCopyrightText: 2026 CAI contributors
SPDX-License-Identifier: Apache-2.0
-->
<script lang="ts">
  import {
    isLoading,
    editingImage,
    clearEditingImage,
    selectedChatModel,
    ttftMs,
    tps,
    totalTokens,
    thinkingEnabled as thinkingEnabledStore,
    setConversationThinking,
    stopGeneration,
  } from "$lib/stores/app.svelte";
  import ChatAttachments from "./ChatAttachments.svelte";
  import ImageParamsPanel from "./ImageParamsPanel.svelte";
  import type { ChatUploadedFile } from "$lib/types/files";
  import { processUploadedFiles, getAcceptString } from "$lib/types/files";
  import { tr } from "$lib/stores/i18n.svelte";

  interface Props {
    class?: string;
    placeholder?: string;
    showHelperText?: boolean;
    autofocus?: boolean;
    showModelSelector?: boolean;
    modelTasks?: Record<string, string[]>;
    modelCapabilities?: Record<string, string[]>;
    onSend?: () => void;
    onAutoSend: (
      content: string,
      files?: {
        id: string;
        name: string;
        type: string;
        textContent?: string;
        preview?: string;
      }[],
    ) => boolean | void | Promise<boolean | void>;
    onOpenModelPicker?: () => void;
    modelDisplayOverride?: string;
    draftValue?: string;
    onDraftChange?: (value: string) => void;
  }

  let {
    class: className = "",
    placeholder = "Ask anything",
    showHelperText = false,
    autofocus = true,
    showModelSelector = false,
    modelTasks = {},
    modelCapabilities = {},
    onSend,
    onAutoSend,
    onOpenModelPicker,
    modelDisplayOverride,
    draftValue,
    onDraftChange,
  }: Props = $props();

  let message = $state("");
  let lastAppliedDraftValue: string | undefined;
  let textareaRef: HTMLTextAreaElement | undefined = $state();
  let fileInputRef: HTMLInputElement | undefined = $state();
  let uploadedFiles = $state<ChatUploadedFile[]>([]);
  let isDragOver = $state(false);
  let submitInFlight = $state(false);
  const thinkingEnabled = $derived(thinkingEnabledStore());
  let loading = $derived(isLoading());
  const currentModel = $derived(selectedChatModel());
  const currentTtft = $derived(ttftMs());
  const currentTps = $derived(tps());
  const currentTokens = $derived(totalTokens());
  const currentEditingImage = $derived(editingImage());
  const isEditMode = $derived(currentEditingImage !== null);

  // Accept all supported file types
  const acceptString = getAcceptString(["image", "text", "pdf"]);

  function modelSupportsImageGeneration(modelId: string): boolean {
    const tasks = modelTasks[modelId] || [];
    return tasks.includes("TextToImage") || tasks.includes("ImageToImage");
  }

  function modelSupportsTextToImage(modelId: string): boolean {
    const tasks = modelTasks[modelId] || [];
    return tasks.includes("TextToImage");
  }

  function modelSupportsOnlyImageEditing(modelId: string): boolean {
    const tasks = modelTasks[modelId] || [];
    return tasks.includes("ImageToImage") && !tasks.includes("TextToImage");
  }

  function modelSupportsImageEditing(modelId: string): boolean {
    const tasks = modelTasks[modelId] || [];
    return tasks.includes("ImageToImage");
  }

  const isImageModel = $derived(() => {
    if (!currentModel) return false;
    return (
      modelSupportsTextToImage(currentModel) ||
      modelSupportsImageEditing(currentModel)
    );
  });

  const modelSupportsThinking = $derived(() => {
    if (!currentModel) return false;
    const caps = modelCapabilities[currentModel] || [];
    return caps.includes("thinking_toggle") && caps.includes("text");
  });

  const isEditOnlyWithoutImage = $derived(
    currentModel !== null &&
      modelSupportsOnlyImageEditing(currentModel) &&
      !isEditMode &&
      uploadedFiles.length === 0,
  );

  // Show edit mode when: explicit edit mode OR (model supports ImageToImage AND files attached)
  const shouldShowEditMode = $derived(
    isEditMode ||
      (currentModel &&
        modelSupportsImageEditing(currentModel) &&
        uploadedFiles.length > 0),
  );

  // Short label for the currently selected model
  const currentModelLabel = $derived(
    currentModel
      ? currentModel.split("/").pop() || currentModel
      : modelDisplayOverride
        ? modelDisplayOverride.split("/").pop() || modelDisplayOverride
        : "",
  );

  async function handleFiles(files: File[]) {
    if (files.length === 0) return;
    const processed = await processUploadedFiles(files);
    uploadedFiles = [...uploadedFiles, ...processed];
  }

  function handleFileInputChange(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      handleFiles(Array.from(input.files));
      input.value = ""; // Reset for next selection
    }
  }

  function handleFileRemove(fileId: string) {
    uploadedFiles = uploadedFiles.filter((f) => f.id !== fileId);
  }

  function handlePaste(event: ClipboardEvent) {
    if (!event.clipboardData) return;

    const files = Array.from(event.clipboardData.items)
      .filter((item) => item.kind === "file")
      .map((item) => item.getAsFile())
      .filter((file): file is File => file !== null);

    if (files.length > 0) {
      event.preventDefault();
      handleFiles(files);
      return;
    }

    // Handle long text paste as file
    const text = event.clipboardData.getData("text/plain");
    if (text.length > 2500) {
      event.preventDefault();
      const textFile = new File([text], "pasted-text.txt", {
        type: "text/plain",
      });
      handleFiles([textFile]);
    }
  }

  function handleDragOver(event: DragEvent) {
    event.preventDefault();
    isDragOver = true;
  }

  function handleDragLeave(event: DragEvent) {
    event.preventDefault();
    isDragOver = false;
  }

  function handleDrop(event: DragEvent) {
    event.preventDefault();
    isDragOver = false;

    if (event.dataTransfer?.files) {
      handleFiles(Array.from(event.dataTransfer.files));
    }
  }

  function handleKeydown(event: KeyboardEvent) {
    // Prevent form submission during IME composition (e.g., Chinese, Japanese, Korean input)
    if (event.isComposing || event.keyCode === 229) {
      return;
    }

    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSubmit();
    }
  }

  async function handleSubmit() {
    if (
      (!message.trim() && uploadedFiles.length === 0) ||
      loading ||
      submitInFlight
    ) {
      return;
    }
    if (isEditOnlyWithoutImage) return;

    const content = message.trim();
    const files = [...uploadedFiles];

    submitInFlight = true;
    let accepted: boolean | void;
    try {
      accepted = await onAutoSend(content, files);
    } catch (error) {
      console.error("Chat submit failed", error);
      accepted = false;
    } finally {
      submitInFlight = false;
    }

    if (accepted === false) {
      lastAppliedDraftValue = content;
      onDraftChange?.(content);
      setTimeout(() => textareaRef?.focus(), 10);
      return;
    }

    message = "";
    lastAppliedDraftValue = "";
    onDraftChange?.("");
    uploadedFiles = [];
    resetTextareaHeight();
    onSend?.();
    setTimeout(() => textareaRef?.focus(), 10);
  }

  function handleInput(event?: Event) {
    if (event?.currentTarget instanceof HTMLTextAreaElement) {
      message = event.currentTarget.value;
    }
    lastAppliedDraftValue = message;
    onDraftChange?.(message);
    resizeTextarea();
  }

  function resizeTextarea() {
    if (!textareaRef) return;
    textareaRef.style.height = "auto";
    textareaRef.style.height = Math.min(textareaRef.scrollHeight, 150) + "px";
  }

  function resetTextareaHeight() {
    if (textareaRef) {
      textareaRef.style.height = "auto";
    }
  }

  function openFilePicker() {
    fileInputRef?.click();
  }

  // Track previous loading state to detect when loading completes
  let wasLoading = $state(false);

  $effect(() => {
    if (autofocus && textareaRef) {
      setTimeout(() => textareaRef?.focus(), 10);
    }
  });

  $effect(() => {
    const nextDraftValue = draftValue;
    if (
      nextDraftValue !== undefined &&
      nextDraftValue !== lastAppliedDraftValue
    ) {
      lastAppliedDraftValue = nextDraftValue;
      message = nextDraftValue;
      setTimeout(resizeTextarea, 0);
    }
  });

  // Refocus after loading completes (AI response finished)
  $effect(() => {
    if (wasLoading && !loading && textareaRef) {
      setTimeout(() => textareaRef?.focus(), 50);
    }
    wasLoading = loading;
  });

  const canSend = $derived(
    !submitInFlight && (message.trim().length > 0 || uploadedFiles.length > 0),
  );
</script>

<!-- Hidden file input -->
<input
  bind:this={fileInputRef}
  type="file"
  accept={acceptString}
  multiple
  class="hidden"
  onchange={handleFileInputChange}
/>

<form
  onsubmit={(e) => {
    e.preventDefault();
    handleSubmit();
  }}
  class="w-full {className}"
  ondragover={handleDragOver}
  ondragleave={handleDragLeave}
  ondrop={handleDrop}
>
  <div
    class="relative command-panel rounded overflow-hidden transition-all duration-200 {isDragOver
      ? 'ring-2 ring-cai-yellow ring-opacity-50'
      : ''}"
  >
    <!-- Top accent line -->
    <div
      class="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-cai-yellow/50 to-transparent"
    ></div>

    <!-- Drag overlay -->
    {#if isDragOver}
      <div
        class="absolute inset-0 bg-cai-dark-gray/80 z-10 flex items-center justify-center"
      >
        <div class="text-cai-yellow text-sm font-mono tracking-wider uppercase">
          {tr("DROP FILES HERE")}
        </div>
      </div>
    {/if}

    <!-- Edit mode banner -->
    {#if isEditMode && currentEditingImage}
      <div
        class="flex items-center gap-3 px-3 py-2 bg-cai-yellow/10 border-b border-cai-yellow/30"
      >
        <img
          src={currentEditingImage.imageDataUrl}
          alt={tr("Edit image")}
          class="w-10 h-10 object-cover rounded border border-cai-yellow/30"
        />
        <div class="flex-1">
          <span
            class="text-xs font-mono tracking-wider uppercase text-cai-yellow"
            >{tr("EDITING IMAGE")}</span
          >
        </div>
        <button
          type="button"
          onclick={() => clearEditingImage()}
          class="px-2 py-1 text-xs font-mono tracking-wider uppercase bg-cai-medium-gray/30 text-cai-light-gray border border-cai-medium-gray/50 rounded hover:bg-cai-medium-gray/50 hover:text-cai-yellow transition-colors cursor-pointer"
        >
          {tr("CANCEL")}
        </button>
      </div>
    {/if}

    <!-- Model selector (when enabled) -->
    {#if showModelSelector}
      <div
        class="flex items-center justify-between gap-2 px-3 py-2 border-b border-cai-medium-gray/30"
      >
        <div class="flex items-center gap-2 flex-1">
          <span
            class="text-xs text-cai-light-gray uppercase tracking-wider flex-shrink-0"
            >{tr("MODEL:")}</span
          >
          <!-- Model button — opens the full model picker -->
          <div class="relative flex-1 max-w-xs">
            <button
              type="button"
              onclick={() => onOpenModelPicker?.()}
              class="w-full bg-cai-medium-gray/50 border border-cai-yellow/30 rounded pl-3 pr-8 py-1.5 text-xs font-mono text-left tracking-wide cursor-pointer transition-all duration-200 hover:border-cai-yellow/50 focus:outline-none focus:border-cai-yellow/70"
            >
              {#if currentModelLabel}
                <span class="text-cai-yellow truncate">{currentModelLabel}</span
                >
              {:else}
                <span class="text-cai-light-gray/50">- {tr("Select a model")} -</span>
              {/if}
            </button>
            <div
              class="absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none"
            >
              <svg
                class="w-3 h-3 text-cai-yellow/60"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M19 9l-7 7-7-7"
                />
              </svg>
            </div>
          </div>
        </div>
        <!-- Thinking toggle -->
        {#if modelSupportsThinking()}
          <button
            type="button"
            onclick={() => setConversationThinking(!thinkingEnabled)}
            class="flex items-center gap-1.5 px-2 py-1 rounded text-xs font-mono tracking-wide transition-all duration-200 flex-shrink-0 cursor-pointer border {thinkingEnabled
              ? 'bg-cai-yellow/15 border-cai-yellow/40 text-cai-yellow'
              : 'bg-cai-medium-gray/30 border-cai-medium-gray/50 text-cai-light-gray/60 hover:text-cai-light-gray'}"
            title={thinkingEnabled
              ? tr("Thinking enabled - click to disable")
              : tr("Thinking disabled - click to enable")}
          >
            <svg
              class="w-3.5 h-3.5"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="1.5"
            >
              <path
                d="M12 2a7 7 0 0 0-7 7c0 2.38 1.19 4.47 3 5.74V17a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-2.26c1.81-1.27 3-3.36 3-5.74a7 7 0 0 0-7-7zM9 20h6M10 22h4"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
            </svg>
            <span>{thinkingEnabled ? tr("THINK") : "NO THINK"}</span>
          </button>
        {/if}

        <!-- Performance stats -->
        {#if currentTtft !== null || currentTps !== null}
          <div class="flex items-center gap-4 text-xs font-mono flex-shrink-0">
            {#if currentTtft !== null}
              <span class="text-cai-light-gray">
                <span class="text-white/70">TTFT</span>
                <span class="text-cai-yellow">{currentTtft.toFixed(1)}ms</span>
              </span>
            {/if}
            {#if currentTps !== null}
              <span class="text-cai-light-gray">
                <span class="text-white/70">TPS</span>
                <span class="text-cai-yellow">{currentTps.toFixed(1)}</span>
                <span class="text-white/60">tok/s</span>
                <span class="text-white/50"
                  >({(1000 / currentTps).toFixed(1)} ms/tok)</span
                >
              </span>
            {/if}
          </div>
        {/if}
      </div>
    {/if}

    <!-- Image params panel (shown for image models or edit mode) -->
    {#if showModelSelector && (isImageModel() || isEditMode)}
      <ImageParamsPanel {isEditMode} />
    {/if}

    <!-- Attached files preview -->
    {#if uploadedFiles.length > 0}
      <div class="px-3 pt-3">
        <ChatAttachments files={uploadedFiles} onRemove={handleFileRemove} />
      </div>
    {/if}

    <!-- Input area -->
    <div class="flex items-start gap-2 sm:gap-3 py-3 px-3 sm:px-4">
      <!-- Attach file button -->
      <button
        type="button"
        onclick={openFilePicker}
        disabled={loading}
        class="flex items-center justify-center w-7 h-7 rounded text-cai-light-gray hover:text-cai-yellow transition-all disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0 cursor-pointer"
        title={tr("Attach file")}
      >
        <svg
          class="w-4 h-4"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"
          />
        </svg>
      </button>

      <!-- Terminal prompt -->
      <span class="text-cai-yellow text-sm font-bold flex-shrink-0 leading-7"
        >▶</span
      >

      <textarea
        bind:this={textareaRef}
        bind:value={message}
        onkeydown={handleKeydown}
        oninput={handleInput}
        onpaste={handlePaste}
        placeholder={isEditOnlyWithoutImage
          ? tr("Attach an image to edit...")
          : isEditMode
            ? tr("Describe how to edit this image...")
            : isImageModel()
              ? tr("Describe the image you want to generate...")
              : tr(placeholder)}
        rows={1}
        class="flex-1 resize-none bg-transparent text-foreground placeholder:text-cai-light-gray/60 placeholder:text-sm placeholder:tracking-[0.15em] placeholder:leading-7 focus:outline-none focus:ring-0 focus:border-none text-sm leading-7 font-mono"
        style="min-height: 28px; max-height: 150px;"
      ></textarea>

      {#if loading}
        <button
          type="button"
          onclick={() => stopGeneration()}
          class="px-2.5 sm:px-4 py-1.5 sm:py-2 rounded text-xs sm:text-xs tracking-[0.1em] sm:tracking-[0.15em] font-medium transition-all duration-200 whitespace-nowrap bg-cai-medium-gray/70 text-cai-light-gray hover:bg-cai-medium-gray hover:text-white"
          aria-label={tr("Stop generation")}
        >
          <span class="inline-flex items-center gap-1 sm:gap-2">
            <svg
              class="w-3 h-3 sm:w-3.5 sm:h-3.5"
              fill="currentColor"
              viewBox="0 0 24 24"
            >
              <rect x="6" y="6" width="12" height="12" rx="1" />
            </svg>
            <span class="hidden sm:inline">{tr("Cancel")}</span>
          </span>
        </button>
      {:else}
        <button
          type="submit"
          disabled={!canSend || submitInFlight || isEditOnlyWithoutImage}
          class="px-2.5 sm:px-4 py-1.5 sm:py-2 rounded text-xs sm:text-xs tracking-[0.1em] sm:tracking-[0.15em] uppercase font-medium transition-all duration-200 whitespace-nowrap
					{!canSend || submitInFlight || isEditOnlyWithoutImage
            ? 'bg-cai-medium-gray/50 text-cai-light-gray cursor-not-allowed'
            : 'bg-cai-yellow text-cai-black hover:bg-cai-yellow-darker hover:shadow-[0_0_20px_rgba(125,211,252,0.3)]'}"
          aria-label={shouldShowEditMode
            ? tr("Edit image")
            : isImageModel()
              ? tr("Generate image")
              : tr("Send message")}
        >
          {#if shouldShowEditMode}
            <span class="inline-flex items-center gap-1.5">
              <svg
                class="w-3.5 h-3.5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                stroke-width="2"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                />
              </svg>
              <span>{tr("EDIT")}</span>
            </span>
          {:else if isEditOnlyWithoutImage}
            <span class="inline-flex items-center gap-1.5">
              <svg
                class="w-3.5 h-3.5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                stroke-width="2"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                />
              </svg>
              <span>{tr("EDIT")}</span>
            </span>
          {:else if isImageModel()}
            <span class="inline-flex items-center gap-1.5">
              <svg
                class="w-3.5 h-3.5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                stroke-width="2"
              >
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                <circle cx="8.5" cy="8.5" r="1.5" />
                <polyline points="21 15 16 10 5 21" />
              </svg>
              <span>{tr("GENERATE")}</span>
            </span>
          {:else}
            {tr("SEND")}
          {/if}
        </button>
      {/if}
    </div>

    <!-- Bottom accent line -->
    <div
      class="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-cai-yellow/30 to-transparent"
    ></div>
  </div>

  {#if showHelperText}
    <p
      class="mt-2 sm:mt-3 text-center text-xs sm:text-xs text-cai-light-gray tracking-[0.1em] sm:tracking-[0.15em] uppercase"
    >
      <kbd
        class="px-1 sm:px-1.5 py-0.5 rounded bg-cai-medium-gray/30 text-cai-light-gray border border-cai-medium-gray/50"
        >{tr("ENTER")}</kbd
      >
      <span class="mx-0.5 sm:mx-1">{tr("TO SEND")}</span>
      <span class="text-cai-medium-gray mx-1 sm:mx-2">|</span>
      <kbd
        class="px-1 sm:px-1.5 py-0.5 rounded bg-cai-medium-gray/30 text-cai-light-gray border border-cai-medium-gray/50"
        >{tr("SHIFT+ENTER")}</kbd
      >
      <span class="mx-0.5 sm:mx-1">{tr("NEW LINE")}</span>
      <span class="text-cai-medium-gray mx-1 sm:mx-2">|</span>
      <span class="text-cai-light-gray">{tr("DRAG & DROP OR PASTE FILES")}</span>
    </p>
  {/if}
</form>
