# CAI llama.cpp patches

This directory is the tracked patch pipeline for the CAI-owned patched
`llama.cpp` backend.

Patch files listed in `series` can be applied by the Windows-native builder:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\build-llama-cpp-patched-win.ps1 -SourceArchive .\cai\.runtime\llama.cpp\archives\llama.cpp-f3e8d14.tar.gz -Clean
```

The WSL builder is still available for Linux-compatible binaries:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\build-llama-cpp-patched-wsl.ps1
```

The wrapper requires at least one CAI patch by default, so a "patched" build
cannot silently become an upstream-only build.

For CPU-only probe builds, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\build-llama-cpp-patched-wsl.ps1 -CpuOnly -SkipSystemPackages
```

To build only the shard equivalence probe:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\build-llama-cpp-patched-wsl.ps1 -CpuOnly -SkipSystemPackages -BuildTarget llama-cai-shard-probe
```

To build only the CAI JSONL shard engine:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\build-llama-cpp-patched-win.ps1 -SourceArchive .\cai\.runtime\llama.cpp\archives\llama.cpp-f3e8d14.tar.gz -BuildTarget llama-cai-shard-engine
```

If WSL networking is unstable, download a `llama.cpp` source archive on the
Windows side and pass it with `-SourceArchive`:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\build-llama-cpp-patched-wsl.ps1 -CpuOnly -SkipSystemPackages -BuildTarget llama-cai-shard-probe -SourceArchive .\cai\.runtime\llama.cpp\archives\llama.cpp-f3e8d14.tar.gz
```

For faster and more stable WSL builds, keep the source/build tree on the Linux
filesystem:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\build-llama-cpp-patched-wsl.ps1 -CpuOnly -SkipSystemPackages -BuildTarget llama-cai-shard-probe -SourceArchive .\cai\.runtime\llama.cpp\archives\llama.cpp-f3e8d14.tar.gz -WslRuntimeRoot /tmp/cai-llama-cpp-wsl
```

Current patch set:

- `0001-cai-qwen3-layer-range-graph.patch` adds the first Qwen3 compute-side
  hook: an env-controlled graph layer window that can stop at a layer boundary
  and expose raw activation, or continue from activation input for a later
  layer range.
- `0002-cai-qwen3-shard-probe-tool.patch` adds `llama-cai-shard-probe`, a
  local equivalence probe for `full logits` vs
  `layers 0..split -> activation -> layers split..N -> logits`.
- `0003-cai-qwen3-layer-range-output-fix.patch` makes intermediate layer
  boundaries use the normal output-row selection path and keeps graph reserve
  compatible with activation-input continuation.
- `0004-cai-qwen3-shard-engine-command.patch` adds
  `llama-cai-shard-engine`, a first CAI JSON/JSONL binary command over the
  checked GGUF layer-range path. It can run `load_shard`, `process_prefill`,
  `process_decode`, and `finalize` against the
  `cai-llama-cpp-patched-binary-request-v1` ABI.
- `0005-cai-llama-layer-range-graph.patch` adds the same env-controlled
  layer-range boundary hook to the Llama graph. TinyLlama now has both local
  equivalence and production binary conformance reports, so `llama` is
  registered as a checked `layer_range_supported` architecture in
  `GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS`.
  Reports:
  `docs/reports/llama-tinyllama-layer-range-equivalence-probe-2026-05-11.json`
  and
  `docs/reports/llama-tinyllama-production-binary-conformance-2026-05-11.json`.
- `0006-cai-qwen2-layer-range-graph.patch` adds the env-controlled layer-range
  boundary hook to the dense Qwen2/Qwen2.5 graph. Qwen2.5 now has passing
  equivalence and production binary conformance reports, so `qwen2` is admitted
  through `GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS`.
- `0007-cai-qwen-layer-range-graph.patch` adds the same compute-side boundary
  hook to the original dense Qwen graph. Qwen 1.8B now has passing equivalence
  and production binary conformance reports, so `qwen` is admitted through
  `GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS`.
- `0008-cai-mistral3-layer-range-graph.patch` adds the env-controlled
  layer-range boundary hook to the dense Mistral3 graph. Ministral 3 3B now has
  passing equivalence and production binary conformance reports, so `mistral3`
  is admitted through `GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS`. `mistral4`
  remains separate because it inherits a different graph path.
- `0009-cai-gemma-layer-range-graph.patch` adds the env-controlled layer-range
  boundary hook to the original dense Gemma graph. Gemma 2B now has passing
  equivalence and production binary conformance reports, so `gemma` is admitted
  through `GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS`. `gemma3`, `gemma3n`, and
  `gemma4` remain separate architectures.
- `0010-cai-gemma2-layer-range-graph.patch` adds the same layer-range hook to
  the dense Gemma2 graph, including the Gemma/Gemma2 continuation rule that
  skips token-embedding scaling when resuming from an activation boundary and
  the ISWA-only early-split graph-leaf guard, covered by splitLayer=1/2
  diagnostic probes. Gemma2 2B IT now has passing equivalence and production
  binary conformance reports, so `gemma2` is admitted through
  `GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS`.
- `0011-cai-gemma3-layer-range-graph.patch` adds the layer-range hook to the
  templated Gemma3 graph for both SWA and non-SWA instantiations. It preserves
  the Gemma-family embedding scaling continuation rule and the ISWA graph-leaf
  guard for early splits. Gemma3 1B IT now has passing splitLayer=13
  equivalence, splitLayer=1/2 diagnostics, and production binary conformance,
  so `gemma3` is admitted through `GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS`.
- `0012-cai-phi3-layer-range-graph.patch` adds the same env-controlled
  layer-range boundary hook to the dense Phi3 graph. Phi-3 mini 4k instruct
  now has passing splitLayer=16 equivalence and production binary conformance,
  so `phi3` is admitted through `GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS`.
- `0013-cai-phi2-layer-range-graph.patch` adds the same env-controlled
  layer-range boundary hook to the dense Phi2 graph. Phi-2 Q2_K now has passing
  splitLayer=16 equivalence and production binary conformance, so `phi2` is
  admitted through `GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS`.
- `0014-cai-falcon-layer-range-graph.patch` adds the same env-controlled
  layer-range boundary hook to the dense Falcon graph. This is compute-side
  hook coverage only for now; `falcon` stays outside the whitelist until a
  real `general.architecture=falcon` GGUF has passing equivalence and
  production conformance reports.
- `0015-cai-gptneox-layer-range-graph.patch` adds the same env-controlled
  layer-range boundary hook to the dense GPT-NeoX graph, including its parallel
  residual branch. Pythia 14M now has passing splitLayer=3 equivalence and
  production binary conformance, so `gptneox` is admitted through
  `GGUF_LAYER_RANGE_ARCHITECTURE_PROOFS`.

Probe command after patched build:

```powershell
.\cai\.runtime\llama.cpp\windows-patched\build\bin\Release\llama-cai-shard-probe.exe -m .\models\Qwen3-0.6B-Q8_0.gguf --split-layer 14 --tolerance 0.0001 --model-id cai-network/Qwen3-0.6B-GGUF --architecture qwen3 --gguf-sha256 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031 --legacy-probe-abi cai-qwen3-layer-range-v1
```

Canonical report helper:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\run-gguf-layer-range-probe.ps1 -ModelPath .\models\Qwen3-0.6B-Q8_0.gguf -ModelId cai-network/Qwen3-0.6B-GGUF -Architecture qwen3 -SplitLayer 14 -LegacyProbeAbi cai-qwen3-layer-range-v1 -OutputReport .\docs\reports\qwen3-layer-range-equivalence-probe-2026-05-10.json
```

Known-good local probe result with `Qwen3-0.6B-Q8_0.gguf`:

```json
{"status":"ok","probeAbi":"cai-layer-range-v1","legacyProbeAbi":"cai-qwen3-layer-range-v1","model":{"modelId":"cai-network/Qwen3-0.6B-GGUF","architecture":"qwen3","ggufFile":".\\models\\Qwen3-0.6B-Q8_0.gguf","ggufSizeBytes":639446688,"ggufSha256Hex":"9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031"},"execution":{"deviceMode":"cpu"},"layerRange":{"totalLayers":28,"splitLayer":14,"tokenCount":5,"nEmbedding":1024,"activationShape":[5,1024],"activationFloatCount":5120},"result":{"realLayerExecution":true,"fullTopToken":12095,"shardedTopToken":12095,"topTokenMatch":true,"maxAbsDiff":0,"meanAbsDiff":0,"tolerance":0.0001}}
```

Engine command after patched build:

```powershell
.\cai\.runtime\llama.cpp\windows-patched\build\bin\Release\llama-cai-shard-engine.exe --jsonl
```

Current engine scope:

- speaks the patched binary request ABI and returns `realLayerExecution=true`;
- writes CAI real-state manifests for raw GGUF layer-range activation handoff;
- can run in proving-mode with a full local GGUF model artifact for local math
  checks;
- can run in strict shard-only mode when `requireShardOnlyLoading=true`: it
  loads `assignmentArtifact.localPath`, validates sparse-full-size coverage and
  chunk hashes, and reports `shardOnlyLoadingReady=true`.
