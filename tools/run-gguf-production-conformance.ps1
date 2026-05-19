# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,
    [Parameter(Mandatory = $true)]
    [string]$ModelId,
    [string]$Architecture = "",
    [int]$TotalLayers = 0,
    [string]$CatalogId = "",
    [string]$Version = "",
    [string]$Quantization = "",
    [string]$OutputReport = "",
    [string]$PythonExe = "",
    [string]$EngineExe = "",
    [string]$WalletDataDirname = "",
    [int]$TargetChunkCount = 4,
    [int]$TimeoutSec = 900
)

$ErrorActionPreference = "Stop"

function Resolve-RepoPath {
    param([string]$Path)

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Path))
}

function ConvertTo-RepoRelativePath {
    param([string]$Path)

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $repoRootFull = [System.IO.Path]::GetFullPath($script:repoRoot).TrimEnd('\', '/')
    if ($fullPath.StartsWith("$repoRootFull\", [System.StringComparison]::OrdinalIgnoreCase)) {
        return $fullPath.Substring($repoRootFull.Length + 1) -replace "\\", "/"
    }
    return $fullPath -replace "\\", "/"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$modelFullPath = Resolve-RepoPath -Path $ModelPath
if (-not (Test-Path -LiteralPath $modelFullPath -PathType Leaf)) {
    throw "GGUF model file not found: $modelFullPath"
}

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $venvPython = Join-Path $repoRoot ".venv-desktop\Scripts\python.exe"
    $PythonExe = if (Test-Path -LiteralPath $venvPython -PathType Leaf) { $venvPython } else { "python" }
}
if ([string]::IsNullOrWhiteSpace($EngineExe)) {
    $EngineExe = Join-Path $repoRoot "cai\.runtime\llama.cpp\windows-patched\build\bin\Release\llama-cai-shard-engine.exe"
}
$engineFullPath = Resolve-RepoPath -Path $EngineExe
if (-not (Test-Path -LiteralPath $engineFullPath -PathType Leaf)) {
    throw "llama-cai-shard-engine executable not found: $engineFullPath"
}

$cleanArch = $Architecture.Trim().ToLowerInvariant()
if ([string]::IsNullOrWhiteSpace($cleanArch)) {
    $cleanArch = "gguf"
}
if ([string]::IsNullOrWhiteSpace($CatalogId)) {
    $CatalogId = "cai-private-$cleanArch-local"
}
if ([string]::IsNullOrWhiteSpace($Version)) {
    $Version = (Get-Date -Format "yyyy.MM.dd") + "-production"
}

$prepareArgs = @(
    (Join-Path $repoRoot "tools\prepare-gguf-model-package.py"),
    "--model-path", $modelFullPath,
    "--model-id", $ModelId,
    "--catalog-id", $CatalogId,
    "--version", $Version,
    "--target-chunk-count", [string]$TargetChunkCount,
    "--cache-chunks",
    "--pin-chunks"
)
if (-not [string]::IsNullOrWhiteSpace($Architecture)) {
    $prepareArgs += @("--architecture", $Architecture)
}
if ($TotalLayers -gt 0) {
    $prepareArgs += @("--total-layers", [string]$TotalLayers)
}
if (-not [string]::IsNullOrWhiteSpace($Quantization)) {
    $prepareArgs += @("--quantization", $Quantization)
}
if (-not [string]::IsNullOrWhiteSpace($WalletDataDirname)) {
    $prepareArgs += @("--wallet-data-dirname", $WalletDataDirname)
}

$prepareOutput = & $PythonExe @prepareArgs
if ($LASTEXITCODE -ne 0) {
    throw "prepare-gguf-model-package failed with exit code $LASTEXITCODE"
}
$prepare = ($prepareOutput | Out-String).Trim() | ConvertFrom-Json
$resolvedArchitecture = if (-not [string]::IsNullOrWhiteSpace([string]$prepare.family)) {
    [string]$prepare.family
} elseif (-not [string]::IsNullOrWhiteSpace($Architecture)) {
    $Architecture
} elseif (-not [string]::IsNullOrWhiteSpace([string]$prepare.ggufArchitecture)) {
    [string]$prepare.ggufArchitecture
} else {
    throw "GGUF architecture is required or must be discoverable from GGUF metadata."
}
$resolvedTotalLayers = [int]$prepare.totalLayers
if ($resolvedTotalLayers -le 0) {
    throw "GGUF total layer count is required or must be discoverable from GGUF metadata."
}

$oldEnv = @{}
foreach ($name in @(
    "PYTHONPATH",
    "CAI_LLM_SHARD_ADAPTER",
    "CAI_LLM_SHARD_ADAPTER_TIMEOUT_SEC",
    "CAI_LLM_SHARD_NATIVE_COMMAND",
    "CAI_LLM_SHARD_NATIVE_TIMEOUT_SEC",
    "CAI_LLM_SHARD_ASSIGNMENT_EXECUTOR_COMMAND",
    "CAI_LLM_SHARD_ASSIGNMENT_EXECUTOR_PERSISTENT",
    "CAI_LLM_SHARD_PATCHED_ENGINE_COMMAND",
    "CAI_LLM_SHARD_PATCHED_ENGINE_PERSISTENT",
    "CAI_LLM_PATCHED_BINARY_COMMAND",
    "CAI_LLM_PATCHED_BINARY_PERSISTENT",
    "CAI_LLM_PATCHED_BINARY_REQUIRE_REAL_LAYER_EXECUTION",
    "CAI_LLM_PATCHED_BINARY_REQUIRE_SHARD_ONLY_LOADING"
)) {
    $oldEnv[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

$runtimeMetadataFile = ""
try {
    $srcPath = Join-Path $repoRoot "src"
    $env:PYTHONPATH = if ([string]::IsNullOrWhiteSpace($oldEnv["PYTHONPATH"])) {
        $srcPath
    } else {
        "$srcPath;$($oldEnv["PYTHONPATH"])"
    }
    $env:CAI_LLM_SHARD_ADAPTER = "native_bridge"
    $env:CAI_LLM_SHARD_ADAPTER_TIMEOUT_SEC = [string]$TimeoutSec
    $env:CAI_LLM_SHARD_NATIVE_TIMEOUT_SEC = [string]$TimeoutSec
    $quotedPython = "`"$PythonExe`""
    $quotedEngine = "`"$engineFullPath`""
    $env:CAI_LLM_SHARD_NATIVE_COMMAND = "$quotedPython -m cai_compute_chain.cai_llama_cpp_assignment_artifact_engine"
    $env:CAI_LLM_SHARD_ASSIGNMENT_EXECUTOR_COMMAND = "$quotedPython -m cai_compute_chain.cai_llama_cpp_patched_executor_host --jsonl"
    $env:CAI_LLM_SHARD_ASSIGNMENT_EXECUTOR_PERSISTENT = "1"
    $env:CAI_LLM_SHARD_PATCHED_ENGINE_COMMAND = "$quotedPython -m cai_compute_chain.cai_llama_cpp_patched_binary_executor --jsonl"
    $env:CAI_LLM_SHARD_PATCHED_ENGINE_PERSISTENT = "1"
    $env:CAI_LLM_PATCHED_BINARY_COMMAND = "$quotedEngine --jsonl"
    $env:CAI_LLM_PATCHED_BINARY_PERSISTENT = "1"
    $env:CAI_LLM_PATCHED_BINARY_REQUIRE_REAL_LAYER_EXECUTION = "1"
    $env:CAI_LLM_PATCHED_BINARY_REQUIRE_SHARD_ONLY_LOADING = "1"

    $runtimeMetadata = [ordered]@{
        totalLayerCount = $resolvedTotalLayers
        candidateGgufArchitecture = $resolvedArchitecture
        shardCompatibility = "layer_range_supported"
        layerRangeSupported = $true
        metadataSource = "gguf-production-conformance-runner"
    } | ConvertTo-Json -Compress
    $runtimeMetadataFile = Join-Path ([System.IO.Path]::GetTempPath()) (
        "cai-gguf-runtime-metadata-" + [System.Guid]::NewGuid().ToString("N") + ".json"
    )
    [System.IO.File]::WriteAllText(
        $runtimeMetadataFile,
        $runtimeMetadata + [Environment]::NewLine,
        (New-Object System.Text.UTF8Encoding $false)
    )

    $conformanceArgs = @(
        "-m", "cai_compute_chain.cai_llm_shard_conformance",
        "--model-id", $ModelId,
        "--payload", "CAI GGUF production conformance",
        "--runtime-metadata-json-file", $runtimeMetadataFile,
        "--require-production"
    )
    $conformanceOutput = & $PythonExe @conformanceArgs
    if ($LASTEXITCODE -ne 0) {
        throw "cai_llm_shard_conformance failed with exit code $LASTEXITCODE`: $(($conformanceOutput | Out-String).Trim())"
    }
    $conformance = ($conformanceOutput | Out-String).Trim() | ConvertFrom-Json
} finally {
    if (-not [string]::IsNullOrWhiteSpace($runtimeMetadataFile) -and (Test-Path -LiteralPath $runtimeMetadataFile)) {
        Remove-Item -LiteralPath $runtimeMetadataFile -Force
    }
    foreach ($entry in $oldEnv.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
    }
}

$modelHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $modelFullPath).Hash.ToLowerInvariant()
$binaryHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $engineFullPath).Hash.ToLowerInvariant()
$report = [ordered]@{
    schemaVersion = 1
    reportKind = "gguf_production_binary_conformance"
    recordedAt = (Get-Date).ToUniversalTime().ToString("o")
    result = [ordered]@{
        status = $conformance.status
        ok = [bool]$conformance.ok
        errors = @($conformance.errors)
    }
    checks = $conformance.checks
    model = [ordered]@{
        modelId = $ModelId
        catalogId = $CatalogId
        version = $Version
        family = $resolvedArchitecture
        quantization = $Quantization
        ggufFile = ConvertTo-RepoRelativePath -Path $modelFullPath
        ggufSizeBytes = (Get-Item -LiteralPath $modelFullPath).Length
        ggufSha256Hex = $modelHash
        totalLayers = $resolvedTotalLayers
        manifestChunkCount = [int]$prepare.manifestChunkCount
        cachedChunkCount = [int]$prepare.cachedChunkCount
    }
    productionBinary = [ordered]@{
        path = ConvertTo-RepoRelativePath -Path $engineFullPath
        sha256Hex = $binaryHash
        sizeBytes = (Get-Item -LiteralPath $engineFullPath).Length
    }
    strictGuards = [ordered]@{
        requireProduction = $true
        requireRealLayerExecution = $true
        requireShardOnlyLoading = $true
        forbidFullModelFallback = $true
    }
    executionPath = @(
        "ExternalLlamaCppShardAdapter",
        "cai_llama_cpp_shard_native_bridge",
        "cai_llama_cpp_assignment_artifact_engine",
        "cai_llama_cpp_patched_executor_host",
        "cai_llama_cpp_patched_binary_executor",
        "llama-cai-shard-engine.exe"
    )
    selfTest = $conformance.selfTest
}

if (-not [string]::IsNullOrWhiteSpace($OutputReport)) {
    $outputPath = Resolve-RepoPath -Path $OutputReport
    $outputParent = Split-Path -Parent $outputPath
    if (-not [string]::IsNullOrWhiteSpace($outputParent)) {
        New-Item -ItemType Directory -Force -Path $outputParent | Out-Null
    }
    $json = $report | ConvertTo-Json -Depth 24
    [System.IO.File]::WriteAllText(
        $outputPath,
        $json + [Environment]::NewLine,
        (New-Object System.Text.UTF8Encoding $false)
    )
    Write-Output "report=$outputPath"
} else {
    $report | ConvertTo-Json -Depth 24
}
