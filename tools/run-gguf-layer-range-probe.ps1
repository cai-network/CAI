# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,
    [Parameter(Mandatory = $true)]
    [string]$ModelId,
    [string]$Architecture = "",
    [int]$SplitLayer = 0,
    [string]$Prompt = "The capital of France is",
    [double]$Tolerance = 0.0001,
    [string]$PythonExe = "",
    [string]$ProbeExe = "",
    [string]$OutputReport = "",
    [string]$LegacyProbeAbi = ""
)

$ErrorActionPreference = "Stop"

function Resolve-RepoPath {
    param([string]$Path)

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Path))
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
if ([string]::IsNullOrWhiteSpace($ProbeExe)) {
    $ProbeExe = Join-Path $repoRoot "cai\.runtime\llama.cpp\windows-patched\build\bin\Release\llama-cai-shard-probe.exe"
}
$probeFullPath = Resolve-RepoPath -Path $ProbeExe
if (-not (Test-Path -LiteralPath $probeFullPath -PathType Leaf)) {
    throw "llama-cai-shard-probe executable not found: $probeFullPath"
}

$resolvedArchitecture = $Architecture.Trim()
$resolvedSplitLayer = [int]$SplitLayer
if ([string]::IsNullOrWhiteSpace($resolvedArchitecture) -or $resolvedSplitLayer -le 0) {
    $inspectCode = @'
import json
import sys

sys.path.insert(0, sys.argv[1])
from cai_compute_chain.model_distribution import read_gguf_model_metadata

metadata = read_gguf_model_metadata(sys.argv[2])
print(json.dumps(dict(
    architecture=metadata.architecture,
    totalLayers=metadata.total_layers,
), ensure_ascii=False))
'@
    $inspectOutput = & $PythonExe -c $inspectCode (Join-Path $repoRoot "src") $modelFullPath
    if ($LASTEXITCODE -ne 0) {
        throw "GGUF metadata inspection failed with exit code $LASTEXITCODE`: $(($inspectOutput | Out-String).Trim())"
    }
    $metadata = ($inspectOutput | Out-String).Trim() | ConvertFrom-Json
    if ([string]::IsNullOrWhiteSpace($resolvedArchitecture)) {
        $resolvedArchitecture = [string]$metadata.architecture
    }
    if ($resolvedSplitLayer -le 0) {
        $totalLayers = [int]$metadata.totalLayers
        if ($totalLayers -le 1) {
            throw "SplitLayer is required because GGUF total layer count is missing or too small."
        }
        $resolvedSplitLayer = [Math]::Floor($totalLayers / 2)
    }
}
if ([string]::IsNullOrWhiteSpace($resolvedArchitecture)) {
    throw "Architecture is required or must be discoverable from GGUF metadata."
}

$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $modelFullPath).Hash.ToLowerInvariant()
$argsList = @(
    "-m", $modelFullPath,
    "--split-layer", [string]$resolvedSplitLayer,
    "--tolerance", [string]$Tolerance,
    "--model-id", $ModelId,
    "--architecture", $resolvedArchitecture,
    "--gguf-sha256", $hash,
    "--prompt", $Prompt
)
if (-not [string]::IsNullOrWhiteSpace($LegacyProbeAbi)) {
    $argsList += @("--legacy-probe-abi", $LegacyProbeAbi)
}

$probeOutput = & $probeFullPath @argsList
if ($LASTEXITCODE -ne 0) {
    $renderedOutput = ($probeOutput | Out-String).Trim()
    throw "llama-cai-shard-probe failed with exit code ${LASTEXITCODE}: $renderedOutput"
}
if ($probeOutput.Count -ne 1) {
    throw "llama-cai-shard-probe returned unexpected multi-line output."
}

$report = $probeOutput | ConvertFrom-Json
$report | Add-Member -NotePropertyName "schemaVersion" -NotePropertyValue 1 -Force
$report | Add-Member -NotePropertyName "reportKind" -NotePropertyValue "gguf_layer_range_equivalence_probe" -Force

$repoRootFullPath = [System.IO.Path]::GetFullPath($repoRoot).TrimEnd('\', '/')
if ($modelFullPath.StartsWith("$repoRootFullPath\", [System.StringComparison]::OrdinalIgnoreCase)) {
    $report.model.ggufFile = $modelFullPath.Substring($repoRootFullPath.Length + 1) -replace "\\", "/"
}
if ($null -eq $report.execution) {
    $report | Add-Member -NotePropertyName "execution" -NotePropertyValue ([pscustomobject]@{}) -Force
}
$report.execution | Add-Member -NotePropertyName "prompt" -NotePropertyValue $Prompt -Force

if (-not [string]::IsNullOrWhiteSpace($OutputReport)) {
    $outputPath = Resolve-RepoPath -Path $OutputReport
    $outputParent = Split-Path -Parent $outputPath
    if (-not [string]::IsNullOrWhiteSpace($outputParent)) {
        New-Item -ItemType Directory -Force -Path $outputParent | Out-Null
    }
    $json = $report | ConvertTo-Json -Depth 12
    [System.IO.File]::WriteAllText(
        $outputPath,
        $json + [Environment]::NewLine,
        (New-Object System.Text.UTF8Encoding $false)
    )
    Write-Output "report=$outputPath"
} else {
    $report | ConvertTo-Json -Depth 12
}
