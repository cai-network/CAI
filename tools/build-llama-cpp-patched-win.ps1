# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
param(
    [string]$Ref = "f3e8d14",
    [string]$PatchDir = "",
    [string]$SourceArchive = "",
    [string]$BuildTarget = "llama-cai-shard-probe",
    [string]$RuntimeRoot = "",
    [string]$Generator = "Visual Studio 18 2026",
    [string]$Platform = "x64",
    [switch]$EnableCuda,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Assert-PathInside {
    param(
        [string]$Path,
        [string]$Parent
    )

    $resolvedParent = [System.IO.Path]::GetFullPath($Parent).TrimEnd('\', '/')
    $resolvedPath = [System.IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    if (-not ($resolvedPath -eq $resolvedParent -or $resolvedPath.StartsWith("$resolvedParent\", [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "Refusing to operate outside '$resolvedParent': $resolvedPath"
    }
}

function Invoke-Native {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$WorkingDirectory = ""
    )

    if ([string]::IsNullOrWhiteSpace($WorkingDirectory)) {
        & $FilePath @Arguments
    } else {
        & $FilePath @Arguments 2>&1
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

function Move-DirectoryWithRetry {
    param(
        [string]$Source,
        [string]$Destination,
        [int]$Attempts = 5
    )

    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        try {
            Move-Item -LiteralPath $Source -Destination $Destination -ErrorAction Stop
            return
        } catch {
            if ($attempt -ge $Attempts) {
                throw
            }
            Start-Sleep -Milliseconds (250 * $attempt)
        }
    }
}

function Test-GitApply {
    param(
        [string]$SourceRoot,
        [string]$PatchPath,
        [switch]$Reverse
    )

    $arguments = @("-C", $SourceRoot, "apply", "--check")
    if ($Reverse) {
        $arguments += "--reverse"
    }
    $arguments += $PatchPath

    & git @arguments *> $null
    return $LASTEXITCODE -eq 0
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedRuntimeRoot = if ([string]::IsNullOrWhiteSpace($RuntimeRoot)) {
    Join-Path $repoRoot "cai\.runtime\llama.cpp\windows-patched"
} else {
    $RuntimeRoot
}
$resolvedRuntimeRoot = [System.IO.Path]::GetFullPath($resolvedRuntimeRoot)
$repoRootFull = [System.IO.Path]::GetFullPath($repoRoot)
Assert-PathInside -Path $resolvedRuntimeRoot -Parent $repoRootFull

$defaultPatchDir = Join-Path $repoRoot "patches\llama.cpp"
$resolvedPatchDir = if ([string]::IsNullOrWhiteSpace($PatchDir)) { $defaultPatchDir } else { $PatchDir }
$resolvedPatchDir = (Resolve-Path -LiteralPath $resolvedPatchDir).Path

$sourceRoot = Join-Path $resolvedRuntimeRoot "source"
$buildRoot = Join-Path $resolvedRuntimeRoot "build"
$downloadRoot = Join-Path $resolvedRuntimeRoot "downloads"
$manifestPath = Join-Path $resolvedRuntimeRoot "installed-patched-source.json"

New-Item -ItemType Directory -Force -Path $resolvedRuntimeRoot, $downloadRoot | Out-Null

if ($Clean) {
    Assert-PathInside -Path $sourceRoot -Parent $resolvedRuntimeRoot
    Assert-PathInside -Path $buildRoot -Parent $resolvedRuntimeRoot
    if (Test-Path $sourceRoot) {
        Remove-Item -LiteralPath $sourceRoot -Recurse -Force
    }
    if (Test-Path $buildRoot) {
        Remove-Item -LiteralPath $buildRoot -Recurse -Force
    }
}

$resolvedSourceArchive = ""
if (-not [string]::IsNullOrWhiteSpace($SourceArchive)) {
    $resolvedSourceArchive = (Resolve-Path -LiteralPath $SourceArchive).Path
} else {
    $resolvedSourceArchive = Join-Path $downloadRoot "llama.cpp-$Ref.tar.gz"
    if (-not (Test-Path $resolvedSourceArchive)) {
        $archiveUrl = "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/$Ref"
        Write-Output "[llama.cpp] downloading source archive $archiveUrl"
        Invoke-WebRequest -Uri $archiveUrl -OutFile $resolvedSourceArchive -MaximumRedirection 5 -TimeoutSec 600
    }
}

Write-Output "[llama.cpp] validating source archive"
& tar.exe -tzf $resolvedSourceArchive *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Invalid source archive: $resolvedSourceArchive"
}

if (-not (Test-Path (Join-Path $sourceRoot ".git"))) {
    $extractRoot = Join-Path $resolvedRuntimeRoot "extract"
    Assert-PathInside -Path $extractRoot -Parent $resolvedRuntimeRoot
    if (Test-Path $extractRoot) {
        Remove-Item -LiteralPath $extractRoot -Recurse -Force
    }
    if (Test-Path $sourceRoot) {
        Remove-Item -LiteralPath $sourceRoot -Recurse -Force
    }

    New-Item -ItemType Directory -Force -Path $extractRoot | Out-Null
    Write-Output "[llama.cpp] extracting source archive"
    Invoke-Native -FilePath "tar.exe" -Arguments @("-xzf", $resolvedSourceArchive, "-C", $extractRoot)

    $extractedRoot = Get-ChildItem -LiteralPath $extractRoot -Directory | Select-Object -First 1
    if (-not $extractedRoot) {
        throw "Source archive did not contain an extracted directory."
    }

    Move-DirectoryWithRetry -Source $extractedRoot.FullName -Destination $sourceRoot
    Remove-Item -LiteralPath $extractRoot -Recurse -Force

    Invoke-Native -FilePath "git" -Arguments @("-C", $sourceRoot, "init", "-q")
    Invoke-Native -FilePath "git" -Arguments @("-C", $sourceRoot, "config", "core.longpaths", "true")
    Invoke-Native -FilePath "git" -Arguments @("-C", $sourceRoot, "config", "core.autocrlf", "false")
    Invoke-Native -FilePath "git" -Arguments @("-C", $sourceRoot, "config", "user.email", "cai-build@example.invalid")
    Invoke-Native -FilePath "git" -Arguments @("-C", $sourceRoot, "config", "user.name", "CAI build")
    Invoke-Native -FilePath "git" -Arguments @("-C", $sourceRoot, "add", "-A")
    Invoke-Native -FilePath "git" -Arguments @("-C", $sourceRoot, "commit", "-q", "-m", "CAI upstream archive $Ref")
}

Invoke-Native -FilePath "git" -Arguments @("-C", $sourceRoot, "reset", "--hard", "HEAD")
Invoke-Native -FilePath "git" -Arguments @("-C", $sourceRoot, "clean", "-fd")

$seriesPath = Join-Path $resolvedPatchDir "series"
$patches = @()
if (Test-Path $seriesPath) {
    $patches = Get-Content -Path $seriesPath | Where-Object {
        $line = $_.Trim()
        -not [string]::IsNullOrWhiteSpace($line) -and -not $line.StartsWith("#")
    } | ForEach-Object { Join-Path $resolvedPatchDir $_.Trim() }
} else {
    $patches = Get-ChildItem -LiteralPath $resolvedPatchDir -Filter "*.patch" | Sort-Object Name | ForEach-Object { $_.FullName }
}

if (-not $patches) {
    throw "No CAI patch files found in $resolvedPatchDir"
}

$appliedPatches = @()
foreach ($patch in $patches) {
    if (-not (Test-Path $patch)) {
        throw "Patch file not found: $patch"
    }

    $patchName = Split-Path -Leaf $patch
    if (Test-GitApply -SourceRoot $sourceRoot -PatchPath $patch) {
        Write-Output "[llama.cpp] applying CAI patch $patchName"
        Invoke-Native -FilePath "git" -Arguments @("-C", $sourceRoot, "apply", $patch)
        $appliedPatches += "${patchName}:applied"
    } elseif (Test-GitApply -SourceRoot $sourceRoot -PatchPath $patch -Reverse) {
        Write-Output "[llama.cpp] CAI patch already applied $patchName"
        $appliedPatches += "${patchName}:already_applied"
    } else {
        throw "Failed to apply CAI patch $patchName"
    }
}

$cmakeAccelFlags = if ($EnableCuda) {
    @("-DGGML_CUDA=ON", "-DGGML_RPC=ON")
} else {
    @("-DGGML_CUDA=OFF", "-DGGML_RPC=ON")
}

New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null
Write-Output "[llama.cpp] configuring patched Windows build"
Invoke-Native -FilePath "cmake" -Arguments (@(
    "-S", $sourceRoot,
    "-B", $buildRoot,
    "-G", $Generator,
    "-A", $Platform
) + $cmakeAccelFlags)

$buildArgs = @("--build", $buildRoot, "--config", "Release")
if (-not [string]::IsNullOrWhiteSpace($BuildTarget)) {
    $buildArgs += @("--target", $BuildTarget)
}
$buildArgs += @("-j", [string]([Environment]::ProcessorCount))

Write-Output "[llama.cpp] building patched Windows target $BuildTarget"
Invoke-Native -FilePath "cmake" -Arguments $buildArgs

$probeExe = Get-ChildItem -Path $buildRoot -Recurse -Filter "llama-cai-shard-probe.exe" | Select-Object -First 1
$engineExe = Get-ChildItem -Path $buildRoot -Recurse -Filter "llama-cai-shard-engine.exe" | Select-Object -First 1
$manifest = [ordered]@{
    installed_at = (Get-Date).ToString("o")
    ref = $Ref
    source_root = $sourceRoot
    build_root = $buildRoot
    patch_dir = $resolvedPatchDir
    patch_count = $patches.Count
    applied_patches = $appliedPatches
    cuda_enabled = [bool]$EnableCuda
    generator = $Generator
    platform = $Platform
    build_target = $BuildTarget
    shard_probe = if ($probeExe) { $probeExe.FullName } else { $null }
    shard_engine = if ($engineExe) { $engineExe.FullName } else { $null }
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -Path $manifestPath

Write-Output "[llama.cpp] patched Windows build complete"
Write-Output "manifest=$manifestPath"
if ($probeExe) {
    Write-Output "shard_probe=$($probeExe.FullName)"
}
if ($engineExe) {
    Write-Output "shard_engine=$($engineExe.FullName)"
}
