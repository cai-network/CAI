# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
param(
    [string]$Version = "latest",

    [ValidateSet("12.4", "13.1")]
    [string]$CudaVariant = "12.4",

    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Get-LlamaCppRelease {
    param([string]$RequestedVersion)

    if ($RequestedVersion -eq "latest") {
        return Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
    }

    return Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/$RequestedVersion"
}

function Find-ReleaseAsset {
    param(
        [object]$Release,
        [string]$Pattern
    )

    $asset = $Release.assets | Where-Object { $_.name -like $Pattern } | Select-Object -First 1
    if (-not $asset) {
        throw "Unable to find release asset matching '$Pattern' in release '$($Release.tag_name)'."
    }
    return $asset
}

function Download-Asset {
    param(
        [object]$Asset,
        [string]$DestinationPath,
        [switch]$ForceDownload
    )

    $parent = Split-Path -Parent $DestinationPath
    if (-not (Test-Path $parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }

    $expectedSize = [int64]$Asset.size
    if ((Test-Path $DestinationPath) -and -not $ForceDownload) {
        $existingSize = (Get-Item -LiteralPath $DestinationPath).Length
        if ($existingSize -eq $expectedSize) {
            Write-Output "[llama.cpp] reusing $DestinationPath"
            return
        }
        Write-Output "[llama.cpp] resuming partial download $($Asset.name) ($existingSize / $expectedSize bytes)"
    } elseif ((Test-Path $DestinationPath) -and $ForceDownload) {
        Remove-Item -LiteralPath $DestinationPath -Force
    }

    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        for ($attempt = 1; $attempt -le 10; $attempt++) {
            $currentSize = 0
            if (Test-Path $DestinationPath) {
                $currentSize = (Get-Item -LiteralPath $DestinationPath).Length
                if ($currentSize -gt $expectedSize) {
                    Write-Output "[llama.cpp] removing oversized partial download $($Asset.name)"
                    Remove-Item -LiteralPath $DestinationPath -Force
                    $currentSize = 0
                }
            }

            Write-Output "[llama.cpp] downloading $($Asset.name) attempt=$attempt current=$currentSize expected=$expectedSize"
            & $curl.Source `
                -L `
                --fail `
                --retry 5 `
                --retry-delay 3 `
                --retry-all-errors `
                --connect-timeout 30 `
                --continue-at - `
                --output $DestinationPath `
                $Asset.browser_download_url
            $exitCode = $LASTEXITCODE
            $actualSize = if (Test-Path $DestinationPath) { (Get-Item -LiteralPath $DestinationPath).Length } else { 0 }
            if ($exitCode -eq 0 -and $actualSize -eq $expectedSize) {
                Write-Output "[llama.cpp] download complete $($Asset.name)"
                return
            }
            if ($attempt -eq 10) {
                throw "Failed to download $($Asset.name): curl exit=$exitCode actual=$actualSize expected=$expectedSize"
            }
            Start-Sleep -Seconds ([Math]::Min(20, 2 * $attempt))
        }
    }

    if (Test-Path $DestinationPath) {
        Remove-Item -LiteralPath $DestinationPath -Force
    }
    Write-Output "[llama.cpp] downloading $($Asset.name)"
    Invoke-WebRequest -Uri $Asset.browser_download_url -OutFile $DestinationPath
    $downloadedSize = (Get-Item -LiteralPath $DestinationPath).Length
    if ($downloadedSize -ne $expectedSize) {
        throw "Downloaded $($Asset.name) has unexpected size: $downloadedSize expected $expectedSize"
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$runtimeRoot = Join-Path $repoRoot "cai\.runtime\llama.cpp\windows"
$downloadRoot = Join-Path $runtimeRoot "downloads"
$buildRoot = Join-Path $runtimeRoot "build"
$manifestPath = Join-Path $runtimeRoot "installed-release.json"

$release = Get-LlamaCppRelease -RequestedVersion $Version
$binaryAsset = Find-ReleaseAsset -Release $release -Pattern "llama-*-bin-win-cuda-$CudaVariant-x64.zip"
$dllAsset = Find-ReleaseAsset -Release $release -Pattern "cudart-llama-bin-win-cuda-$CudaVariant-x64.zip"

$releaseDownloadRoot = Join-Path $downloadRoot $release.tag_name
$binaryZipPath = Join-Path $releaseDownloadRoot $binaryAsset.name
$dllZipPath = Join-Path $releaseDownloadRoot $dllAsset.name

Download-Asset -Asset $binaryAsset -DestinationPath $binaryZipPath -ForceDownload:$Force
Download-Asset -Asset $dllAsset -DestinationPath $dllZipPath -ForceDownload:$Force

if (Test-Path $buildRoot) {
    Remove-Item -LiteralPath $buildRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null

Write-Output "[llama.cpp] extracting $($binaryAsset.name)"
Expand-Archive -Path $binaryZipPath -DestinationPath $buildRoot -Force
Write-Output "[llama.cpp] extracting $($dllAsset.name)"
Expand-Archive -Path $dllZipPath -DestinationPath $buildRoot -Force

$llamaServer = Get-ChildItem -Path $buildRoot -Recurse -Filter "llama-server.exe" | Select-Object -First 1
$rpcServer = Get-ChildItem -Path $buildRoot -Recurse -Filter "rpc-server.exe" | Select-Object -First 1

if (-not $llamaServer) {
    throw "llama-server.exe was not found after extraction into $buildRoot"
}

$manifest = [ordered]@{
    installed_at = (Get-Date).ToString("o")
    tag = $release.tag_name
    cuda_variant = $CudaVariant
    binary_asset = $binaryAsset.name
    dll_asset = $dllAsset.name
    llama_server = $llamaServer.FullName
    rpc_server = $rpcServer.FullName
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Path $manifestPath

Write-Output ""
Write-Output "[llama.cpp] install complete"
Write-Output "tag=$($release.tag_name)"
Write-Output "cuda_variant=$CudaVariant"
Write-Output "llama_server=$($llamaServer.FullName)"
if ($rpcServer) {
    Write-Output "rpc_server=$($rpcServer.FullName)"
}
Write-Output ""
Write-Output "[llama.cpp] next steps"
Write-Output "  1. python .\\tools\\run-cai-main.py"
Write-Output "  2. For Linux/VPS validators: bash ./tools/join-mainnet-validator.sh"
