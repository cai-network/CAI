# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
param(
    [string]$Distro = "Ubuntu-24.04",
    [string]$Ref = "master",
    [string]$PatchDir = "",
    [string]$SourceArchive = "",
    [string]$BuildTarget = "",
    [string]$WslRuntimeRoot = "",
    [switch]$SkipSystemPackages,
    [switch]$AllowEmptyPatchSet,
    [switch]$CpuOnly,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$arguments = @(
    "-Distro", $Distro,
    "-Ref", $Ref
)

if (-not [string]::IsNullOrWhiteSpace($PatchDir)) {
    $arguments += @("-PatchDir", $PatchDir)
}
if (-not [string]::IsNullOrWhiteSpace($SourceArchive)) {
    $arguments += @("-SourceArchive", $SourceArchive)
}
if (-not [string]::IsNullOrWhiteSpace($BuildTarget)) {
    $arguments += @("-BuildTarget", $BuildTarget)
}
if (-not [string]::IsNullOrWhiteSpace($WslRuntimeRoot)) {
    $arguments += @("-WslRuntimeRoot", $WslRuntimeRoot)
}
if ($SkipSystemPackages) {
    $arguments += "-SkipSystemPackages"
}
if ($CpuOnly) {
    $arguments += "-CpuOnly"
}
if ($Clean) {
    $arguments += "-Clean"
}
if (-not $AllowEmptyPatchSet) {
    $arguments += "-RequireCaiPatches"
}

$installer = Join-Path $PSScriptRoot "install-llama-cpp-wsl.ps1"
& powershell -ExecutionPolicy Bypass -File $installer @arguments
exit $LASTEXITCODE
