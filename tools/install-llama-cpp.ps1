# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
param(
    [ValidateSet("windows", "wsl")]
    [string]$Target = "windows",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ForwardArgs
)

$ErrorActionPreference = "Stop"

$scriptName = "install-llama-cpp-$Target.ps1"
$scriptPath = Join-Path $PSScriptRoot $scriptName
if (-not (Test-Path $scriptPath)) {
    throw "Install script not found: $scriptPath"
}

& powershell -ExecutionPolicy Bypass -File $scriptPath @ForwardArgs
exit $LASTEXITCODE
