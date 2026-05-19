# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ForwardArgs
)

$ScriptPath = Join-Path $PSScriptRoot "bootstrap.ps1"
& powershell -ExecutionPolicy Bypass -File $ScriptPath @ForwardArgs
exit $LASTEXITCODE
