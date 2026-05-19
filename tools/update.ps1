# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ForwardArgs
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot "cai\.venv-win\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "py"
}

$arguments = @("-m", "cai_compute_chain.cli", "update", "apply")
if ($ForwardArgs.Count -gt 0) {
    $arguments = @("-m", "cai_compute_chain.cli", "update") + $ForwardArgs
}

Push-Location $repoRoot
try {
    & $pythonExe @arguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
