# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ForwardArgs
)

$ScriptPath = Join-Path $PSScriptRoot "bootstrap.py"

if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($Version in @("-3.14", "-3.13", "-3")) {
        & py $Version -c "import sys" *> $null
        if ($LASTEXITCODE -eq 0) {
            & py $Version $ScriptPath @ForwardArgs
            exit $LASTEXITCODE
        }
    }
}

if (Get-Command python -ErrorAction SilentlyContinue) {
    & python $ScriptPath @ForwardArgs
    exit $LASTEXITCODE
}

Write-Error "Python launcher not found. Install Python 3 and rerun tools\\bootstrap.ps1"
exit 1
