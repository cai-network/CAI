# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
param(
    [string]$Path = ".dist\CAI-portable.zip"
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not [System.IO.Path]::IsPathRooted($Path)) {
    $Path = Join-Path $RepoRoot $Path
}
$ResolvedPath = [System.IO.Path]::GetFullPath($Path)
if (-not (Test-Path -LiteralPath $ResolvedPath)) {
    throw "Portable artifact not found: $ResolvedPath"
}

$ForbiddenPatterns = @(
    '^data/',
    '^\.cai-api-token$',
    '^\.cai-peer-book\.json$',
    '^\.cai/',
    '^\.cai-local/',
    '^\.cai-local-testnet/',
    '^secrets/',
    '^wallets\.json$',
    '^session\.json$',
    '^ledger\.json$',
    '^chain\.json$',
    '^journal\.jsonl$',
    '^node-config\.json$',
    '^settlements\.json$',
    '^worker-payouts\.json$',
    '^job-intents\.json$',
    '^execution-receipts\.json$',
    '^unlocked-wallet-signing-key\.json$',
    '(^|/)(developer-treasury|ai-development)-(seed|password)\.txt$',
    '^cai_log/',
    '^desktop\.log$'
)

function Test-ForbiddenPath {
    param([Parameter(Mandatory = $true)][string]$EntryPath)
    $normalized = $EntryPath.Replace('\', '/').TrimStart('/')
    foreach ($pattern in $ForbiddenPatterns) {
        if ($normalized -match $pattern) {
            return $true
        }
    }
    return $false
}

$violations = @()
if ((Get-Item -LiteralPath $ResolvedPath).PSIsContainer) {
    $root = (Resolve-Path -LiteralPath $ResolvedPath).Path
    $violations = Get-ChildItem -LiteralPath $root -Recurse -Force |
        ForEach-Object {
            $relative = $_.FullName.Substring($root.Length).TrimStart('\', '/')
            if (Test-ForbiddenPath $relative) {
                $relative
            }
        }
} else {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::OpenRead($ResolvedPath)
    try {
        $violations = $archive.Entries |
            ForEach-Object {
                if (Test-ForbiddenPath $_.FullName) {
                    $_.FullName
                }
            }
    } finally {
        $archive.Dispose()
    }
}

$violations = @($violations | Sort-Object -Unique)
if ($violations.Count -gt 0) {
    $preview = ($violations | Select-Object -First 30) -join "`n"
    throw "Portable artifact contains runtime state:`n$preview"
}

Write-Host "Portable artifact is clean: $ResolvedPath"
