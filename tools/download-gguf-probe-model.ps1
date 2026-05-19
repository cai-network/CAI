# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
param(
    [Parameter(Mandatory = $true)]
    [string]$RepoId,
    [Parameter(Mandatory = $true)]
    [string]$Filename,
    [string]$Revision = "main",
    [string]$OutputDir = "models",
    [string]$OutputPath = "",
    [int]$TimeoutSeconds = 1800,
    [int]$AttemptDelaySeconds = 5,
    [int]$MaxAttempts = 0,
    [double]$MinimumFreeSpaceMultiplier = 1.15,
    [string]$Token = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Resolve-RepoPath {
    param([string]$Path)

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Path))
}

function Format-ByteSize {
    param([long]$Bytes)

    if ($Bytes -ge 1GB) {
        return ("{0:N2} GiB" -f ($Bytes / 1GB))
    }
    if ($Bytes -ge 1MB) {
        return ("{0:N2} MiB" -f ($Bytes / 1MB))
    }
    if ($Bytes -ge 1KB) {
        return ("{0:N2} KiB" -f ($Bytes / 1KB))
    }
    return ("{0} B" -f $Bytes)
}

function ConvertTo-HfPath {
    param([string]$Path)

    $segments = $Path -split "/"
    $encoded = foreach ($segment in $segments) {
        [System.Uri]::EscapeDataString($segment)
    }
    return ($encoded -join "/")
}

function Get-HfFileMetadata {
    param(
        [string]$RepoId,
        [string]$Revision,
        [string]$Filename,
        [hashtable]$Headers
    )

    $apiUrl = "https://huggingface.co/api/models/$RepoId/revision/$Revision" + "?blobs=true"
    try {
        $modelInfo = Invoke-RestMethod -Method Get -Uri $apiUrl -Headers $Headers
    } catch {
        $fallbackUrl = "https://huggingface.co/api/models/$RepoId" + "?blobs=true"
        $modelInfo = Invoke-RestMethod -Method Get -Uri $fallbackUrl -Headers $Headers
    }

    foreach ($sibling in @($modelInfo.siblings)) {
        $name = [string]$sibling.rfilename
        if ($name -ne $Filename) {
            continue
        }
        $size = 0L
        if ($null -ne $sibling.size) {
            $size = [long]$sibling.size
        } elseif ($null -ne $sibling.lfs -and $null -ne $sibling.lfs.size) {
            $size = [long]$sibling.lfs.size
        }
        return [pscustomobject]@{
            repoId = $RepoId
            revision = $Revision
            filename = $Filename
            sizeBytes = $size
        }
    }
    throw "GGUF file '$Filename' was not found in Hugging Face repo '$RepoId'."
}

function Get-AvailableBytes {
    param([string]$Path)

    $root = [System.IO.Path]::GetPathRoot($Path)
    if ([string]::IsNullOrWhiteSpace($root)) {
        $root = [System.IO.Path]::GetPathRoot((Get-Location).Path)
    }
    $drive = [System.IO.DriveInfo]::new($root)
    return [long]$drive.AvailableFreeSpace
}

function ConvertTo-ProcessArgument {
    param([string]$Argument)

    if ($null -eq $Argument) {
        return '""'
    }
    if ($Argument -notmatch '[\s"]') {
        return $Argument
    }
    return '"' + ($Argument -replace '"', '\"') + '"'
}

function Invoke-CurlAttempt {
    param(
        [string]$Url,
        [string]$OutputPath,
        [string]$Token,
        [int]$TimeoutSeconds
    )

    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($null -eq $curl) {
        throw "curl.exe is required for resumable GGUF downloads."
    }

    $arguments = @(
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--retry", "0",
        "--connect-timeout", "30",
        "--max-time", [string][Math]::Max(1, $TimeoutSeconds),
        "--speed-time", "60",
        "--speed-limit", "1024",
        "-C", "-",
        "-o", $OutputPath
    )
    if (-not [string]::IsNullOrWhiteSpace($Token)) {
        $arguments += @("-H", "Authorization: Bearer $Token")
    }
    $arguments += $Url

    $argumentLine = ($arguments | ForEach-Object { ConvertTo-ProcessArgument $_ }) -join " "
    $process = Start-Process `
        -FilePath $curl.Source `
        -ArgumentList $argumentLine `
        -NoNewWindow `
        -Wait `
        -PassThru
    if ($null -eq $process) {
        throw "Failed to start curl.exe."
    }
    $process.Refresh()
    $exitCode = $process.ExitCode
    if ($null -eq $exitCode) {
        $exitCode = -1
    }
    return [int]$exitCode
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$headers = @{}
$resolvedToken = $Token
if ([string]::IsNullOrWhiteSpace($resolvedToken)) {
    $resolvedToken = [string]$env:HF_TOKEN
}
if (-not [string]::IsNullOrWhiteSpace($resolvedToken)) {
    $headers["Authorization"] = "Bearer $resolvedToken"
}

$metadata = Get-HfFileMetadata `
    -RepoId $RepoId `
    -Revision $Revision `
    -Filename $Filename `
    -Headers $headers

$resolvedOutputDir = Resolve-RepoPath -Path $OutputDir
New-Item -ItemType Directory -Force -Path $resolvedOutputDir | Out-Null
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $leafName = Split-Path -Leaf $Filename
    $OutputPath = Join-Path $resolvedOutputDir $leafName
}
$resolvedOutputPath = Resolve-RepoPath -Path $OutputPath
$outputParent = Split-Path -Parent $resolvedOutputPath
if (-not [string]::IsNullOrWhiteSpace($outputParent)) {
    New-Item -ItemType Directory -Force -Path $outputParent | Out-Null
}
$partialPath = "$resolvedOutputPath.part"
$expectedSize = [long]$metadata.sizeBytes

if ((Test-Path -LiteralPath $partialPath -PathType Leaf) -and
    ((Get-Item -LiteralPath $partialPath).Length -eq 0)) {
    Remove-Item -LiteralPath $partialPath -Force
}

if (Test-Path -LiteralPath $resolvedOutputPath -PathType Leaf) {
    $existingSize = [long](Get-Item -LiteralPath $resolvedOutputPath).Length
    if ($expectedSize -gt 0 -and $existingSize -eq $expectedSize) {
        $summary = [ordered]@{
            status = "already_present"
            repoId = $RepoId
            revision = $Revision
            filename = $Filename
            outputPath = $resolvedOutputPath
            sizeBytes = $existingSize
            size = Format-ByteSize -Bytes $existingSize
            sha256Hex = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolvedOutputPath).Hash.ToLowerInvariant()
        }
        $summary | ConvertTo-Json -Depth 4
        exit 0
    }
    if ($expectedSize -gt 0 -and $existingSize -lt $expectedSize) {
        if (Test-Path -LiteralPath $partialPath -PathType Leaf) {
            throw "Both final and partial files exist; resolve manually: $resolvedOutputPath and $partialPath"
        }
        Move-Item -LiteralPath $resolvedOutputPath -Destination $partialPath
    } else {
        throw "Existing output file has unexpected size: $resolvedOutputPath ($existingSize bytes, expected $expectedSize bytes)."
    }
}

$partialBytes = 0L
if (Test-Path -LiteralPath $partialPath -PathType Leaf) {
    $partialBytes = [long](Get-Item -LiteralPath $partialPath).Length
    if ($expectedSize -gt 0 -and $partialBytes -gt $expectedSize) {
        throw "Partial file is larger than expected: $partialPath"
    }
}

$remainingBytes = if ($expectedSize -gt 0) { [Math]::Max(0L, $expectedSize - $partialBytes) } else { 0L }
$requiredFreeBytes = [long][Math]::Ceiling([double]$remainingBytes * [Math]::Max(1.0, $MinimumFreeSpaceMultiplier))
$availableBytes = Get-AvailableBytes -Path $resolvedOutputPath
if ($requiredFreeBytes -gt 0 -and $availableBytes -lt $requiredFreeBytes) {
    throw "Not enough free space for GGUF download. Required about $(Format-ByteSize -Bytes $requiredFreeBytes), available $(Format-ByteSize -Bytes $availableBytes)."
}

$encodedFilename = ConvertTo-HfPath -Path $Filename
$downloadUrl = "https://huggingface.co/$RepoId/resolve/$Revision/$encodedFilename" + "?download=true"
$preflight = [ordered]@{
    status = if ($DryRun) { "dry_run" } else { "downloading" }
    repoId = $RepoId
    revision = $Revision
    filename = $Filename
    outputPath = $resolvedOutputPath
    partialPath = $partialPath
    sizeBytes = $expectedSize
    size = Format-ByteSize -Bytes $expectedSize
    existingPartialBytes = $partialBytes
    existingPartialSize = Format-ByteSize -Bytes $partialBytes
    remainingBytes = $remainingBytes
    remainingSize = Format-ByteSize -Bytes $remainingBytes
    availableBytes = $availableBytes
    availableSize = Format-ByteSize -Bytes $availableBytes
    timeoutSeconds = $TimeoutSeconds
    attemptDelaySeconds = $AttemptDelaySeconds
    maxAttempts = $MaxAttempts
}
$preflight | ConvertTo-Json -Depth 4
if ($DryRun) {
    exit 0
}

$startedAt = Get-Date
$attempt = 0
$lastPartialBytes = $partialBytes
$downloadComplete = $false
while (-not $downloadComplete) {
    $attempt += 1
    if ($MaxAttempts -gt 0 -and $attempt -gt $MaxAttempts) {
        throw "GGUF download stopped after $MaxAttempts attempts. Kept partial for resume: $partialPath"
    }
    $elapsedSeconds = [int]([DateTime]::UtcNow - $startedAt.ToUniversalTime()).TotalSeconds
    $remainingTimeoutSeconds = [Math]::Max(1, $TimeoutSeconds - $elapsedSeconds)
    if ($elapsedSeconds -ge $TimeoutSeconds) {
        throw "GGUF download timed out after $TimeoutSeconds seconds. Kept partial for resume: $partialPath"
    }

    $currentPartialBytes = 0L
    if (Test-Path -LiteralPath $partialPath -PathType Leaf) {
        $currentPartialBytes = [long](Get-Item -LiteralPath $partialPath).Length
    }
    Write-Output (
        "attempt=$attempt partial=$(Format-ByteSize -Bytes $currentPartialBytes) " +
        "remainingTimeoutSeconds=$remainingTimeoutSeconds"
    )

    $exitCode = Invoke-CurlAttempt `
        -Url $downloadUrl `
        -OutputPath $partialPath `
        -Token $resolvedToken `
        -TimeoutSeconds $remainingTimeoutSeconds

    if ((Test-Path -LiteralPath $partialPath -PathType Leaf) -and
        ((Get-Item -LiteralPath $partialPath).Length -eq 0)) {
        Remove-Item -LiteralPath $partialPath -Force
    }
    if (-not (Test-Path -LiteralPath $partialPath -PathType Leaf)) {
        throw "GGUF download did not create a partial file: $partialPath"
    }

    $downloadedSize = [long](Get-Item -LiteralPath $partialPath).Length
    if ($expectedSize -gt 0 -and $downloadedSize -eq $expectedSize) {
        $downloadComplete = $true
        break
    }
    if ($exitCode -eq 0 -and $expectedSize -le 0) {
        $downloadComplete = $true
        break
    }
    if ($exitCode -eq 0) {
        throw "curl.exe exited successfully but downloaded $downloadedSize bytes; expected $expectedSize bytes."
    }

    $madeProgress = $downloadedSize -gt $lastPartialBytes
    $lastPartialBytes = $downloadedSize
    $elapsedSeconds = [int]([DateTime]::UtcNow - $startedAt.ToUniversalTime()).TotalSeconds
    if ($elapsedSeconds -ge $TimeoutSeconds) {
        throw "GGUF download timed out after $TimeoutSeconds seconds. Kept partial for resume: $partialPath"
    }
    if (-not $madeProgress) {
        Write-Warning "curl.exe failed with exit code $exitCode and made no progress."
    } else {
        Write-Warning (
            "curl.exe failed with exit code $exitCode; kept partial " +
            "$(Format-ByteSize -Bytes $downloadedSize) for resume."
        )
    }
    Start-Sleep -Seconds ([Math]::Max(0, $AttemptDelaySeconds))
}

$downloadedSize = [long](Get-Item -LiteralPath $partialPath).Length
if ($expectedSize -gt 0 -and $downloadedSize -ne $expectedSize) {
    throw "Downloaded partial file size mismatch: $downloadedSize bytes, expected $expectedSize bytes. Kept partial for resume: $partialPath"
}
Move-Item -LiteralPath $partialPath -Destination $resolvedOutputPath -Force

$finalSize = [long](Get-Item -LiteralPath $resolvedOutputPath).Length
$summary = [ordered]@{
    status = "ok"
    repoId = $RepoId
    revision = $Revision
    filename = $Filename
    outputPath = $resolvedOutputPath
    sizeBytes = $finalSize
    size = Format-ByteSize -Bytes $finalSize
    sha256Hex = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolvedOutputPath).Hash.ToLowerInvariant()
}
$summary | ConvertTo-Json -Depth 4
