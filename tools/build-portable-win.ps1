# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
param(
    [string]$OutputDir = "",
    [switch]$SkipDashboardBuild,
    [switch]$Zip,
    [switch]$PreserveData,
    [switch]$NoStopRunningOutputProcesses
)

$ErrorActionPreference = "Stop"
Set-Variable -Name PSNativeCommandUseErrorActionPreference -Value $false -Scope Script -ErrorAction SilentlyContinue

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $RepoRoot ".dist\CAI-portable"
} elseif (-not [System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $RepoRoot $OutputDir
}

$DistRoot = Join-Path $RepoRoot ".dist"
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
$DistRootFull = [System.IO.Path]::GetFullPath($DistRoot)
$PythonExe = Join-Path $RepoRoot "cai\.venv-win\Scripts\python.exe"
$BuildDir = Join-Path $RepoRoot "build\cai-portable"
$SpecPath = Join-Path $RepoRoot "packaging\pyinstaller\cai-portable.spec"
$IconPath = Join-Path $BuildDir "cai.ico"
$PreservedDataBackup = Join-Path $DistRootFull "_cai-portable-data-backup"
$PortableDistName = "_cai-portable-stage"
$PortableStageDir = Join-Path $DistRootFull $PortableDistName
$PortableRuntimeStateRelativePaths = @(
    "data",
    ".cai",
    ".cai-local",
    ".cai-local-testnet",
    ".cai-api-token",
    ".cai-peer-book.json",
    "cai_log",
    "desktop.log",
    "event_log",
    "wallets.json",
    "session.json",
    "ledger.json",
    "chain.json",
    "journal.jsonl",
    "node-config.json",
    "settlements.json",
    "worker-payouts.json",
    "job-intents.json",
    "execution-receipts.json",
    "unlocked-wallet-signing-key.json"
)

function Remove-PortableRuntimeState {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PortableRoot
    )

    foreach ($relativePath in $PortableRuntimeStateRelativePaths) {
        $target = Join-Path $PortableRoot $relativePath
        if (Test-Path $target) {
            Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

function Remove-PortableRuntimeData {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PortableRoot
    )

    Remove-PortableRuntimeState -PortableRoot $PortableRoot
}

function Remove-PortableDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    if (-not (Test-Path $Path)) {
        return
    }

    Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
    if (Test-Path $Path) {
        throw "Unable to clean $Description directory: $Path"
    }
}

function Get-PortableOutputProcesses {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PortableRoot
    )

    $rootFull = [System.IO.Path]::GetFullPath($PortableRoot).TrimEnd('\')
    $rootWithSlash = $rootFull + '\'
    @(Get-CimInstance Win32_Process | Where-Object {
        if ([string]::IsNullOrWhiteSpace($_.ExecutablePath)) {
            $false
        } else {
            try {
                $exeFull = [System.IO.Path]::GetFullPath($_.ExecutablePath)
                $exeFull.Equals($rootFull, [System.StringComparison]::OrdinalIgnoreCase) -or
                    $exeFull.StartsWith($rootWithSlash, [System.StringComparison]::OrdinalIgnoreCase)
            } catch {
                $false
            }
        }
    })
}

function Stop-PortableOutputProcesses {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PortableRoot
    )

    $processes = @(Get-PortableOutputProcesses -PortableRoot $PortableRoot)
    if ($processes.Count -eq 0) {
        return
    }

    $processLabels = $processes | ForEach-Object { "$($_.Name):$($_.ProcessId)" }
    Write-Warning "Stopping running portable process(es) before rebuilding ${PortableRoot}: $($processLabels -join ', ')"
    foreach ($process in $processes) {
        Stop-Process -Id $process.ProcessId -Force -ErrorAction Stop
    }

    $deadline = [DateTimeOffset]::UtcNow.AddSeconds(20)
    do {
        Start-Sleep -Milliseconds 250
        $remaining = @(Get-PortableOutputProcesses -PortableRoot $PortableRoot)
        if ($remaining.Count -eq 0) {
            return
        }
    } while ([DateTimeOffset]::UtcNow -lt $deadline)

    $remainingLabels = $remaining | ForEach-Object { "$($_.Name):$($_.ProcessId)" }
    throw "Portable output is still locked by running process(es): $($remainingLabels -join ', ')"
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [string]$WorkingDirectory = $RepoRoot
    )

    Push-Location $WorkingDirectory
    $oldErrorActionPreference = $ErrorActionPreference
    $oldNativeCommandUseErrorActionPreference = $null
    $hasNativeCommandUseErrorActionPreference = Test-Path Variable:\PSNativeCommandUseErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        if ($hasNativeCommandUseErrorActionPreference) {
            $oldNativeCommandUseErrorActionPreference = $PSNativeCommandUseErrorActionPreference
            $PSNativeCommandUseErrorActionPreference = $false
        }
        & $FilePath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "$FilePath failed with exit code $LASTEXITCODE"
        }
    } finally {
        if ($hasNativeCommandUseErrorActionPreference) {
            $PSNativeCommandUseErrorActionPreference = $oldNativeCommandUseErrorActionPreference
        }
        $ErrorActionPreference = $oldErrorActionPreference
        Pop-Location
    }
}

function Get-CommandText {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [string]$WorkingDirectory = $RepoRoot
    )

    Push-Location $WorkingDirectory
    try {
        $output = & $FilePath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "$FilePath failed with exit code $LASTEXITCODE"
        }
        ($output | Out-String).Trim()
    } finally {
        Pop-Location
    }
}

function Write-Utf8Json {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [object]$Value
    )

    $json = $Value | ConvertTo-Json -Depth 20
    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($Path, $json + "`n", $utf8NoBom)
}

function Get-CaiPackageVersion {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Python
    )

    try {
        return Get-CommandText $Python @("-c", "from cai_compute_chain import __version__; print(__version__)") -WorkingDirectory $RepoRoot
    } catch {
        return "0.0.0"
    }
}

function Get-NextPortableBuildNumber {
    $envBuildNumber = [string]$env:CAI_PORTABLE_BUILD_NUMBER
    if (-not [string]::IsNullOrWhiteSpace($envBuildNumber)) {
        return [int]$envBuildNumber
    }

    $counterPath = Join-Path $DistRootFull "portable-build-counter.json"
    if (-not (Test-Path -LiteralPath $counterPath)) {
        return 1
    }
    try {
        $counter = Get-Content -LiteralPath $counterPath -Raw | ConvertFrom-Json
        $lastBuildNumber = 0
        if ($null -ne $counter.lastBuildNumber) {
            $lastBuildNumber = [int]$counter.lastBuildNumber
        }
        return $lastBuildNumber + 1
    } catch {
        return 1
    }
}

function Write-PortableReleaseMetadata {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PortableRoot,
        [Parameter(Mandatory = $true)]
        [string]$Python
    )

    $packageVersion = Get-CaiPackageVersion -Python $Python
    $timestamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")
    $buildNumber = Get-NextPortableBuildNumber
    $buildNumberLabel = "{0:D4}" -f $buildNumber
    $gitCommit = $null
    $gitBranch = $null
    $gitDirty = $false
    try {
        $gitCommit = Get-CommandText "git" @("rev-parse", "HEAD")
        $gitBranch = Get-CommandText "git" @("rev-parse", "--abbrev-ref", "HEAD")
        $gitDirty = -not [string]::IsNullOrWhiteSpace((Get-CommandText "git" @("status", "--short")))
    } catch {
        $gitCommit = $null
        $gitBranch = $null
        $gitDirty = $false
    }
    $shortCommit = if ([string]::IsNullOrWhiteSpace($gitCommit)) { "nogit" } else { $gitCommit.Substring(0, [Math]::Min(12, $gitCommit.Length)) }
    $buildId = "$packageVersion-$buildNumberLabel-g$shortCommit-$timestamp"
    $metadata = [ordered]@{
        schemaVersion = 1
        kind = "cai-release-artifact-metadata"
        version = $packageVersion
        versionLabel = "$packageVersion $buildNumberLabel"
        gitCommit = $gitCommit
        gitBranch = $gitBranch
        gitDirty = $gitDirty
        buildId = $buildId
        buildNumber = $buildNumber
        buildNumberLabel = $buildNumberLabel
        generatedAt = [DateTimeOffset]::UtcNow.ToString("o")
        artifacts = @()
    }

    Write-Utf8Json -Path (Join-Path $PortableRoot "release-metadata.json") -Value $metadata
    Write-Utf8Json -Path (Join-Path $DistRootFull "release-metadata.json") -Value $metadata
    Write-Utf8Json -Path (Join-Path $DistRootFull "portable-build-counter.json") -Value ([ordered]@{
        lastBuildNumber = $buildNumber
        updatedAt = [DateTimeOffset]::UtcNow.ToString("o")
    })
    Write-Host "Portable build number: $buildNumberLabel"
}

New-Item -ItemType Directory -Force -Path $DistRootFull | Out-Null

if (Test-Path $OutputDir) {
    $isInsideDist = $OutputDir.StartsWith($DistRootFull, [System.StringComparison]::OrdinalIgnoreCase)
    if (-not $isInsideDist) {
        throw "Refusing to clean output outside .dist: $OutputDir"
    }
    if (-not $NoStopRunningOutputProcesses) {
        Stop-PortableOutputProcesses -PortableRoot $OutputDir
    }
    $existingDataDir = Join-Path $OutputDir "data"
    if ($PreserveData -and (Test-Path $existingDataDir)) {
        if (Test-Path $PreservedDataBackup) {
            Remove-PortableDirectory -Path $PreservedDataBackup -Description "preserved data backup"
        }
        Move-Item -LiteralPath $existingDataDir -Destination $PreservedDataBackup
    }
    Remove-PortableDirectory -Path $OutputDir -Description "portable output"
}
if (Test-Path $PortableStageDir) {
    if (-not $NoStopRunningOutputProcesses) {
        Stop-PortableOutputProcesses -PortableRoot $PortableStageDir
    }
    Remove-PortableDirectory -Path $PortableStageDir -Description "portable stage"
}

if (-not $SkipDashboardBuild) {
    $DashboardDir = Join-Path $RepoRoot "cai\dashboard"
    Invoke-Checked "npm" @("run", "build") -WorkingDirectory $DashboardDir
}

if (-not (Test-Path $PythonExe)) {
    throw "Cannot find CAI runtime Windows Python environment: $PythonExe. Run tools\\install.ps1 first."
}

Invoke-Checked $PythonExe @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked $PythonExe @("-m", "pip", "install", "-e", (Join-Path $RepoRoot "cai\rust\cai_pyo3_bindings"))
Invoke-Checked $PythonExe @("-c", "import cai_pyo3_bindings")
Invoke-Checked $PythonExe @("-m", "pip", "install", "-e", $RepoRoot, "pyinstaller", "pystray", "pillow")

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
Invoke-Checked $PythonExe @(
    "-m",
    "cai_compute_chain.cai_desktop_app",
    "--repo-root",
    $RepoRoot,
    "--write-icon",
    $IconPath,
    "--no-start",
    "--no-tray",
    "--no-browser"
)

$oldCaiRepoRoot = $env:CAI_REPO_ROOT
$oldCaiRuntimeRoot = $env:CAI_RUNTIME_ROOT
$oldCaiIconPath = $env:CAI_ICON_PATH
$oldCaiPortableDistName = $env:CAI_PORTABLE_DIST_NAME
try {
    $env:CAI_REPO_ROOT = $RepoRoot
    $env:CAI_RUNTIME_ROOT = Join-Path $RepoRoot "cai"
    $env:CAI_ICON_PATH = $IconPath
    $env:CAI_PORTABLE_DIST_NAME = $PortableDistName
    Invoke-Checked $PythonExe @(
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--distpath",
        $DistRootFull,
        "--workpath",
        $BuildDir,
        $SpecPath
    )
} finally {
    $env:CAI_REPO_ROOT = $oldCaiRepoRoot
    $env:CAI_RUNTIME_ROOT = $oldCaiRuntimeRoot
    $env:CAI_ICON_PATH = $oldCaiIconPath
    $env:CAI_PORTABLE_DIST_NAME = $oldCaiPortableDistName
}

if (-not (Test-Path $PortableStageDir)) {
    throw "Portable stage directory was not created: $PortableStageDir"
}
if (Test-Path $OutputDir) {
    Remove-PortableDirectory -Path $OutputDir -Description "portable output"
}
Move-Item -LiteralPath $PortableStageDir -Destination $OutputDir

try {
    $ExePath = Join-Path $OutputDir "CAI.exe"
    if (-not (Test-Path $ExePath)) {
        $ExePath = Join-Path $OutputDir "cai.exe"
    }
    if (-not (Test-Path $ExePath)) {
        throw "Portable CAI.exe was not created at $ExePath"
    }
    $ResolvedExePath = (Resolve-Path -LiteralPath $ExePath).Path

    $forbiddenExecutables = Get-ChildItem -LiteralPath $OutputDir -Recurse -File -Filter *.exe |
        Where-Object {
            $_.FullName -ine $ResolvedExePath -and $_.Name -match '^cai(\.|$)'
        }
    if ($forbiddenExecutables.Count -gt 0) {
        throw "Portable output contains forbidden legacy runtime executables: $($forbiddenExecutables.FullName -join ', ')"
    }

    $SmokeDoctor = Start-Process `
        -FilePath $ExePath `
        -ArgumentList @("--repo-root", $OutputDir, "--doctor") `
        -Wait `
        -PassThru `
        -WorkingDirectory $OutputDir `
        -NoNewWindow
    if ($SmokeDoctor.ExitCode -ne 0) {
        throw "Portable cai.exe --doctor failed with exit code $($SmokeDoctor.ExitCode)"
    }

    $NativeBindingsDir = Join-Path $OutputDir "_internal\cai_pyo3_bindings"
    if (-not (Test-Path $NativeBindingsDir)) {
        throw "Portable output is missing cai_pyo3_bindings. Rebuild the native bindings before publishing updates."
    }

    Write-PortableReleaseMetadata -PortableRoot $OutputDir -Python $PythonExe
    Remove-PortableRuntimeState -PortableRoot $OutputDir

    if ($Zip) {
        $ZipPath = "$OutputDir.zip"
        if (Test-Path $ZipPath) {
            Remove-Item -LiteralPath $ZipPath -Force
        }
        Compress-Archive -Path (Join-Path $OutputDir '*') -DestinationPath $ZipPath -Force
        Write-Host "Portable zip: $ZipPath"
    }

    $exeHash = (Get-FileHash $ExePath -Algorithm SHA256).Hash
    Write-Host "Portable package: $ExePath"
    Write-Host "SHA256: $exeHash"
} finally {
    $portableDataDir = Join-Path $OutputDir "data"
    if ($PreserveData -and (Test-Path $PreservedDataBackup)) {
        if (Test-Path $portableDataDir) {
            Remove-Item -LiteralPath $portableDataDir -Recurse -Force -ErrorAction SilentlyContinue
        }
        Move-Item -LiteralPath $PreservedDataBackup -Destination $portableDataDir
    } else {
        Remove-PortableRuntimeState -PortableRoot $OutputDir
        if (Test-Path $PreservedDataBackup) {
            Remove-PortableDirectory -Path $PreservedDataBackup -Description "preserved data backup"
        }
    }
}


