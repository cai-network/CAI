# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
param(
    [string]$OutputDir = "",
    [switch]$Console
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $RepoRoot ".dist\cai-desktop"
} elseif (-not [System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $RepoRoot $OutputDir
}

$VenvDir = Join-Path $RepoRoot ".venv-desktop"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$PythonVersion = if ([string]::IsNullOrWhiteSpace($env:CAI_DESKTOP_PYTHON_VERSION)) {
    "3.14"
} else {
    $env:CAI_DESKTOP_PYTHON_VERSION
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE"
    }
}

if (Test-Path $PythonExe) {
    $ExistingVersion = (& $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
    if ($ExistingVersion -ne $PythonVersion) {
        $ResolvedVenv = [System.IO.Path]::GetFullPath($VenvDir)
        $ResolvedRepo = [System.IO.Path]::GetFullPath($RepoRoot)
        if (-not $ResolvedVenv.StartsWith($ResolvedRepo, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to recreate desktop venv outside repository: $ResolvedVenv"
        }
        Remove-Item -LiteralPath $ResolvedVenv -Recurse -Force
    }
}

if (-not (Test-Path $PythonExe)) {
    py -$PythonVersion -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        throw "py -$PythonVersion -m venv failed with exit code $LASTEXITCODE"
    }
}

Invoke-Checked $PythonExe @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked $PythonExe @("-m", "pip", "install", "-e", (Join-Path $RepoRoot "cai\rust\cai_pyo3_bindings"))
Invoke-Checked $PythonExe @("-c", "import cai_pyo3_bindings")
Invoke-Checked $PythonExe @("-m", "pip", "install", "-e", $RepoRoot, "pyinstaller", "pystray", "pillow")

$BuildDir = Join-Path $RepoRoot "build\cai-desktop"
$SpecDir = Join-Path $BuildDir "spec"
$EntryPoint = Join-Path $RepoRoot "src\cai_compute_chain\cai_desktop_app.py"
$SrcPath = Join-Path $RepoRoot "src"
$IconPath = Join-Path $BuildDir "cai.ico"
$ModeArg = "--windowed"
if ($Console) {
    $ModeArg = "--console"
}

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

$pyinstallerArgs = @(
    "--noconfirm",
    "--clean",
    "--onefile",
    $ModeArg,
    "--name",
    "CAI",
    "--icon",
    $IconPath,
    "--add-data",
    "$IconPath;assets",
    "--distpath",
    $OutputDir,
    "--workpath",
    $BuildDir,
    "--specpath",
    $SpecDir,
    "--paths",
    $SrcPath,
    "--collect-submodules",
    "pystray",
    "--hidden-import",
    "PIL.Image",
    "--hidden-import",
    "PIL.ImageDraw",
    $EntryPoint
)

$BuildStartedAt = Get-Date
Invoke-Checked $PythonExe (@("-m", "PyInstaller") + $pyinstallerArgs)

$ExePath = Join-Path $OutputDir "cai.exe"
if (-not (Test-Path $ExePath)) {
    throw "cai.exe was not created at $ExePath"
}

$ExeItem = Get-Item $ExePath
if ($ExeItem.LastWriteTime -lt $BuildStartedAt.AddSeconds(-1)) {
    throw "cai.exe exists at $ExePath, but it was not rebuilt. Close running cai.exe processes and try again."
}

Write-Host "Built $ExePath"


