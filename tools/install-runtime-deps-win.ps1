# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
param(
    [string]$RepoRoot = "",
    [string]$BundleRoot = "",

    [ValidateSet("12.4", "13.1")]
    [string]$CudaVariant = "12.4",

    [switch]$Install,
    [switch]$OpenDriverPage,
    [switch]$ForceLlamaCpp,
    [switch]$Pause
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "== $Message ==" -ForegroundColor Cyan
}

function Write-Ok {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-WarnLine {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Fail {
    param([string]$Message)
    Write-Host "[FAIL] $Message" -ForegroundColor Red
}

function Normalize-RootArgument {
    param([string]$Value)

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $null
    }

    $candidate = $Value.Trim().Trim('"').Trim("'")
    $candidate = $candidate -replace [string][char]0, ""

    # If argument parsing injected trailing switches into RepoRoot, keep only path-like prefix.
    if ($candidate -match "^(?<path>[A-Za-z]:\\.+?)\s+-[A-Za-z]") {
        $candidate = $Matches["path"]
    }

    return $candidate
}

function Resolve-CaiRoot {
    $normalized = Normalize-RootArgument -Value $RepoRoot
    if ($null -ne $normalized) {
        try {
            return [System.IO.Path]::GetFullPath($normalized)
        } catch {
            try {
                return (Resolve-Path -LiteralPath $normalized -ErrorAction Stop).Path
            } catch {
                throw "Unable to resolve RepoRoot path '$RepoRoot'. Parsed value: '$normalized'."
            }
        }
    }

    $scriptDir = Split-Path -Parent $PSCommandPath
    if ((Split-Path -Leaf $scriptDir) -ieq "tools") {
        return [System.IO.Path]::GetFullPath((Join-Path $scriptDir ".."))
    }
    return [System.IO.Path]::GetFullPath($scriptDir)
}

function Test-IsPortableRoot {
    param([string]$Root)
    return (Test-Path (Join-Path $Root "runtime\cai")) -or (Test-Path (Join-Path $Root "cai.exe"))
}

function Find-Executable {
    param([string[]]$Candidates)
    foreach ($candidate in $Candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }
    return $null
}

function Test-VcRuntime {
    $system32 = Join-Path $env:WINDIR "System32"
    $required = @("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll")
    $missing = @()
    foreach ($name in $required) {
        if (-not (Test-Path (Join-Path $system32 $name))) {
            $missing += $name
        }
    }
    return $missing
}

function Install-VcRuntime {
    $downloadDir = Join-Path $env:TEMP "cai-runtime-deps"
    $installer = Join-Path $downloadDir "vc_redist.x64.exe"
    New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null

    Write-Host "Downloading Microsoft Visual C++ Redistributable..."
    Invoke-WebRequest `
        -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" `
        -OutFile $installer

    Write-Host "Installing Microsoft Visual C++ Redistributable..."
    $process = Start-Process `
        -FilePath $installer `
        -ArgumentList @("/install", "/quiet", "/norestart") `
        -Wait `
        -PassThru

    if ($process.ExitCode -in @(0, 3010)) {
        if ($process.ExitCode -eq 3010) {
            Write-WarnLine "VC++ runtime was installed, but Windows requests a reboot."
        } else {
            Write-Ok "VC++ runtime installed."
        }
        return
    }

    throw "VC++ runtime installer failed with exit code $($process.ExitCode). Try running this script as Administrator."
}

function Get-LlamaCppPaths {
    param([string]$Root)

    $portable = Test-IsPortableRoot -Root $Root
    if ($portable) {
        $bundledRuntimeDir = $null
        if (-not [string]::IsNullOrWhiteSpace($BundleRoot)) {
            $candidate = Join-Path $BundleRoot "llama.cpp"
            if (Test-Path (Join-Path $candidate "llama-server.exe")) {
                $bundledRuntimeDir = $candidate
            }
        }

        $runtimeDir = if ($bundledRuntimeDir) {
            $bundledRuntimeDir
        } else {
            Join-Path $Root "data\runtime\llama.cpp"
        }
        return [PSCustomObject]@{
            RuntimeDir = $runtimeDir
            Server = Join-Path $runtimeDir "llama-server.exe"
            Rpc = Join-Path $runtimeDir "rpc-server.exe"
            Portable = $true
        }
    }

    $buildDir = Join-Path $Root "cai\.runtime\llama.cpp\windows\build"
    $server = Join-Path $buildDir "llama-server.exe"
    $rpc = Join-Path $buildDir "rpc-server.exe"
    if (-not (Test-Path $server)) {
        $server = Join-Path $buildDir "bin\llama-server.exe"
    }
    if (-not (Test-Path $rpc)) {
        $rpc = Join-Path $buildDir "bin\rpc-server.exe"
    }

    return [PSCustomObject]@{
        RuntimeDir = $buildDir
        Server = $server
        Rpc = $rpc
        Portable = $false
    }
}

function Get-LlamaCppRelease {
    return Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
}

function Find-ReleaseAsset {
    param(
        [object]$Release,
        [string]$Pattern
    )

    $asset = $Release.assets | Where-Object { $_.name -like $Pattern } | Select-Object -First 1
    if (-not $asset) {
        throw "Unable to find llama.cpp release asset matching '$Pattern' in '$($Release.tag_name)'."
    }
    return $asset
}

function Save-ReleaseAsset {
    param(
        [object]$Asset,
        [string]$Path,
        [switch]$ForceDownload
    )

    if ((Test-Path $Path) -and -not $ForceDownload) {
        Write-Host "Reusing $Path"
        return
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    Write-Host "Downloading $($Asset.name)..."
    Invoke-WebRequest -Uri $Asset.browser_download_url -OutFile $Path
}

function Install-LlamaCppRuntime {
    param(
        [string]$Root,
        [string]$Variant
    )

    $paths = Get-LlamaCppPaths -Root $Root
    if (-not $paths.Portable) {
        $sourceInstaller = Join-Path $Root "tools\install-llama-cpp-win.ps1"
        if (-not (Test-Path $sourceInstaller)) {
            throw "Cannot find source installer: $sourceInstaller"
        }
        & powershell -NoProfile -ExecutionPolicy Bypass -File $sourceInstaller -CudaVariant $Variant -Force:$ForceLlamaCpp
        if ($LASTEXITCODE -ne 0) {
            throw "tools\install-llama-cpp-win.ps1 failed with exit code $LASTEXITCODE"
        }
        return
    }

    $targetDir = $paths.RuntimeDir
    $downloadRoot = Join-Path $Root "data\downloads\llama.cpp"
    $release = Get-LlamaCppRelease
    $binaryAsset = Find-ReleaseAsset -Release $release -Pattern "llama-*-bin-win-cuda-$Variant-x64.zip"
    $dllAsset = Find-ReleaseAsset -Release $release -Pattern "cudart-llama-bin-win-cuda-$Variant-x64.zip"
    $releaseDownloadRoot = Join-Path $downloadRoot $release.tag_name
    $binaryZip = Join-Path $releaseDownloadRoot $binaryAsset.name
    $dllZip = Join-Path $releaseDownloadRoot $dllAsset.name
    $extractRoot = Join-Path $releaseDownloadRoot "extract"

    Save-ReleaseAsset -Asset $binaryAsset -Path $binaryZip -ForceDownload:$ForceLlamaCpp
    Save-ReleaseAsset -Asset $dllAsset -Path $dllZip -ForceDownload:$ForceLlamaCpp

    if (Test-Path $extractRoot) {
        Remove-Item -LiteralPath $extractRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $extractRoot | Out-Null
    Expand-Archive -Path $binaryZip -DestinationPath $extractRoot -Force
    Expand-Archive -Path $dllZip -DestinationPath $extractRoot -Force
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

    $required = @("llama-server.exe", "rpc-server.exe")
    $files = Get-ChildItem -Path $extractRoot -Recurse -File | Where-Object {
        $_.Extension -ieq ".dll" -or $_.Name -in $required
    }
    foreach ($file in $files) {
        Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $targetDir $file.Name) -Force
    }

    foreach ($name in $required) {
        if (-not (Test-Path (Join-Path $targetDir $name))) {
            throw "Downloaded llama.cpp runtime is missing $name"
        }
    }

    $manifest = [ordered]@{
        installed_at = (Get-Date).ToUniversalTime().ToString("o")
        tag = $release.tag_name
        cuda_variant = $Variant
        binary_asset = $binaryAsset.name
        dll_asset = $dllAsset.name
    }
    $manifest | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $targetDir "installed-release.json") -Encoding UTF8
}

function Get-NvidiaGpuNames {
    try {
        $controllers = Get-CimInstance Win32_VideoController -ErrorAction Stop
        return @($controllers | Where-Object { $_.Name -match "NVIDIA" } | ForEach-Object { $_.Name })
    } catch {
        return @()
    }
}

function Test-NvidiaDriver {
    $nvidiaSmi = Find-Executable @(
        "nvidia-smi.exe",
        (Join-Path $env:WINDIR "System32\nvidia-smi.exe"),
        "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    )

    if (-not $nvidiaSmi) {
        return [PSCustomObject]@{
            Found = $false
            Path = $null
            Query = $null
        }
    }

    $query = $null
    try {
        $query = & $nvidiaSmi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null
    } catch {
        $query = $null
    }

    return [PSCustomObject]@{
        Found = $true
        Path = $nvidiaSmi
        Query = $query
    }
}

$root = Resolve-CaiRoot
$rootIsPortable = Test-IsPortableRoot -Root $root
$issues = 0

try {
    Write-Host "CAI runtime doctor" -ForegroundColor Cyan
    Write-Host "Root: $root"
    Write-Host "Mode: $(if ($rootIsPortable) { 'portable' } else { 'source tree' })"
    Write-Host "Install missing: $Install"

    Write-Step "Core Windows runtime"
    $missingVc = Test-VcRuntime
    if ($missingVc.Count -eq 0) {
        Write-Ok "Microsoft VC++ runtime DLLs are present."
    } else {
        $issues += 1
        Write-WarnLine "Missing VC++ runtime DLLs: $($missingVc -join ', ')"
        if ($Install) {
            Install-VcRuntime
            $missingVcAfter = Test-VcRuntime
            if ($missingVcAfter.Count -eq 0) {
                Write-Ok "VC++ runtime is now available."
            } else {
                Write-WarnLine "VC++ runtime still appears incomplete: $($missingVcAfter -join ', ')"
            }
        } else {
            Write-Host "Run with -Install to download and install Microsoft VC++ Redistributable."
        }
    }

    Write-Step "llama.cpp runtime"
    $llamaPaths = Get-LlamaCppPaths -Root $root
    $missingLlama = @()
    if (-not (Test-Path $llamaPaths.Server)) {
        $missingLlama += "llama-server.exe"
    }
    if (-not (Test-Path $llamaPaths.Rpc)) {
        $missingLlama += "rpc-server.exe"
    }

    if ($missingLlama.Count -eq 0) {
        Write-Ok "llama.cpp runtime is present: $($llamaPaths.RuntimeDir)"
    } else {
        $issues += 1
        Write-WarnLine "Missing llama.cpp files: $($missingLlama -join ', ')"
        if ($Install) {
            Install-LlamaCppRuntime -Root $root -Variant $CudaVariant
            Write-Ok "llama.cpp runtime installed."
        } else {
            Write-Host "Run with -Install to download llama.cpp CUDA runtime."
        }
    }

    Write-Step "GPU driver"
    $gpuNames = Get-NvidiaGpuNames
    if ($gpuNames.Count -eq 0) {
        Write-Ok "No NVIDIA GPU detected. CAI can run in CPU mode."
    } else {
        Write-Host "NVIDIA GPU detected:"
        foreach ($name in $gpuNames) {
            Write-Host "  - $name"
        }

        $driver = Test-NvidiaDriver
        if ($driver.Found) {
            Write-Ok "nvidia-smi found: $($driver.Path)"
            if ($driver.Query) {
                foreach ($line in $driver.Query) {
                    Write-Host "  $line"
                }
            }
        } else {
            $issues += 1
            Write-WarnLine "NVIDIA GPU was detected, but nvidia-smi/driver runtime was not found."
            Write-Host "Install or update the NVIDIA display driver, then reboot if the installer asks."
            if ($OpenDriverPage -or $Install) {
                Start-Process "https://www.nvidia.com/Download/index.aspx"
                Write-Host "Opened NVIDIA driver download page."
            } else {
                Write-Host "Run with -OpenDriverPage to open the official NVIDIA driver page."
            }
            Write-Host "CAI will still be able to run in CPU mode until the driver is installed."
        }
    }

    Write-Step "Summary"
    if ($issues -eq 0) {
        Write-Ok "No missing runtime dependencies were detected."
        exit 0
    }

    if ($Install) {
        Write-WarnLine "Some items may still require admin rights, manual driver installation, or reboot."
        exit 0
    }

    Write-WarnLine "$issues issue(s) detected. Re-run with -Install to fix supported items."
    exit 2
} finally {
    if ($Pause) {
        Write-Host ""
        Read-Host "Press Enter to close"
    }
}

