# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
param(
    [string]$Distro = "Ubuntu-24.04"
)

$ErrorActionPreference = "Stop"

function Convert-WindowsPathToWsl {
    param([string]$WindowsPath)

    $normalizedPath = $WindowsPath -replace "\\", "/"
    if ($normalizedPath -match "^([A-Za-z]):/(.+)$") {
        return "/mnt/$($matches[1].ToLower())/$($matches[2])"
    }

    throw "Could not convert Windows path to WSL path: $WindowsPath"
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
$CAIRepoRoot = Join-Path $repoRoot "cai"
if (-not (Test-Path $CAIRepoRoot)) {
    throw "Expected CAI runtime directory inside monorepo: $CAIRepoRoot"
}

$CAIRepoRootLinux = Convert-WindowsPathToWsl -WindowsPath $CAIRepoRoot
$bashScript = @'
set -e
cd '__cai_REPO__'
PY='.venv-wsl/bin/python'
if [ ! -x "$PY" ]; then
  echo 'Missing .venv-wsl. Build the WSL environment first.'
  exit 1
fi
UV_BIN=/root/.local/bin/uv
if [ ! -x "$UV_BIN" ]; then
  UV_BIN=$(command -v uv 2>/dev/null || true)
fi
if [ -z "$UV_BIN" ]; then
  echo 'uv is not installed in WSL.'
  exit 1
fi
"$UV_BIN" pip uninstall --python "$PY" -y mlx-cpu >/dev/null 2>&1 || true
"$UV_BIN" pip install --python "$PY" --reinstall 'mlx[cuda13]==0.30.6'
SITE_PACKAGES=$("$PY" -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="$SITE_PACKAGES/mlx/lib:$SITE_PACKAGES/nvidia/cu13/lib:$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cusparselt/lib:$SITE_PACKAGES/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
"$PY" - <<'PY'
import importlib.metadata as md
import mlx.core as mx

names = sorted(
    dist.metadata["Name"]
    for dist in md.distributions()
    if dist.metadata["Name"].lower().startswith("mlx")
)
print("packages=" + ", ".join(names))
print("default_device=" + str(mx.default_device()))
PY
'@
$bashScript = $bashScript.Replace('__cai_REPO__', $CAIRepoRootLinux)
$bashScript = $bashScript.Replace("`r", "")

$tempScriptWin = Join-Path $repoRoot "tmp\\enable-wsl-gpu.sh"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $tempScriptWin) | Out-Null
[System.IO.File]::WriteAllText($tempScriptWin, $bashScript.Replace("`r`n", "`n"))
$tempScriptLinux = Convert-WindowsPathToWsl -WindowsPath $tempScriptWin
wsl -d $Distro -u root -- bash $tempScriptLinux
