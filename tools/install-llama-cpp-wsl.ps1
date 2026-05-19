# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
param(
    [string]$Distro = "Ubuntu-24.04",
    [string]$Ref = "master",
    [string]$PatchDir = "",
    [string]$SourceArchive = "",
    [string]$BuildTarget = "",
    [string]$WslRuntimeRoot = "",
    [switch]$SkipSystemPackages,
    [switch]$SkipCaiPatches,
    [switch]$RequireCaiPatches,
    [switch]$CpuOnly,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

function Convert-WindowsPathToWsl {
    param([string]$WindowsPath)

    $normalizedPath = $WindowsPath -replace "\\", "/"
    if ($normalizedPath -match "^([A-Za-z]):/(.+)$") {
        return "/mnt/$($matches[1].ToLower())/$($matches[2])"
    }

    throw "Unable to convert Windows path to WSL path: $WindowsPath"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$CAIRepoRoot = Join-Path $repoRoot "cai"
$CAIRepoRootLinux = Convert-WindowsPathToWsl -WindowsPath $CAIRepoRoot
$defaultPatchDir = Join-Path $repoRoot "patches\llama.cpp"
$resolvedPatchDir = if ([string]::IsNullOrWhiteSpace($PatchDir)) { $defaultPatchDir } else { $PatchDir }
$patchDirLinux = Convert-WindowsPathToWsl -WindowsPath (Resolve-Path -LiteralPath $resolvedPatchDir).Path
$sourceArchiveLinux = if ([string]::IsNullOrWhiteSpace($SourceArchive)) {
    ""
} else {
    Convert-WindowsPathToWsl -WindowsPath (Resolve-Path -LiteralPath $SourceArchive).Path
}
$runtimeRootLinux = if ([string]::IsNullOrWhiteSpace($WslRuntimeRoot)) {
    "$CAIRepoRootLinux/.runtime/llama.cpp/wsl"
} elseif ($WslRuntimeRoot.StartsWith("/")) {
    $WslRuntimeRoot.TrimEnd("/")
} else {
    Convert-WindowsPathToWsl -WindowsPath (Resolve-Path -LiteralPath $WslRuntimeRoot).Path
}
$sourceRootLinux = "$runtimeRootLinux/source"
$buildRootLinux = "$runtimeRootLinux/build"
$applyCaiPatches = if ($SkipCaiPatches) { "0" } else { "1" }
$requireCaiPatchesValue = if ($RequireCaiPatches) { "1" } else { "0" }
$cmakeAccelFlags = if ($CpuOnly) {
    "-DGGML_CUDA=OFF -DGGML_RPC=ON"
} else {
    "-DGGML_CUDA=ON -DGGML_RPC=ON"
}
$systemPackages = if ($CpuOnly) {
    "build-essential cmake ninja-build git"
} else {
    "build-essential cmake ninja-build nvidia-cuda-toolkit git"
}

$packageCmd = if ($SkipSystemPackages) {
    "echo '[llama.cpp] skipping apt install'"
} else {
    @"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y __SYSTEM_PACKAGES__
"@
}

$cleanCmd = if ($Clean) {
    "rm -rf '$buildRootLinux'"
} else {
    ":"
}

$bashTemplate = @'
set -e
__PACKAGE_CMD__
mkdir -p '__RUNTIME_ROOT__'
__CLEAN_CMD__
source_from_archive=0
archive_marker='__SOURCE_ROOT__/.cai-archive-source-ref'
source_archive='__SOURCE_ARCHIVE__'
build_target='__BUILD_TARGET__'

install_source_from_archive() {
  archive_ref="$1"
  archive_tmp="__RUNTIME_ROOT__/llama.cpp-source.tar.gz"
  extract_tmp=$(mktemp -d)

  rm -rf '__SOURCE_ROOT__'
  if [ -n "$source_archive" ]; then
    echo "[llama.cpp] using provided upstream source archive: $source_archive"
    cp "$source_archive" "$archive_tmp"
  else
    echo "[llama.cpp] downloading upstream source archive for $archive_ref"
    curl -L --retry 3 --retry-delay 5 --fail \
      -o "$archive_tmp" \
      "https://github.com/ggml-org/llama.cpp/archive/${archive_ref}.tar.gz"
  fi
  tar -xzf "$archive_tmp" -C "$extract_tmp"
  extracted_root=$(find "$extract_tmp" -mindepth 1 -maxdepth 1 -type d | head -n 1)
  if [ -z "$extracted_root" ]; then
    echo "[llama.cpp] source archive did not contain an extracted directory" >&2
    exit 1
  fi

  mkdir -p '__SOURCE_ROOT__'
  shopt -s dotglob
  cp -a "$extracted_root"/* '__SOURCE_ROOT__'/
  shopt -u dotglob
  rm -rf "$extract_tmp"
  printf '%s\n' "$archive_ref" > "$archive_marker"

  git -C '__SOURCE_ROOT__' init -q
  git -C '__SOURCE_ROOT__' config user.email "cai-build@example.invalid"
  git -C '__SOURCE_ROOT__' config user.name "CAI build"
  git -C '__SOURCE_ROOT__' add -A
  git -C '__SOURCE_ROOT__' commit -q -m "CAI upstream archive $archive_ref"
  source_from_archive=1
}

if [ -e '__SOURCE_ROOT__' ] && [ ! -d '__SOURCE_ROOT__/.git' ]; then
  echo "[llama.cpp] removing incomplete source checkout"
  rm -rf '__SOURCE_ROOT__'
fi
if [ -f "$archive_marker" ]; then
  archive_ref=$(cat "$archive_marker")
  if [ "$archive_ref" = '__REF__' ]; then
    source_from_archive=1
  else
    echo "[llama.cpp] archive source ref changed from $archive_ref to __REF__"
    rm -rf '__SOURCE_ROOT__'
    source_from_archive=0
  fi
fi
if [ -n "$source_archive" ] && [ "$source_from_archive" = "0" ]; then
  install_source_from_archive '__REF__'
fi
if [ ! -d '__SOURCE_ROOT__/.git' ]; then
  clone_attempt=1
  while [ "$clone_attempt" -le 3 ]; do
    echo "[llama.cpp] cloning upstream source, attempt $clone_attempt"
    if git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git '__SOURCE_ROOT__'; then
      break
    fi
    if [ "$clone_attempt" -eq 3 ]; then
      echo "[llama.cpp] failed to clone upstream source after $clone_attempt attempts, trying source archive"
      rm -rf '__SOURCE_ROOT__'
      break
    fi
    rm -rf '__SOURCE_ROOT__'
    clone_attempt=$((clone_attempt + 1))
    sleep 5
  done
fi
if [ ! -d '__SOURCE_ROOT__/.git' ]; then
  install_source_from_archive '__REF__'
fi
if [ "$source_from_archive" = "0" ]; then
  if ! git -C '__SOURCE_ROOT__' fetch --tags --force; then
    echo "[llama.cpp] git fetch failed, falling back to source archive"
    install_source_from_archive '__REF__'
  fi
fi
if [ "$source_from_archive" = "0" ]; then
  if ! git -C '__SOURCE_ROOT__' checkout '__REF__'; then
    if git -C '__SOURCE_ROOT__' fetch --depth 1 origin '__REF__'; then
      git -C '__SOURCE_ROOT__' checkout FETCH_HEAD
    else
      echo "[llama.cpp] ref fetch failed, falling back to source archive"
      install_source_from_archive '__REF__'
    fi
  fi
fi
git -C '__SOURCE_ROOT__' reset --hard HEAD
git -C '__SOURCE_ROOT__' clean -fd

apply_cai_patches='__APPLY_CAI_PATCHES__'
require_cai_patches='__REQUIRE_CAI_PATCHES__'
patch_dir='__PATCH_DIR__'
patch_series="$patch_dir/series"
patch_count=0
applied_json="["
json_comma=""

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

apply_patch_file() {
  patch_path="$1"
  patch_name="$2"
  if [ ! -f "$patch_path" ]; then
    echo "[llama.cpp] patch file not found: $patch_path" >&2
    exit 1
  fi
  if git -C '__SOURCE_ROOT__' apply --check "$patch_path"; then
    echo "[llama.cpp] applying CAI patch $patch_name"
    git -C '__SOURCE_ROOT__' apply "$patch_path"
    patch_state="applied"
  elif git -C '__SOURCE_ROOT__' apply --reverse --check "$patch_path"; then
    echo "[llama.cpp] CAI patch already applied $patch_name"
    patch_state="already_applied"
  else
    echo "[llama.cpp] failed to apply CAI patch $patch_name" >&2
    exit 1
  fi
  patch_count=$((patch_count + 1))
  escaped_patch_name=$(json_escape "$patch_name:$patch_state")
  applied_json="$applied_json$json_comma\"$escaped_patch_name\""
  json_comma=","
}

if [ "$apply_cai_patches" = "1" ]; then
  if [ -f "$patch_series" ]; then
    while IFS= read -r patch_name || [ -n "$patch_name" ]; do
      case "$patch_name" in
        ""|\#*) continue ;;
      esac
      apply_patch_file "$patch_dir/$patch_name" "$patch_name"
    done < "$patch_series"
  else
    for patch_path in "$patch_dir"/*.patch; do
      [ -e "$patch_path" ] || continue
      apply_patch_file "$patch_path" "$(basename "$patch_path")"
    done
  fi
fi

if [ "$require_cai_patches" = "1" ] && [ "$patch_count" -eq 0 ]; then
  echo "[llama.cpp] CAI patches are required, but no patches were applied." >&2
  exit 1
fi
applied_json="$applied_json]"

if command -v ninja >/dev/null 2>&1; then
  cmake -S '__SOURCE_ROOT__' -B '__BUILD_ROOT__' -G Ninja __CMAKE_ACCEL_FLAGS__ -DCMAKE_BUILD_TYPE=Release
else
  cmake -S '__SOURCE_ROOT__' -B '__BUILD_ROOT__' __CMAKE_ACCEL_FLAGS__ -DCMAKE_BUILD_TYPE=Release
fi
if [ -n "$build_target" ]; then
  cmake --build '__BUILD_ROOT__' --config Release --target "$build_target" -j
else
  cmake --build '__BUILD_ROOT__' --config Release -j
fi
manifest_path="__RUNTIME_ROOT__/installed-patched-source.json"
cat > "$manifest_path" <<EOF
{
  "installed_at": "$(date -Iseconds)",
  "ref": "__REF__",
  "source_root": "__SOURCE_ROOT__",
  "build_root": "__BUILD_ROOT__",
  "patch_dir": "$patch_dir",
  "patch_count": $patch_count,
  "applied_patches": $applied_json,
  "cpu_only": __CPU_ONLY_JSON__,
  "cmake_accel_flags": "__CMAKE_ACCEL_FLAGS__",
  "build_target": "__BUILD_TARGET__",
  "llama_server": "__BUILD_ROOT__/bin/llama-server",
  "rpc_server": "__BUILD_ROOT__/bin/rpc-server"
}
EOF
printf 'llama_server=%s\n' '__BUILD_ROOT__/bin/llama-server'
printf 'rpc_server=%s\n' '__BUILD_ROOT__/bin/rpc-server'
printf 'patch_count=%s\n' "$patch_count"
printf 'patched_manifest=%s\n' "$manifest_path"
'@

$bashCmd = $bashTemplate
$bashCmd = $bashCmd.Replace("__PACKAGE_CMD__", $packageCmd)
$bashCmd = $bashCmd.Replace("__SYSTEM_PACKAGES__", $systemPackages)
$bashCmd = $bashCmd.Replace("__RUNTIME_ROOT__", $runtimeRootLinux)
$bashCmd = $bashCmd.Replace("__CLEAN_CMD__", $cleanCmd)
$bashCmd = $bashCmd.Replace("__SOURCE_ROOT__", $sourceRootLinux)
$bashCmd = $bashCmd.Replace("__BUILD_ROOT__", $buildRootLinux)
$bashCmd = $bashCmd.Replace("__REF__", $Ref)
$bashCmd = $bashCmd.Replace("__PATCH_DIR__", $patchDirLinux)
$bashCmd = $bashCmd.Replace("__SOURCE_ARCHIVE__", $sourceArchiveLinux)
$bashCmd = $bashCmd.Replace("__BUILD_TARGET__", $BuildTarget)
$bashCmd = $bashCmd.Replace("__APPLY_CAI_PATCHES__", $applyCaiPatches)
$bashCmd = $bashCmd.Replace("__REQUIRE_CAI_PATCHES__", $requireCaiPatchesValue)
$bashCmd = $bashCmd.Replace("__CMAKE_ACCEL_FLAGS__", $cmakeAccelFlags)
$bashCmd = $bashCmd.Replace("__CPU_ONLY_JSON__", ($(if ($CpuOnly) { "true" } else { "false" })))

$bashCmd | wsl.exe -d $Distro -u root -- bash -s
exit $LASTEXITCODE
