# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from . import __version__
from .cai_runtime_launcher import state_url_from_multiaddr
from .model import CaiNetworkConfig
from .wallet_signing import (
    SIGNING_SCHEME_ML_DSA_65,
    address_from_public_key_b64,
    decode_bytes,
    hybrid_address_from_public_keys_b64,
    mldsa65_keypair_b64_from_seed,
    public_key_b64_from_seed,
    sign_payload_mldsa65_b64,
    sign_payload_b64,
    verify_hybrid_payload_signature,
    verify_payload_signature,
)

UPDATE_MANIFEST_VERSION = 1
UPDATE_PROTOCOL_VERSION = 1
UPDATE_MIN_COMPATIBLE_PROTOCOL_VERSION = 1
PACKAGE_METADATA_PATH = ".cai-update/package.json"
UPDATE_SERVER_ENABLED_ENV = "CAI_UPDATE_SERVER_ENABLED"
AUTO_UPDATE_ENABLED_ENV = "CAI_AUTO_UPDATE"
AUTO_UPDATE_CHECK_TIMEOUT_SECONDS_ENV = "CAI_AUTO_UPDATE_CHECK_TIMEOUT_SECONDS"
AUTO_UPDATE_IDLE_SECONDS_ENV = "CAI_AUTO_UPDATE_IDLE_SECONDS"
AUTO_UPDATE_IDLE_TIMEOUT_SECONDS_ENV = "CAI_AUTO_UPDATE_IDLE_TIMEOUT_SECONDS"
UPDATE_BASE_URL_ENV = "CAI_UPDATE_BASE_URL"
UPDATE_CHANNEL_ENV = "CAI_UPDATE_CHANNEL"
UPDATE_SOURCE_ARTIFACT_ENV = "CAI_UPDATE_SOURCE_ARTIFACT"
UPDATE_SOURCE_ROOT_ENV = "CAI_UPDATE_SOURCE_ROOT"
UPDATE_PORTABLE_ARTIFACT_ENV = "CAI_UPDATE_PORTABLE_ARTIFACT"
UPDATE_PORTABLE_MANIFEST_ENV = "CAI_UPDATE_PORTABLE_MANIFEST"
UPDATE_PORTABLE_RELEASE_METADATA_ENV = "CAI_UPDATE_PORTABLE_RELEASE_METADATA"
UPDATE_GITHUB_REPOSITORY_ENV = "CAI_UPDATE_GITHUB_REPOSITORY"
UPDATE_GITHUB_BRANCH_ENV = "CAI_UPDATE_GITHUB_BRANCH"
UPDATE_GITHUB_TOKEN_ENV = "CAI_UPDATE_GITHUB_TOKEN"
UPDATE_GITHUB_API_BASE_ENV = "CAI_UPDATE_GITHUB_API_BASE"
UPDATE_SIGNING_SEED_ENV = "CAI_UPDATE_SIGNING_SEED_B64"
UPDATE_SIGNING_PUBLIC_KEY_ENV = "CAI_UPDATE_SIGNING_PUBLIC_KEY_B64"
UPDATE_TRUSTED_PUBLIC_KEYS_ENV = "CAI_UPDATE_TRUSTED_PUBLIC_KEYS_B64"
REQUIRE_SIGNED_UPDATES_ENV = "CAI_REQUIRE_SIGNED_UPDATES"
UPDATE_STATUS_DIR = ".cai-update"
UPDATE_STATUS_PATH = ".cai-update/status.json"
UPDATE_ACTIVITY_PATH = ".cai-update/activity.json"
UPDATE_STAGE_PATH = ".cai-update/stage"
PORTABLE_UPDATE_STAGE_PATH = ".cai-update/stage-portable"
PORTABLE_UPDATE_PLAN_PATH = ".cai-update/portable-update-plan.json"
PORTABLE_UPDATE_SCRIPT_PATH = ".cai-update/apply-portable-update.ps1"
PORTABLE_UPDATE_BATCH_PATH = ".cai-update/apply-portable-update.bat"
PORTABLE_UPDATE_CANCEL_PATH = ".cai-update/cancel-portable-update.json"
PORTABLE_UPDATE_APPLY_LOG_PATH = ".cai-update/apply-portable-update.log"
UPDATE_ROLLBACK_MARKER_PATH = ".cai-update/rollback.json"
UPDATE_ROLLBACK_BACKUP_PATH = ".cai-update/rollback-backup"
GITHUB_API_BASE_URL = "https://api.github.com"
UPDATE_SIGNATURE_SCHEME_ED25519 = "cai-update-manifest-ed25519-v1"
UPDATE_SIGNATURE_SCHEME_HYBRID = "cai-update-manifest-hybrid-ed25519-ml-dsa-65-v1"
UPDATE_SIGNATURE_SCHEME = UPDATE_SIGNATURE_SCHEME_HYBRID
GENERATED_UPDATE_ROOTS: tuple[str, ...] = ("cai/dashboard/build",)
FORBIDDEN_PACKAGE_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(^|/)\.cai-api-token$", re.IGNORECASE),
    re.compile(r"(^|/)(\.cai|\.cai-local|\.cai-local-testnet)(/|$)", re.IGNORECASE),
    re.compile(r"(^|/)(\.cai-update|\.cai-update-cache)(/|$)", re.IGNORECASE),
    re.compile(r"(^|/)data(/|$)", re.IGNORECASE),
    re.compile(r"(^|/)cai_log(/|$)", re.IGNORECASE),
    re.compile(
        r"(^|/)(wallets|session|ledger|chain|node-config|settlements|"
        r"worker-payouts|job-intents|execution-receipts)\.json$",
        re.IGNORECASE,
    ),
    re.compile(r"(^|/)journal\.jsonl$", re.IGNORECASE),
)
_TRUTHY = {"1", "true", "yes", "on"}
_FALSEY = {"0", "false", "no", "off"}
_PORTABLE_UPDATE_CANCELABLE_STATUSES = {
    "checking",
    "waiting_for_idle",
    "downloading",
    "staging",
    "restart_pending",
}
_DOWNLOAD_PROGRESS_START = 20
_DOWNLOAD_PROGRESS_END = 65
_RESTART_PENDING_PROGRESS = 68
_DOWNLOAD_CHUNK_SIZE_BYTES = 1024 * 1024
_DOWNLOAD_MAX_ATTEMPTS = 4
_DOWNLOAD_RANGE_CHUNK_SIZE_BYTES = 16 * 1024 * 1024
_DOWNLOAD_RANGE_CHUNK_MAX_ATTEMPTS = 8
_DEFAULT_AUTO_UPDATE_CHECK_TIMEOUT_SECONDS = 60
_DEFAULT_AUTO_UPDATE_IDLE_SECONDS = 45
_DEFAULT_AUTO_UPDATE_IDLE_TIMEOUT_SECONDS = 30 * 60
_STALE_ACTIVE_REQUEST_SECONDS = 2 * 60 * 60
_GITHUB_REPOSITORY_RE = re.compile(
    r"^(?:https?://github\.com/|git@github\.com:|ssh://git@github\.com/)([^/]+)/([^/]+?)(?:\.git)?/?$",
    re.IGNORECASE,
)


class UpdateError(RuntimeError):
    """Raised when a CAI update cannot be prepared or applied safely."""


class UpdateCancelled(UpdateError):
    """Raised when a CAI update was cancelled by local user request."""


@dataclass(frozen=True)
class LocalUpdateState:
    repo_root: Path
    install_kind: str
    version: str | None
    git_commit: str | None
    git_branch: str | None
    git_dirty: bool
    tracked_files: tuple[str, ...]
    build_id: str | None = None
    build_number: int | None = None
    build_number_label: str | None = None


@dataclass(frozen=True)
class GitHubUpdateSource:
    repository: str
    branch: str
    api_base_url: str
    repo_url: str


@dataclass(frozen=True)
class ValidatorUpdateSource:
    base_url: str


def update_server_enabled() -> bool:
    return str(os.getenv(UPDATE_SERVER_ENABLED_ENV) or "").strip().lower() in _TRUTHY


def auto_update_enabled() -> bool:
    raw_value = str(os.getenv(AUTO_UPDATE_ENABLED_ENV) or "").strip().lower()
    if raw_value in _FALSEY:
        return False
    if raw_value in _TRUTHY:
        return True
    return True


def resolve_repo_root(repo_root: Path | None = None) -> Path:
    if repo_root is not None:
        return repo_root.expanduser().resolve()

    env_repo_root = (
        str(os.getenv("CAI_RUNTIME_REPO") or "")
    ).strip()
    if env_repo_root:
        return Path(env_repo_root).expanduser().resolve()

    return Path(__file__).resolve().parents[2]


def update_status_path(repo_root: Path | None = None) -> Path:
    return resolve_repo_root(repo_root) / Path(UPDATE_STATUS_PATH)


def update_activity_path(repo_root: Path | None = None) -> Path:
    return resolve_repo_root(repo_root) / Path(UPDATE_ACTIVITY_PATH)


def update_stage_dir(repo_root: Path | None = None) -> Path:
    return resolve_repo_root(repo_root) / Path(UPDATE_STAGE_PATH)


def portable_update_plan_path(repo_root: Path | None = None) -> Path:
    return resolve_repo_root(repo_root) / Path(PORTABLE_UPDATE_PLAN_PATH)


def portable_update_script_path(repo_root: Path | None = None) -> Path:
    return resolve_repo_root(repo_root) / Path(PORTABLE_UPDATE_SCRIPT_PATH)


def portable_update_batch_path(repo_root: Path | None = None) -> Path:
    return resolve_repo_root(repo_root) / Path(PORTABLE_UPDATE_BATCH_PATH)


def portable_update_cancel_path(repo_root: Path | None = None) -> Path:
    return resolve_repo_root(repo_root) / Path(PORTABLE_UPDATE_CANCEL_PATH)


def portable_update_apply_log_path(repo_root: Path | None = None) -> Path:
    return resolve_repo_root(repo_root) / Path(PORTABLE_UPDATE_APPLY_LOG_PATH)


def update_rollback_marker_path(repo_root: Path | None = None) -> Path:
    return resolve_repo_root(repo_root) / Path(UPDATE_ROLLBACK_MARKER_PATH)


def update_rollback_backup_dir(repo_root: Path | None = None) -> Path:
    return resolve_repo_root(repo_root) / Path(UPDATE_ROLLBACK_BACKUP_PATH)


def source_repo_looks_valid(repo_root: Path) -> bool:
    runtime_main = repo_root / "cai" / "src" / "cai" / "main.py"
    return (repo_root / "src" / "cai_compute_chain").is_dir() and runtime_main.is_file()


def portable_install_looks_valid(portable_root: Path) -> bool:
    root = portable_root.expanduser().resolve()
    return (
        (root / "CAI.exe").is_file()
        or (root / "cai.exe").is_file()
        or (root / "runtime" / "cai").exists()
    )


def normalize_update_install_kind(value: str | None = None) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return "source"
    if normalized in {"source", "repo", "repository"}:
        return "source"
    if normalized in {"portable", "runtime", "portable-runtime"}:
        return "portable"
    raise UpdateError(
        f"Unsupported CAI update install kind {value!r}. Expected 'source' or 'portable'."
    )


def detect_local_update_install_kind(repo_root: Path) -> str:
    root = resolve_repo_root(repo_root)
    if source_repo_looks_valid(root):
        return "source"
    if portable_install_looks_valid(root):
        return "portable"
    return "unknown"


def _load_release_metadata_file(metadata_path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError) as exc:
        raise UpdateError(f"Unable to read CAI release metadata at {metadata_path}: {exc}") from exc
    return payload if isinstance(payload, dict) else None


def _release_metadata_string(metadata: dict[str, Any] | None, key: str) -> str | None:
    if not isinstance(metadata, dict):
        return None
    value = str(metadata.get(key) or "").strip()
    return value or None


def build_runtime_version_label(
    version: str | None,
    *,
    git_commit: str | None = None,
    build_id: str | None = None,
    build_number_label: str | None = None,
) -> str:
    resolved_version = str(version or __version__).strip() or __version__
    resolved_build_number = str(build_number_label or "").strip()
    if resolved_build_number:
        return f"{resolved_version.split('+', 1)[0]} {resolved_build_number}"

    resolved_build_id = str(build_id or "").strip()
    if resolved_build_id:
        return resolved_build_id

    if "+" in resolved_version:
        return resolved_version

    resolved_commit = str(git_commit or "").strip()
    if resolved_commit:
        return f"{resolved_version}+g{resolved_commit[:12]}"
    return resolved_version


def _portable_release_metadata_candidates(
    repo_root: Path,
    *,
    portable_artifact: Path | None = None,
) -> tuple[Path, ...]:
    root = resolve_repo_root(repo_root)
    candidates: list[Path] = []
    configured = str(os.getenv(UPDATE_PORTABLE_RELEASE_METADATA_ENV) or "").strip()
    if configured:
        configured_path = Path(configured).expanduser()
        if not configured_path.is_absolute():
            configured_path = root / configured_path
        candidates.append(configured_path.resolve())
    if portable_artifact is not None:
        artifact_root = portable_artifact if portable_artifact.is_dir() else portable_artifact.parent
        candidates.extend(
            [
                artifact_root / "release-metadata.json",
                artifact_root / ".cai-update" / "release-metadata.json",
            ]
        )
    candidates.extend(
        [
            root / ".dist" / "release-metadata.json",
            root / "release-metadata.json",
        ]
    )

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(resolved)
    return tuple(unique_candidates)


def load_portable_release_metadata(
    repo_root: Path,
    *,
    portable_artifact: Path | None = None,
) -> dict[str, Any] | None:
    for candidate in _portable_release_metadata_candidates(
        repo_root,
        portable_artifact=portable_artifact,
    ):
        metadata = _load_release_metadata_file(candidate)
        if metadata is not None:
            return metadata
    return None


def _portable_update_manifest_candidates(
    repo_root: Path,
    *,
    portable_artifact: Path | None = None,
) -> tuple[Path, ...]:
    root = resolve_repo_root(repo_root)
    candidates: list[Path] = []
    configured = str(os.getenv(UPDATE_PORTABLE_MANIFEST_ENV) or "").strip()
    if configured:
        configured_path = Path(configured).expanduser()
        if not configured_path.is_absolute():
            configured_path = root / configured_path
        candidates.append(configured_path.resolve())
    if portable_artifact is not None:
        artifact_root = (
            portable_artifact if portable_artifact.is_dir() else portable_artifact.parent
        )
        candidates.extend(
            [
                artifact_root / "portable-update-manifest.json",
                artifact_root / "CAI-portable.zip.manifest.json",
                artifact_root / ".cai-update" / "portable-update-manifest.json",
            ]
        )
    candidates.extend(
        [
            root / ".dist" / "portable-update-manifest.json",
            root / ".dist" / "CAI-portable.zip.manifest.json",
        ]
    )

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(resolved)
    return tuple(unique_candidates)


def _load_json_object_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _portable_manifest_matches_artifact(
    manifest: dict[str, Any],
    archive_path: Path,
) -> bool:
    if _remote_install_kind(manifest, default="portable") != "portable":
        return False
    raw_size = manifest.get("archiveSizeBytes")
    if raw_size in (None, ""):
        return True
    try:
        expected_size = int(raw_size)
    except (TypeError, ValueError):
        return False
    return expected_size == archive_path.stat().st_size


def load_portable_update_manifest(
    repo_root: Path,
    *,
    portable_artifact: Path | None = None,
) -> dict[str, Any] | None:
    archive_path = (
        portable_artifact.expanduser().resolve()
        if portable_artifact is not None
        else None
    )
    for candidate in _portable_update_manifest_candidates(
        repo_root,
        portable_artifact=archive_path,
    ):
        manifest = _load_json_object_file(candidate)
        if manifest is None:
            continue
        try:
            validate_update_manifest(
                manifest,
                require_archive_hash=True,
                require_signature=False,
            )
        except UpdateError:
            continue
        if archive_path is not None and not _portable_manifest_matches_artifact(
            manifest,
            archive_path,
        ):
            continue
        return manifest
    return None


def _portable_update_manifest_response(
    manifest: dict[str, Any],
    *,
    base_url: str,
) -> dict[str, Any]:
    payload = dict(manifest)
    if not isinstance(payload.get("signature"), dict):
        normalized_base_url = base_url.rstrip("/")
        payload["archiveUrl"] = (
            f"{normalized_base_url}/v1/cai/update-package.zip?install_kind=portable"
        )
        payload = maybe_sign_update_manifest(payload)
    return payload


def resolve_update_base_url(
    *,
    explicit_base_url: str | None = None,
    network_config: CaiNetworkConfig | None = None,
) -> str:
    configured = (
        str(explicit_base_url or "")
        or str(os.getenv(UPDATE_BASE_URL_ENV) or "")
    ).strip()
    if configured:
        return configured.rstrip("/")

    active_network_config = network_config or CaiNetworkConfig()
    for peer in active_network_config.bootstrap_peers:
        state_url = state_url_from_multiaddr(peer, active_network_config.default_api_port)
        if state_url:
            return state_url.removesuffix("/state")

    raise UpdateError("Unable to resolve a CAI validator update base URL.")


def collect_local_update_state(repo_root: Path) -> LocalUpdateState:
    root = resolve_repo_root(repo_root)
    install_kind = detect_local_update_install_kind(root)
    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty = False
    tracked_files: tuple[str, ...] = ()
    build_id: str | None = None
    build_number: int | None = None
    build_number_label: str | None = None
    version: str | None = __version__

    if (root / ".git").exists():
        try:
            git_commit = _run_git_text(root, "rev-parse", "HEAD")
            git_branch = _run_git_text(root, "rev-parse", "--abbrev-ref", "HEAD")
            git_dirty = bool(_run_git_text(root, "status", "--short"))
            tracked_files = _run_git_tracked_files(root)
        except Exception as exc:  # noqa: BLE001
            raise UpdateError(f"Unable to inspect local git checkout at {root}: {exc}") from exc

    if install_kind == "portable":
        release_metadata = load_portable_release_metadata(root, portable_artifact=root)
        build_id = _release_metadata_string(release_metadata, "buildId")
        raw_build_number = _release_metadata_string(release_metadata, "buildNumber")
        try:
            build_number = int(raw_build_number) if raw_build_number else None
        except ValueError:
            build_number = None
        build_number_label = _release_metadata_string(
            release_metadata, "buildNumberLabel"
        )
        if build_number_label is None and build_number is not None and build_number > 0:
            build_number_label = f"{build_number:04d}"
        version = _release_metadata_string(release_metadata, "version") or version
        git_commit = _release_metadata_string(release_metadata, "gitCommit") or git_commit
        git_branch = _release_metadata_string(release_metadata, "gitBranch") or git_branch

    return LocalUpdateState(
        repo_root=root,
        install_kind=install_kind,
        version=version,
        git_commit=git_commit,
        git_branch=git_branch,
        git_dirty=git_dirty,
        tracked_files=tracked_files,
        build_id=build_id,
        build_number=build_number,
        build_number_label=build_number_label,
    )


def resolve_update_source(
    repo_root: Path,
    *,
    base_url: str | None = None,
) -> GitHubUpdateSource | ValidatorUpdateSource:
    root = resolve_repo_root(repo_root)
    local_state = collect_local_update_state(root)
    configured_channel = str(os.getenv(UPDATE_CHANNEL_ENV) or "auto").strip().lower() or "auto"

    if configured_channel not in {"auto", "github", "validator"}:
        raise UpdateError(
            f"Unsupported CAI update channel {configured_channel!r}. "
            "Expected one of: auto, github, validator."
        )

    if local_state.install_kind == "portable" and configured_channel == "github":
        raise UpdateError("Portable CAI installs can only update from a validator package.")

    if local_state.install_kind != "portable" and configured_channel in {"auto", "github"}:
        github_repository = resolve_github_update_repository(root)
        if github_repository:
            github_branch = resolve_github_update_branch(local_state)
            api_base_url = str(os.getenv(UPDATE_GITHUB_API_BASE_ENV) or GITHUB_API_BASE_URL).strip().rstrip("/")
            if not api_base_url:
                api_base_url = GITHUB_API_BASE_URL
            return GitHubUpdateSource(
                repository=github_repository,
                branch=github_branch,
                api_base_url=api_base_url,
                repo_url=f"https://github.com/{github_repository}.git",
            )
        if configured_channel == "github":
            raise UpdateError(
                "GitHub auto-update is enabled, but no repository is configured. "
                f"Set {UPDATE_GITHUB_REPOSITORY_ENV} or configure git origin."
            )

    return ValidatorUpdateSource(base_url=resolve_update_base_url(explicit_base_url=base_url))


def resolve_github_update_repository(repo_root: Path) -> str | None:
    configured_repository = str(os.getenv(UPDATE_GITHUB_REPOSITORY_ENV) or "").strip()
    if configured_repository:
        normalized = _normalize_github_repository(configured_repository)
        if normalized is None:
            raise UpdateError(
                f"Invalid GitHub repository {configured_repository!r}. "
                "Expected the form owner/repo."
            )
        return normalized

    origin_url = _git_remote_origin_url(resolve_repo_root(repo_root))
    if not origin_url:
        return None
    return _normalize_github_repository(origin_url)


def resolve_github_update_branch(local_state: LocalUpdateState) -> str:
    configured_branch = str(os.getenv(UPDATE_GITHUB_BRANCH_ENV) or "").strip()
    if configured_branch:
        return configured_branch
    if local_state.git_branch and local_state.git_branch != "HEAD":
        return local_state.git_branch
    return "main"


def _portable_archive_cache_name(portable_root: Path) -> str:
    label = re.sub(r"[^a-z0-9]+", "-", portable_root.name.lower()).strip("-") or "portable"
    return f"cai-portable-update-{label}.zip"


def build_portable_update_archive(
    repo_root: Path,
    *,
    portable_root: Path,
) -> Path:
    runtime_root = portable_root.expanduser().resolve()
    if not portable_install_looks_valid(runtime_root):
        raise UpdateError(f"CAI portable runtime not found at {runtime_root}.")

    root = resolve_repo_root(repo_root)
    cache_dir = root / ".cai-update-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / _portable_archive_cache_name(runtime_root)

    with tempfile.TemporaryDirectory(prefix="cai-portable-update-", dir=cache_dir) as temp_dir:
        temp_archive = Path(temp_dir) / archive_path.name
        with zipfile.ZipFile(temp_archive, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
            wrote_files = 0
            for source_path in sorted(runtime_root.rglob("*")):
                if not source_path.is_file():
                    continue
                relative_path = source_path.relative_to(runtime_root).as_posix()
                if _is_forbidden_package_path(relative_path):
                    continue
                _write_zip_file(bundle, source_path, relative_path)
                wrote_files += 1
            if wrote_files == 0:
                raise UpdateError(
                    f"Portable update package at {runtime_root} does not contain any safe files."
                )
        temp_archive.replace(archive_path)

    return archive_path


def resolve_portable_update_artifact(repo_root: Path) -> Path:
    root = resolve_repo_root(repo_root)
    candidates: list[Path] = []
    configured = str(os.getenv(UPDATE_PORTABLE_ARTIFACT_ENV) or "").strip()
    if configured:
        configured_path = Path(configured).expanduser()
        if not configured_path.is_absolute():
            configured_path = root / configured_path
        candidates.append(configured_path.resolve())
    candidates.extend(
        [
            root / ".dist" / "CAI-portable.zip",
            root / ".dist" / "CAI-portable",
        ]
    )

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
        if resolved.is_dir():
            return build_portable_update_archive(root, portable_root=resolved)

    raise UpdateError(
        "Portable update artifact is not configured. "
        f"Set {UPDATE_PORTABLE_ARTIFACT_ENV} or place CAI-portable(.zip) in .dist."
    )


def resolve_source_update_root(repo_root: Path) -> Path:
    root = resolve_repo_root(repo_root)
    configured = str(os.getenv(UPDATE_SOURCE_ROOT_ENV) or "").strip()
    if not configured:
        return root

    configured_path = Path(configured).expanduser()
    if not configured_path.is_absolute():
        configured_path = root / configured_path
    resolved = configured_path.resolve()
    if not source_repo_looks_valid(resolved):
        raise UpdateError(f"CAI source update repository not found at {resolved}.")
    return resolved


def resolve_source_update_artifact(repo_root: Path) -> Path:
    root = resolve_repo_root(repo_root)
    configured = str(os.getenv(UPDATE_SOURCE_ARTIFACT_ENV) or "").strip()
    if configured:
        configured_path = Path(configured).expanduser()
        if not configured_path.is_absolute():
            configured_path = root / configured_path
        resolved = configured_path.resolve()
        if not resolved.is_file():
            raise UpdateError(f"CAI source update artifact not found at {resolved}.")
        load_package_metadata_from_archive(resolved)
        return resolved

    return build_update_archive(resolve_source_update_root(root))


def build_update_package(
    repo_root: Path,
    *,
    install_kind: str | None = None,
) -> Path:
    root = resolve_repo_root(repo_root)
    normalized_install_kind = normalize_update_install_kind(install_kind)
    if normalized_install_kind == "portable":
        return resolve_portable_update_artifact(root)
    return resolve_source_update_artifact(root)


def build_portable_update_manifest(
    repo_root: Path,
    *,
    base_url: str,
) -> dict[str, Any]:
    root = resolve_repo_root(repo_root)
    archive_path = build_update_package(root, install_kind="portable")
    cached_manifest = load_portable_update_manifest(
        root,
        portable_artifact=archive_path,
    )
    if cached_manifest is not None:
        return _portable_update_manifest_response(cached_manifest, base_url=base_url)

    archive_sha256 = sha256_file(archive_path)
    local_state = collect_local_update_state(root)
    release_metadata = load_portable_release_metadata(root, portable_artifact=archive_path)
    normalized_base_url = base_url.rstrip("/")
    manifest = {
        "manifestVersion": UPDATE_MANIFEST_VERSION,
        "protocolVersion": UPDATE_PROTOCOL_VERSION,
        "minCompatibleProtocolVersion": UPDATE_MIN_COMPATIBLE_PROTOCOL_VERSION,
        "maxCompatibleProtocolVersion": UPDATE_PROTOCOL_VERSION,
        "channel": "validator",
        "version": _release_metadata_string(release_metadata, "version") or __version__,
        "generatedAt": datetime.now(tz=UTC).isoformat(),
        "gitCommit": _release_metadata_string(release_metadata, "gitCommit") or local_state.git_commit,
        "gitBranch": _release_metadata_string(release_metadata, "gitBranch") or local_state.git_branch,
        "gitDirty": False,
        "buildId": _release_metadata_string(release_metadata, "buildId"),
        "archiveUrl": f"{normalized_base_url}/v1/cai/update-package.zip?install_kind=portable",
        "archiveSha256": archive_sha256,
        "archiveSizeBytes": archive_path.stat().st_size,
        "repoKind": "portable",
        "installKind": "portable",
    }
    return maybe_sign_update_manifest(manifest)


def build_update_manifest(
    repo_root: Path,
    *,
    base_url: str,
    install_kind: str | None = None,
) -> dict[str, Any]:
    normalized_install_kind = normalize_update_install_kind(install_kind)
    if normalized_install_kind == "portable":
        return build_portable_update_manifest(repo_root, base_url=base_url)

    archive_path = build_update_package(repo_root, install_kind="source")
    package_metadata = load_package_metadata_from_archive(archive_path)
    archive_sha256 = sha256_file(archive_path)
    git_commit = str(package_metadata.get("gitCommit") or "").strip() or None
    git_branch = str(package_metadata.get("gitBranch") or "").strip() or None
    git_dirty = bool(package_metadata.get("gitDirty"))
    normalized_base_url = base_url.rstrip("/")
    manifest = {
        "manifestVersion": UPDATE_MANIFEST_VERSION,
        "protocolVersion": UPDATE_PROTOCOL_VERSION,
        "minCompatibleProtocolVersion": UPDATE_MIN_COMPATIBLE_PROTOCOL_VERSION,
        "maxCompatibleProtocolVersion": UPDATE_PROTOCOL_VERSION,
        "channel": "validator",
        "version": str(package_metadata.get("version") or "").strip() or __version__,
        "generatedAt": datetime.now(tz=UTC).isoformat(),
        "gitCommit": git_commit,
        "gitBranch": git_branch,
        "gitDirty": git_dirty,
        "archiveUrl": (
            f"{normalized_base_url}/v1/cai/update-package.zip"
            if not git_commit
            else f"{normalized_base_url}/v1/cai/update-package.zip?git_commit={git_commit}"
        ),
        "archiveSha256": archive_sha256,
        "archiveSizeBytes": archive_path.stat().st_size,
        "trackedFileCount": len(package_metadata.get("trackedFiles") or []),
        "generatedRoots": list(package_metadata.get("generatedRoots") or []),
        "repoKind": "source",
        "installKind": "source",
    }
    return maybe_sign_update_manifest(manifest)


def build_update_archive(repo_root: Path) -> Path:
    root = repo_root.expanduser().resolve()
    if not source_repo_looks_valid(root):
        raise UpdateError(f"CAI source repository not found at {root}.")

    local_state = collect_local_update_state(root)
    package_files = _collect_package_files(root, local_state.tracked_files)
    if not package_files:
        raise UpdateError("No files were selected for the CAI update package.")

    generated_roots = _generated_roots(root)
    package_metadata = {
        "packageFormatVersion": UPDATE_MANIFEST_VERSION,
        "protocolVersion": UPDATE_PROTOCOL_VERSION,
        "minCompatibleProtocolVersion": UPDATE_MIN_COMPATIBLE_PROTOCOL_VERSION,
        "maxCompatibleProtocolVersion": UPDATE_PROTOCOL_VERSION,
        "version": __version__,
        "generatedAt": datetime.now(tz=UTC).isoformat(),
        "gitCommit": local_state.git_commit,
        "gitBranch": local_state.git_branch,
        "gitDirty": local_state.git_dirty,
        "trackedFiles": list(local_state.tracked_files),
        "generatedRoots": list(generated_roots),
        "files": package_files,
    }

    cache_dir = root / ".cai-update-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    commit_label = (
        (local_state.git_commit or "snapshot")[:12]
        + ("-dirty" if local_state.git_dirty else "")
    )
    archive_path = cache_dir / f"cai-update-{commit_label}.zip"

    with tempfile.TemporaryDirectory(prefix="cai-update-", dir=cache_dir) as temp_dir:
        temp_archive = Path(temp_dir) / archive_path.name
        with zipfile.ZipFile(temp_archive, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for file_entry in package_files:
                source_path = root / Path(file_entry["path"])
                _write_zip_file(bundle, source_path, file_entry["path"])
            bundle.writestr(
                PACKAGE_METADATA_PATH,
                json.dumps(package_metadata, ensure_ascii=False, indent=2) + "\n",
            )
        temp_archive.replace(archive_path)

    return archive_path


def load_package_metadata_from_archive(archive_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(archive_path, mode="r") as bundle:
        with bundle.open(PACKAGE_METADATA_PATH, mode="r") as handle:
            payload = json.loads(handle.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise UpdateError(f"Invalid CAI update package metadata in {archive_path}.")
    return payload


def fetch_remote_update_manifest(
    *,
    base_url: str | None = None,
    install_kind: str | None = None,
    timeout_sec: int = 10,
) -> dict[str, Any]:
    manifest_url = f"{resolve_update_base_url(explicit_base_url=base_url)}/v1/cai/update-manifest"
    normalized_install_kind = normalize_update_install_kind(install_kind)
    if normalized_install_kind != "source":
        manifest_url = f"{manifest_url}?install_kind={quote(normalized_install_kind, safe='')}"
    with urlopen(Request(manifest_url), timeout=timeout_sec) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise UpdateError(f"Update manifest at {manifest_url} is not a JSON object.")
    return payload


def fetch_github_update_manifest(
    source: GitHubUpdateSource,
    *,
    timeout_sec: int = 10,
) -> dict[str, Any]:
    commit_url = (
        f"{source.api_base_url}/repos/{source.repository}/commits/"
        f"{quote(source.branch, safe='')}"
    )
    payload = _fetch_json_payload(commit_url, timeout_sec=timeout_sec)
    if not isinstance(payload, dict):
        raise UpdateError(f"GitHub commit response at {commit_url} is not a JSON object.")

    remote_commit = str(payload.get("sha") or "").strip()
    if not remote_commit:
        raise UpdateError(
            f"GitHub did not return a commit SHA for {source.repository}@{source.branch}."
        )

    commit_info = payload.get("commit")
    commit_date = None
    if isinstance(commit_info, dict):
        author_info = commit_info.get("author")
        if isinstance(author_info, dict):
            commit_date = author_info.get("date")

    return {
        "manifestVersion": UPDATE_MANIFEST_VERSION,
        "protocolVersion": UPDATE_PROTOCOL_VERSION,
        "minCompatibleProtocolVersion": UPDATE_MIN_COMPATIBLE_PROTOCOL_VERSION,
        "maxCompatibleProtocolVersion": UPDATE_PROTOCOL_VERSION,
        "channel": "github",
        "provider": "github",
        "repository": source.repository,
        "gitBranch": source.branch,
        "gitCommit": remote_commit,
        "generatedAt": commit_date,
        "repoUrl": source.repo_url.removesuffix(".git"),
        "commitUrl": payload.get("html_url"),
        "version": None,
    }


def _remote_install_kind(manifest: dict[str, Any], *, default: str = "source") -> str:
    try:
        return normalize_update_install_kind(
            str(
                manifest.get("installKind")
                or manifest.get("repoKind")
                or default
            )
        )
    except UpdateError:
        return default


def _update_available(local_state: LocalUpdateState, remote_manifest: dict[str, Any]) -> bool:
    remote_build_id = str(remote_manifest.get("buildId") or "").strip() or None
    if remote_build_id and local_state.build_id:
        return remote_build_id != local_state.build_id
    if remote_build_id and local_state.install_kind == "portable":
        return True

    remote_commit = str(remote_manifest.get("gitCommit") or "").strip() or None
    if remote_commit and local_state.git_commit:
        return remote_commit != local_state.git_commit

    remote_version = str(remote_manifest.get("version") or "").strip() or None
    local_version = str(local_state.version or "").strip() or None
    if remote_version and local_version:
        return remote_version != local_version

    if remote_commit and local_state.git_commit is None:
        return bool(remote_version and local_version and remote_version != local_version)

    return False


def check_for_updates(
    repo_root: Path,
    *,
    base_url: str | None = None,
    timeout_sec: int = 10,
) -> dict[str, Any]:
    root = resolve_repo_root(repo_root)
    local_state = collect_local_update_state(root)
    source = resolve_update_source(root, base_url=base_url)
    if isinstance(source, GitHubUpdateSource):
        remote_manifest = fetch_github_update_manifest(source, timeout_sec=timeout_sec)
        channel = "github"
        source_url = f"{source.repo_url.removesuffix('.git')}/tree/{source.branch}"
        base_url_value = None
        repository = source.repository
        target_branch = source.branch
    else:
        remote_manifest = fetch_remote_update_manifest(
            base_url=source.base_url,
            install_kind=local_state.install_kind,
            timeout_sec=timeout_sec,
        )
        validate_update_manifest(remote_manifest, require_archive_hash=True)
        channel = "validator"
        source_url = source.base_url
        base_url_value = source.base_url
        repository = None
        target_branch = remote_manifest.get("gitBranch")

    remote_commit = str(remote_manifest.get("gitCommit") or "").strip() or None
    remote_install_kind = _remote_install_kind(remote_manifest, default=local_state.install_kind)
    update_available = _update_available(local_state, remote_manifest)
    can_apply, apply_reason = _can_apply_update(
        local_state,
        target_channel=channel,
        target_branch=target_branch,
        target_install_kind=remote_install_kind,
    )
    status = "up_to_date" if not update_available else ("update_available" if can_apply else "skipped")
    message = (
        "Local CAI checkout is already up to date."
        if not update_available
        else ("CAI update is available." if can_apply else apply_reason)
    )
    result = {
        "checked": True,
        "updated": False,
        "status": status,
        "phase": status,
        "progress": 15 if update_available else 0,
        "message": message,
        "channel": channel,
        "provider": channel,
        "sourceUrl": source_url,
        "repository": repository,
        "targetBranch": target_branch,
        "baseUrl": base_url_value,
        "installKind": local_state.install_kind,
        "localVersion": local_state.version or __version__,
        "localGitCommit": local_state.git_commit,
        "localGitBranch": local_state.git_branch,
        "localGitDirty": local_state.git_dirty,
        "localBuildId": local_state.build_id,
        "remoteGitCommit": remote_commit,
        "remoteGitBranch": remote_manifest.get("gitBranch"),
        "remoteVersion": remote_manifest.get("version"),
        "remoteBuildId": remote_manifest.get("buildId"),
        "remoteInstallKind": remote_install_kind,
        "updateAvailable": update_available,
        "canApply": can_apply,
        "applyReason": apply_reason,
        "canCancel": False,
        "cancelRequested": False,
        "restartRequired": False,
        "restartScheduled": False,
        "archivePath": None,
        "portableUpdatePlanPath": None,
        "portableUpdateScriptPath": None,
        "portableUpdateCancelPath": None,
        "manifest": remote_manifest,
    }
    _write_update_status(root, result)
    return result


def apply_remote_update(
    repo_root: Path,
    *,
    base_url: str | None = None,
    timeout_sec: int = 20,
) -> dict[str, Any]:
    root = resolve_repo_root(repo_root)
    local_state = collect_local_update_state(root)
    source = resolve_update_source(root, base_url=base_url)
    target_branch = source.branch if isinstance(source, GitHubUpdateSource) else None
    target_install_kind = local_state.install_kind
    can_apply, apply_reason = _can_apply_update(
        local_state,
        target_channel="github" if isinstance(source, GitHubUpdateSource) else "validator",
        target_branch=target_branch,
        target_install_kind=target_install_kind,
    )
    if not can_apply:
        raise UpdateError(apply_reason)

    remote_manifest = (
        fetch_github_update_manifest(source, timeout_sec=timeout_sec)
        if isinstance(source, GitHubUpdateSource)
        else fetch_remote_update_manifest(
            base_url=source.base_url,
            install_kind=local_state.install_kind,
            timeout_sec=timeout_sec,
        )
    )
    if not isinstance(source, GitHubUpdateSource):
        validate_update_manifest(remote_manifest, require_archive_hash=True)
        target_install_kind = _remote_install_kind(remote_manifest, default=local_state.install_kind)
        can_apply, apply_reason = _can_apply_update(
            local_state,
            target_channel="validator",
            target_install_kind=target_install_kind,
        )
        if not can_apply:
            raise UpdateError(apply_reason)
    remote_commit = str(remote_manifest.get("gitCommit") or "").strip() or None
    if not _update_available(local_state, remote_manifest):
        result = {
            "checked": True,
            "updated": False,
            "status": "up_to_date",
            "message": "Local CAI checkout is already up to date.",
            "channel": "github" if isinstance(source, GitHubUpdateSource) else "validator",
            "provider": "github" if isinstance(source, GitHubUpdateSource) else "validator",
            "sourceUrl": (
                f"{source.repo_url.removesuffix('.git')}/tree/{source.branch}"
                if isinstance(source, GitHubUpdateSource)
                else source.base_url
            ),
            "repository": source.repository if isinstance(source, GitHubUpdateSource) else None,
            "targetBranch": source.branch if isinstance(source, GitHubUpdateSource) else remote_manifest.get("gitBranch"),
            "baseUrl": None if isinstance(source, GitHubUpdateSource) else source.base_url,
            "installKind": local_state.install_kind,
            "localVersion": local_state.version or __version__,
            "localGitCommit": local_state.git_commit,
            "localBuildId": local_state.build_id,
            "remoteGitCommit": remote_commit,
            "remoteVersion": remote_manifest.get("version"),
            "remoteBuildId": remote_manifest.get("buildId"),
            "remoteInstallKind": target_install_kind,
        }
        _write_update_status(root, result)
        return result

    if isinstance(source, GitHubUpdateSource):
        result = _apply_github_update(
            local_state,
            source=source,
            remote_manifest=remote_manifest,
        )
        _write_update_status(root, result)
        return result

    download_dir = local_state.repo_root / ".cai-update-cache"
    download_dir.mkdir(parents=True, exist_ok=True)
    archive_path = _download_update_archive(remote_manifest, download_dir, timeout_sec=timeout_sec)
    applied = (
        apply_portable_update_archive(local_state.repo_root, archive_path)
        if local_state.install_kind == "portable"
        else apply_update_archive(local_state.repo_root, archive_path)
    )
    result = {
        **applied,
        "checked": True,
        "updated": True,
        "status": "updated",
        "channel": "validator",
        "provider": "validator",
        "sourceUrl": source.base_url,
        "repository": None,
        "targetBranch": remote_manifest.get("gitBranch"),
        "baseUrl": source.base_url,
        "installKind": local_state.install_kind,
        "localVersion": local_state.version or __version__,
        "localBuildId": local_state.build_id,
        "remoteGitCommit": remote_commit,
        "remoteGitBranch": remote_manifest.get("gitBranch"),
        "remoteVersion": remote_manifest.get("version"),
        "remoteBuildId": remote_manifest.get("buildId"),
        "remoteInstallKind": target_install_kind,
        "updateAvailable": False,
        "canApply": True,
        "applyReason": "ok",
    }
    _write_update_status(root, result)
    return result


def resume_pending_portable_update_on_launch(
    repo_root: Path,
    *,
    relaunch_command: list[str] | tuple[str, ...],
    parent_pid: int | None = None,
    start_process: bool = True,
) -> dict[str, Any] | None:
    """Resume a downloaded portable update instead of downloading it again.

    Portable updates are intentionally applied by a detached helper after the
    running CAI.exe exits. If the app is launched again while a downloaded
    package is still pending, we must apply that package first rather than
    starting a fresh update check and re-downloading the same archive.
    """

    root = resolve_repo_root(repo_root)
    if detect_local_update_install_kind(root) != "portable":
        return None

    saved_status = _read_update_status(root)
    status = str(saved_status.get("status") or saved_status.get("phase") or "").strip().lower()
    plan_path = portable_update_plan_path(root)
    if not plan_path.is_file():
        return None
    if status and status not in {"restart_pending", "applying"} and not saved_status.get(
        "restartRequired"
    ):
        return None

    if _portable_update_cancel_requested(root):
        result = _portable_update_cancelled_status(
            root,
            saved_status,
            message="CAI portable update was cancelled before applying.",
        )
        _write_update_status(root, result)
        return result

    try:
        plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        result = _portable_pending_update_error(
            root,
            saved_status,
            f"CAI portable update plan could not be read: {exc}",
        )
        _write_update_status(root, result)
        return result
    if not isinstance(plan_payload, dict):
        result = _portable_pending_update_error(
            root,
            saved_status,
            "CAI portable update plan is invalid.",
        )
        _write_update_status(root, result)
        return result

    archive_text = str(plan_payload.get("archivePath") or saved_status.get("archivePath") or "")
    if not archive_text.strip():
        result = _portable_pending_update_error(
            root,
            saved_status,
            "CAI portable update archive path is missing.",
        )
        _write_update_status(root, result)
        return result
    archive_path = Path(archive_text).expanduser().resolve()
    if not archive_path.is_file():
        result = _portable_pending_update_error(
            root,
            saved_status,
            f"CAI portable update archive was not found: {archive_path}.",
        )
        _write_update_status(root, result)
        return result

    manifest = saved_status.get("manifest")
    expected_sha256 = ""
    if isinstance(manifest, dict):
        expected_sha256 = str(manifest.get("archiveSha256") or "").strip().lower()
    if expected_sha256 and _is_sha256_hex(expected_sha256):
        actual_sha256 = sha256_file(archive_path)
        if actual_sha256 != expected_sha256:
            result = _portable_pending_update_error(
                root,
                saved_status,
                "CAI portable update archive checksum does not match the manifest.",
            )
            _write_update_status(root, result)
            return result

    command = [str(item) for item in relaunch_command if str(item).strip()]
    if not command:
        planned_command = plan_payload.get("relaunchCommand", [])
        if isinstance(planned_command, str):
            command = [planned_command] if planned_command.strip() else []
        elif isinstance(planned_command, list):
            command = [str(item) for item in planned_command if str(item).strip()]
    refreshed_plan = {
        **plan_payload,
        "portableRoot": str(root),
        "archivePath": str(archive_path),
        "cancelPath": str(portable_update_cancel_path(root)),
        "parentPid": int(parent_pid or 0),
        "waitTimeoutSeconds": int(plan_payload.get("waitTimeoutSeconds") or 3600),
        "autoTerminateParent": bool(plan_payload.get("autoTerminateParent", True)),
        "relaunchCommand": command,
        "resumedAt": datetime.now(tz=UTC).isoformat(),
    }
    plan_path.write_text(
        json.dumps(refreshed_plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    batch_path = portable_update_batch_path(root)
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    batch_path.write_text(_portable_update_batch_script(), encoding="utf-8")

    result = {
        **saved_status,
        "checked": True,
        "updated": False,
        "status": "restart_pending",
        "phase": "restart_pending",
        "progress": _RESTART_PENDING_PROGRESS,
        "message": (
            "CAI portable update is already downloaded; applying it before launch."
            if start_process
            else "CAI portable update is already downloaded and waiting to be applied."
        ),
        "installKind": "portable",
        "canCancel": True,
        "cancelRequested": False,
        "restartRequired": True,
        "restartScheduled": bool(start_process),
        "archivePath": str(archive_path),
        "portableUpdatePlanPath": str(plan_path),
        "portableUpdateScriptPath": str(batch_path),
        "portableUpdateBatchPath": str(batch_path),
        "portableUpdatePowerShellPath": None,
        "portableUpdateCancelPath": str(portable_update_cancel_path(root)),
        "portableUpdateApplyLogPath": str(portable_update_apply_log_path(root)),
    }
    if start_process:
        result["portableUpdateApplyPid"] = _start_portable_update_apply_process(
            root,
            batch_path,
            plan_path,
        )
    _write_update_status(root, result)
    return result


def _wait_for_portable_update_idle(
    repo_root: Path,
    *,
    base_status: dict[str, Any],
    idle_seconds: int | None,
    timeout_seconds: int | None,
) -> tuple[bool, dict[str, Any]]:
    root = resolve_repo_root(repo_root)
    required_idle_seconds = int(idle_seconds or auto_update_idle_seconds())
    wait_timeout = int(timeout_seconds or auto_update_idle_timeout_seconds())
    deadline = time.monotonic() + max(1, wait_timeout)
    last_snapshot: dict[str, Any] = {}

    while True:
        if _portable_update_cancel_requested(root):
            raise UpdateCancelled("CAI portable update was cancelled while waiting for idle.")

        snapshot = portable_update_activity_snapshot(
            root,
            idle_seconds=required_idle_seconds,
        )
        last_snapshot = snapshot
        if bool(snapshot.get("idle")):
            return True, snapshot

        reason = str(snapshot.get("reason") or "busy")
        if reason == "active_request":
            message = "CAI update is ready; waiting for the active request to finish."
        else:
            message = "CAI update is ready; waiting until the interface is idle."
        _write_update_status(
            root,
            {
                **base_status,
                "updated": False,
                "status": "waiting_for_idle",
                "phase": "waiting_for_idle",
                "progress": max(15, int(base_status.get("progress") or 15)),
                "message": message,
                "installKind": "portable",
                "canCancel": True,
                "cancelRequested": False,
                "restartRequired": False,
                "restartScheduled": False,
                "activity": snapshot,
            },
        )

        if time.monotonic() >= deadline:
            return False, last_snapshot
        time.sleep(1.0)


def maybe_stage_portable_auto_update_on_launch(
    repo_root: Path,
    *,
    relaunch_command: list[str] | tuple[str, ...],
    parent_pid: int | None = None,
    base_url: str | None = None,
    timeout_sec: int = 10,
    start_process: bool = True,
    wait_for_idle: bool = True,
    idle_seconds: int | None = None,
    idle_timeout_sec: int | None = None,
) -> dict[str, Any]:
    if not auto_update_enabled():
        result = {
            "checked": False,
            "updated": False,
            "status": "disabled",
            "phase": "disabled",
            "progress": 0,
            "message": "CAI auto-update is disabled.",
            "canCancel": False,
            "cancelRequested": False,
            "restartRequired": False,
            "restartScheduled": False,
        }
        _write_update_status(resolve_repo_root(repo_root), result)
        return result

    root = resolve_repo_root(repo_root)
    if detect_local_update_install_kind(root) != "portable":
        result = {
            "checked": False,
            "updated": False,
            "status": "unsupported",
            "phase": "unsupported",
            "progress": 0,
            "message": "Deferred portable auto-update only supports portable installs.",
            "canCancel": False,
            "cancelRequested": False,
            "restartRequired": False,
            "restartScheduled": False,
        }
        _write_update_status(root, result)
        return result

    pending_result = resume_pending_portable_update_on_launch(
        root,
        relaunch_command=relaunch_command,
        parent_pid=parent_pid,
        start_process=start_process,
    )
    if pending_result is not None:
        return pending_result

    _clear_portable_update_cancel_marker(root)
    _write_update_status(
        root,
        {
            "checked": False,
            "updated": False,
            "status": "checking",
            "phase": "checking",
            "progress": 5,
            "message": "Checking for a CAI portable update...",
            "installKind": "portable",
            "canCancel": True,
            "cancelRequested": False,
            "restartRequired": False,
            "restartScheduled": False,
        },
    )

    try:
        check_result = check_for_updates(root, base_url=base_url, timeout_sec=timeout_sec)
    except Exception as exc:  # noqa: BLE001
        result = {
            "checked": True,
            "updated": False,
            "status": "error",
            "phase": "error",
            "progress": 0,
            "message": f"CAI auto-update check failed: {exc}",
            "canCancel": False,
            "cancelRequested": False,
            "restartRequired": False,
            "restartScheduled": False,
        }
        _write_update_status(root, result)
        return result

    if _portable_update_cancel_requested(root):
        result = _portable_update_cancelled_status(
            root,
            check_result,
            message="CAI portable update was cancelled before download.",
        )
        _write_update_status(root, result)
        return result

    if not check_result.get("updateAvailable"):
        return check_result
    if not check_result.get("canApply"):
        result = {
            **check_result,
            "updated": False,
            "status": "skipped",
            "phase": "skipped",
            "progress": 0,
            "message": str(check_result.get("applyReason") or "CAI auto-update skipped."),
            "canCancel": False,
            "cancelRequested": False,
            "restartRequired": False,
            "restartScheduled": False,
        }
        _write_update_status(root, result)
        return result

    if wait_for_idle:
        try:
            idle_ready, activity_snapshot = _wait_for_portable_update_idle(
                root,
                base_status=check_result,
                idle_seconds=idle_seconds,
                timeout_seconds=idle_timeout_sec,
            )
        except UpdateCancelled:
            result = _portable_update_cancelled_status(
                root,
                check_result,
                message="CAI portable update was cancelled while waiting for idle.",
            )
            _write_update_status(root, result)
            return result
        if not idle_ready:
            result = {
                **check_result,
                "updated": False,
                "status": "deferred",
                "phase": "waiting_for_idle",
                "progress": max(15, int(check_result.get("progress") or 15)),
                "message": "CAI portable update is ready but was deferred until the interface is idle.",
                "canCancel": False,
                "cancelRequested": False,
                "restartRequired": False,
                "restartScheduled": False,
                "activity": activity_snapshot,
            }
            _write_update_status(root, result)
            return result

    remote_manifest = check_result.get("manifest")
    if not isinstance(remote_manifest, dict):
        try:
            remote_manifest = fetch_remote_update_manifest(
                base_url=base_url or str(check_result.get("baseUrl") or ""),
                install_kind="portable",
                timeout_sec=timeout_sec,
            )
            validate_update_manifest(remote_manifest, require_archive_hash=True)
        except Exception as exc:  # noqa: BLE001
            result = {
                **check_result,
                "updated": False,
                "status": "error",
                "phase": "error",
                "progress": 0,
                "message": f"CAI auto-update check failed: {exc}",
                "canCancel": False,
                "cancelRequested": False,
                "restartRequired": False,
                "restartScheduled": False,
            }
            _write_update_status(root, result)
            return result

    try:
        download_dir = root / ".cai-update-cache"
        download_dir.mkdir(parents=True, exist_ok=True)
        total_bytes = _manifest_archive_size_bytes(remote_manifest)

        def write_download_progress(snapshot: dict[str, Any]) -> None:
            current_progress = int(snapshot.get("progress") or _DOWNLOAD_PROGRESS_START)
            downloaded_bytes = int(snapshot.get("downloadedBytes") or 0)
            current_total_bytes = snapshot.get("totalBytes")
            download_percent = snapshot.get("downloadPercent")
            speed_bytes_per_sec = snapshot.get("downloadSpeedBytesPerSec")
            if isinstance(download_percent, (int, float)):
                message = (
                    "Downloading a CAI portable update "
                    f"({download_percent:.0f}% of archive)..."
                )
            elif current_total_bytes:
                message = "Downloading a CAI portable update archive..."
            else:
                message = (
                    "Downloading a CAI portable update "
                    f"({downloaded_bytes} bytes received)..."
                )
            _write_update_status(
                root,
                {
                    **check_result,
                    "updated": False,
                    "status": "downloading",
                    "phase": "downloading",
                    "progress": current_progress,
                    "message": message,
                    "installKind": "portable",
                    "canCancel": True,
                    "cancelRequested": False,
                    "restartRequired": False,
                    "restartScheduled": False,
                    "downloadedBytes": downloaded_bytes,
                    "totalBytes": current_total_bytes,
                    "downloadPercent": download_percent,
                    "downloadSpeedBytesPerSec": speed_bytes_per_sec,
                },
            )

        _write_update_status(
            root,
            {
                **check_result,
                "updated": False,
                "status": "downloading",
                "phase": "downloading",
                "progress": _DOWNLOAD_PROGRESS_START,
                "message": "Downloading a CAI portable update...",
                "installKind": "portable",
                "canCancel": True,
                "cancelRequested": False,
                "restartRequired": False,
                "restartScheduled": False,
                "downloadedBytes": 0,
                "totalBytes": total_bytes,
                "downloadPercent": 0 if total_bytes else None,
                "downloadSpeedBytesPerSec": 0,
            },
        )
        archive_path = _download_update_archive(
            remote_manifest,
            download_dir,
            timeout_sec=timeout_sec,
            progress_callback=write_download_progress,
            cancel_requested=lambda: _portable_update_cancel_requested(root),
        )
        if _portable_update_cancel_requested(root):
            result = _portable_update_cancelled_status(
                root,
                check_result,
                message="CAI portable update was cancelled after download.",
                archive_path=archive_path,
            )
            _write_update_status(root, result)
            return result
        return schedule_portable_update_after_exit(
            root,
            archive_path,
            relaunch_command=list(relaunch_command),
            parent_pid=parent_pid,
            remote_manifest=remote_manifest,
            check_result=check_result,
            start_process=start_process,
        )
    except UpdateCancelled:
        result = _portable_update_cancelled_status(
            root,
            check_result,
            message="CAI portable update was cancelled during download.",
        )
        _write_update_status(root, result)
        return result
    except Exception as exc:  # noqa: BLE001
        result = {
            **check_result,
            "updated": False,
            "status": "error",
            "phase": "error",
            "progress": 0,
            "message": f"CAI portable auto-update could not be staged: {exc}",
            "canCancel": False,
            "cancelRequested": False,
            "restartRequired": False,
            "restartScheduled": False,
        }
        _write_update_status(root, result)
        return result


def cancel_pending_portable_update(repo_root: Path | None = None) -> dict[str, Any]:
    root = resolve_repo_root(repo_root)
    saved_status = _read_update_status(root)
    status = str(saved_status.get("status") or "").strip().lower()
    plan_path = portable_update_plan_path(root)
    is_pending = (
        status in _PORTABLE_UPDATE_CANCELABLE_STATUSES
        or bool(saved_status.get("restartRequired"))
        or plan_path.is_file()
    )
    if not is_pending:
        return {
            **saved_status,
            "cancelled": False,
            "status": status or "not_pending",
            "message": "No pending CAI portable update to cancel.",
            "canCancel": False,
            "cancelRequested": False,
        }

    cancel_path = portable_update_cancel_path(root)
    cancel_path.parent.mkdir(parents=True, exist_ok=True)
    cancel_payload = {
        "schemaVersion": 1,
        "kind": "cai-portable-update-cancel",
        "createdAt": datetime.now(tz=UTC).isoformat(),
        "previousStatus": status or None,
        "portableRoot": str(root),
    }
    cancel_path.write_text(
        json.dumps(cancel_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    result = _portable_update_cancelled_status(
        root,
        saved_status,
        message="CAI portable update was cancelled by the user.",
    )
    _write_update_status(root, result)
    persisted = _read_update_status(root)
    return {
        **persisted,
        "cancelled": True,
    }


def build_local_update_summary(repo_root: Path | None = None) -> dict[str, Any]:
    root = resolve_repo_root(repo_root)
    local_state = collect_local_update_state(root)
    saved_status = _read_update_status(root)

    source_kind = None
    repository = saved_status.get("repository")
    target_branch = saved_status.get("targetBranch")
    source_url = saved_status.get("sourceUrl")
    base_url = saved_status.get("baseUrl")
    source_resolution_error: dict[str, str] | None = None
    try:
        source = resolve_update_source(root)
    except Exception as exc:
        source = None
        source_resolution_error = {
            "errorType": type(exc).__name__,
            "message": str(exc),
        }
    if isinstance(source, GitHubUpdateSource):
        source_kind = "github"
        repository = repository or source.repository
        target_branch = target_branch or source.branch
        source_url = source_url or f"{source.repo_url.removesuffix('.git')}/tree/{source.branch}"
    elif isinstance(source, ValidatorUpdateSource):
        source_kind = "validator"
        base_url = base_url or source.base_url
        source_url = source_url or source.base_url

    if saved_status:
        status = str(saved_status.get("status") or "").strip() or None
        message = saved_status.get("message")
    elif not auto_update_enabled():
        status = "disabled"
        message = "CAI auto-update is disabled."
    elif source_resolution_error is not None:
        status = "error"
        message = (
            "CAI auto-update source could not be resolved: "
            f"{source_resolution_error['message']}"
        )
    elif source_kind is None:
        status = "unconfigured"
        message = (
            "GitHub auto-update is not configured yet. "
            f"Set {UPDATE_GITHUB_REPOSITORY_ENV} or configure git origin."
        )
    else:
        can_apply, apply_reason = _can_apply_update(
            local_state,
            target_channel=source_kind,
            target_branch=target_branch,
            target_install_kind=str(saved_status.get("remoteInstallKind") or local_state.install_kind),
        )
        status = "ready" if can_apply else "skipped"
        message = (
            "Automatic update will run on the next launch."
            if can_apply
            else apply_reason
        )

    can_apply, apply_reason = _can_apply_update(
        local_state,
        target_channel=source_kind,
        target_branch=target_branch,
        target_install_kind=str(saved_status.get("remoteInstallKind") or local_state.install_kind),
    )
    activity_snapshot = portable_update_activity_snapshot(root)
    return {
        "runtime": {
            "installKind": local_state.install_kind,
            "version": local_state.version or __version__,
            "versionLabel": build_runtime_version_label(
                local_state.version,
                git_commit=local_state.git_commit,
                build_id=local_state.build_id,
                build_number_label=local_state.build_number_label,
            ),
            "gitCommit": local_state.git_commit,
            "gitBranch": local_state.git_branch,
            "gitDirty": local_state.git_dirty,
            "buildId": local_state.build_id,
            "buildNumber": local_state.build_number,
            "buildNumberLabel": local_state.build_number_label,
        },
        "updates": {
            "autoUpdateEnabled": auto_update_enabled(),
            "channel": source_kind or saved_status.get("channel"),
            "provider": saved_status.get("provider") or source_kind,
            "installKind": local_state.install_kind,
            "repository": repository,
            "targetBranch": target_branch,
            "sourceUrl": source_url,
            "baseUrl": base_url,
            "checked": bool(saved_status.get("checked", False)),
            "checkedAt": saved_status.get("checkedAt"),
            "lastUpdatedAt": saved_status.get("lastUpdatedAt"),
            "updated": bool(saved_status.get("updated", False)),
            "updateAvailable": bool(saved_status.get("updateAvailable", False)),
            "remoteGitCommit": saved_status.get("remoteGitCommit"),
            "remoteGitBranch": saved_status.get("remoteGitBranch"),
            "remoteVersion": saved_status.get("remoteVersion"),
            "remoteBuildId": saved_status.get("remoteBuildId"),
            "remoteInstallKind": saved_status.get("remoteInstallKind"),
            "status": status,
            "phase": saved_status.get("phase") or status,
            "progress": saved_status.get("progress"),
            "downloadedBytes": saved_status.get("downloadedBytes"),
            "totalBytes": saved_status.get("totalBytes"),
            "downloadPercent": saved_status.get("downloadPercent"),
            "downloadSpeedBytesPerSec": saved_status.get("downloadSpeedBytesPerSec"),
            "message": message,
            "canApply": can_apply,
            "applyReason": apply_reason,
            "canCancel": bool(saved_status.get("canCancel", False)),
            "cancelRequested": bool(saved_status.get("cancelRequested", False)),
            "restartScheduled": bool(saved_status.get("restartScheduled", False)),
            "restartRequired": bool(saved_status.get("restartRequired", False)),
            "archivePath": saved_status.get("archivePath"),
            "portableUpdatePlanPath": saved_status.get("portableUpdatePlanPath"),
            "portableUpdateScriptPath": saved_status.get("portableUpdateScriptPath"),
            "portableUpdateCancelPath": saved_status.get("portableUpdateCancelPath"),
            "sourceResolutionError": source_resolution_error,
            "dashboardBuildStatus": saved_status.get("dashboardBuildStatus"),
            "dashboardBuildMessage": saved_status.get("dashboardBuildMessage"),
            "activity": saved_status.get("activity") or activity_snapshot,
        },
    }


def maybe_auto_update_on_launch(
    repo_root: Path,
    *,
    base_url: str | None = None,
    timeout_sec: int = 10,
) -> dict[str, Any]:
    if not auto_update_enabled():
        result = {
            "checked": False,
            "updated": False,
            "status": "disabled",
            "message": "CAI auto-update is disabled.",
        }
        _write_update_status(resolve_repo_root(repo_root), result)
        return result

    root = resolve_repo_root(repo_root)
    install_kind = detect_local_update_install_kind(root)
    if install_kind == "unknown":
        result = {
            "checked": False,
            "updated": False,
            "status": "unsupported",
            "message": "CAI auto-update could not detect a supported install type.",
        }
        _write_update_status(root, result)
        return result

    try:
        check_result = check_for_updates(root, base_url=base_url, timeout_sec=timeout_sec)
    except Exception as exc:  # noqa: BLE001
        result = {
            "checked": True,
            "updated": False,
            "status": "error",
            "message": f"CAI auto-update check failed: {exc}",
        }
        _write_update_status(root, result)
        return result

    if not check_result.get("updateAvailable"):
        return check_result

    if not check_result.get("canApply"):
        result = {
            **check_result,
            "updated": False,
            "status": "skipped",
            "message": str(check_result.get("applyReason") or "CAI auto-update skipped."),
        }
        _write_update_status(root, result)
        return result

    try:
        return apply_remote_update(root, base_url=base_url, timeout_sec=timeout_sec)
    except Exception as exc:  # noqa: BLE001
        result = {
            **check_result,
            "updated": False,
            "status": "error",
            "message": f"CAI auto-update skipped: {exc}",
        }
        _write_update_status(root, result)
        return result


def schedule_portable_update_after_exit(
    portable_root: Path,
    archive_path: Path,
    *,
    relaunch_command: list[str] | tuple[str, ...],
    parent_pid: int | None = None,
    remote_manifest: dict[str, Any] | None = None,
    check_result: dict[str, Any] | None = None,
    start_process: bool = True,
) -> dict[str, Any]:
    root = resolve_repo_root(portable_root)
    if not portable_install_looks_valid(root):
        raise UpdateError(f"CAI portable runtime not found at {root}.")
    resolved_archive_path = archive_path.expanduser().resolve()
    if not resolved_archive_path.is_file():
        raise UpdateError(f"Portable update archive was not found: {resolved_archive_path}.")

    plan_path = portable_update_plan_path(root)
    batch_path = portable_update_batch_path(root)
    cancel_path = portable_update_cancel_path(root)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    batch_path.parent.mkdir(parents=True, exist_ok=True)

    command = [str(item) for item in relaunch_command if str(item).strip()]
    manifest = dict(remote_manifest or {})
    base = dict(check_result or {})
    plan = {
        "schemaVersion": 1,
        "kind": "cai-portable-deferred-update",
        "createdAt": datetime.now(tz=UTC).isoformat(),
        "portableRoot": str(root),
        "archivePath": str(resolved_archive_path),
        "cancelPath": str(cancel_path),
        "parentPid": int(parent_pid or 0),
        "waitTimeoutSeconds": 3600,
        "autoTerminateParent": True,
        "relaunchCommand": command,
    }
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    batch_path.write_text(_portable_update_batch_script(), encoding="utf-8")

    result = {
        **base,
        "checked": True,
        "updated": False,
        "status": "restart_pending",
        "phase": "restart_pending",
        "progress": _RESTART_PENDING_PROGRESS,
        "message": "CAI portable update downloaded; restart scheduled to apply it safely.",
        "channel": base.get("channel") or "validator",
        "provider": base.get("provider") or "validator",
        "installKind": "portable",
        "localVersion": base.get("localVersion"),
        "localBuildId": base.get("localBuildId"),
        "remoteGitCommit": manifest.get("gitCommit") or base.get("remoteGitCommit"),
        "remoteGitBranch": manifest.get("gitBranch") or base.get("remoteGitBranch"),
        "remoteVersion": manifest.get("version") or base.get("remoteVersion"),
        "remoteBuildId": manifest.get("buildId") or base.get("remoteBuildId"),
        "remoteInstallKind": "portable",
        "updateAvailable": True,
        "canApply": True,
        "applyReason": "ok",
        "canCancel": True,
        "cancelRequested": False,
        "restartScheduled": bool(start_process),
        "restartRequired": True,
        "archivePath": str(resolved_archive_path),
        "portableUpdatePlanPath": str(plan_path),
        "portableUpdateScriptPath": str(batch_path),
        "portableUpdateBatchPath": str(batch_path),
        "portableUpdatePowerShellPath": None,
        "portableUpdateCancelPath": str(cancel_path),
        "portableUpdateApplyLogPath": str(portable_update_apply_log_path(root)),
    }

    if start_process:
        result["portableUpdateApplyPid"] = _start_portable_update_apply_process(
            root,
            batch_path,
            plan_path,
        )
    _write_update_status(root, result)

    return result


def _portable_pending_update_error(
    repo_root: Path,
    base: dict[str, Any] | None,
    message: str,
) -> dict[str, Any]:
    payload = dict(base or {})
    payload.update(
        {
            "checked": True,
            "updated": False,
            "status": "error",
            "phase": "error",
            "progress": 0,
            "message": message,
            "installKind": "portable",
            "canCancel": False,
            "cancelRequested": False,
            "restartRequired": True,
            "restartScheduled": False,
            "portableUpdatePlanPath": str(portable_update_plan_path(repo_root)),
            "portableUpdateScriptPath": str(portable_update_batch_path(repo_root)),
            "portableUpdateBatchPath": str(portable_update_batch_path(repo_root)),
            "portableUpdatePowerShellPath": None,
            "portableUpdateCancelPath": str(portable_update_cancel_path(repo_root)),
            "portableUpdateApplyLogPath": str(portable_update_apply_log_path(repo_root)),
        }
    )
    return payload


def _powershell_executable() -> str:
    if os.name != "nt":
        return "powershell"
    discovered = shutil.which("powershell.exe") or shutil.which("powershell")
    if discovered:
        return discovered
    system_root = os.environ.get("SystemRoot") or os.environ.get("WINDIR")
    if system_root:
        candidate = Path(system_root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
        if candidate.is_file():
            return str(candidate)
    return "powershell.exe"


def _cmd_executable() -> str:
    if os.name != "nt":
        return "cmd"
    system_root = os.environ.get("SystemRoot") or os.environ.get("WINDIR")
    if system_root:
        candidate = Path(system_root) / "System32" / "cmd.exe"
        if candidate.is_file():
            return str(candidate)
    return shutil.which("cmd.exe") or "cmd.exe"


def _start_portable_update_apply_process(
    repo_root: Path,
    launcher_path: Path,
    plan_path: Path,
) -> int:
    root = resolve_repo_root(repo_root)
    log_path = portable_update_apply_log_path(root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if launcher_path.suffix.lower() == ".bat":
        command_line = [
            _cmd_executable(),
            "/d",
            "/c",
            str(launcher_path),
            str(plan_path),
        ]
    else:
        command_line = [
            _powershell_executable(),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(launcher_path),
            "-PlanPath",
            str(plan_path),
        ]
    creationflags = 0
    startupinfo = None
    if os.name == "nt":
        creationflags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 1)
        startupinfo.wShowWindow = 0
    with log_path.open("ab") as log_handle:
        process = subprocess.Popen(
            command_line,
            cwd=str(root),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            startupinfo=startupinfo,
        )
    return int(process.pid)


def _portable_update_batch_script() -> str:
    powershell_payload = _portable_update_powershell_script()
    return r'''@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PLAN_PATH=%~1"
if "%PLAN_PATH%"=="" set "PLAN_PATH=%SCRIPT_DIR%portable-update-plan.json"
set "CAI_UPDATE_LAUNCHER=%~f0"
set "CAI_UPDATE_PLAN=%PLAN_PATH%"

set "PS_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%PS_EXE%" set "PS_EXE=powershell.exe"

"%PS_EXE%" -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command "$marker = [string][char]35 + ' CAI_PORTABLE_UPDATE_POWERSHELL_PAYLOAD'; $parts = (Get-Content -Raw -LiteralPath $env:CAI_UPDATE_LAUNCHER).Split(@($marker), 2, [System.StringSplitOptions]::None); if ($parts.Count -lt 2) { throw 'CAI portable update payload was not found in launcher.' }; & ([scriptblock]::Create($parts[1])) -PlanPath $env:CAI_UPDATE_PLAN"
exit /b %ERRORLEVEL%
REM # CAI_PORTABLE_UPDATE_POWERSHELL_PAYLOAD
''' + powershell_payload


def _portable_update_powershell_script() -> str:
    return r'''param(
    [Parameter(Mandatory = $true)]
    [string]$PlanPath
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)

function Write-CaiJson {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [object]$Value
    )
    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText(
        $Path,
        (($Value | ConvertTo-Json -Depth 20) + "`n"),
        $utf8NoBom
    )
}

function Start-CaiUpdateWindow {
    if ($null -ne $script:CaiUpdateForm) {
        return
    }
    try {
        Add-Type -AssemblyName System.Windows.Forms
        Add-Type -AssemblyName System.Drawing
        $script:CaiUpdateForm = [System.Windows.Forms.Form]::new()
        $script:CaiUpdateForm.Text = "CAI update"
        $script:CaiUpdateForm.Width = 460
        $script:CaiUpdateForm.Height = 150
        $script:CaiUpdateForm.StartPosition = "CenterScreen"
        $script:CaiUpdateForm.TopMost = $true
        $script:CaiUpdateForm.ControlBox = $false
        $script:CaiUpdateForm.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::FixedDialog

        $script:CaiUpdateLabel = [System.Windows.Forms.Label]::new()
        $script:CaiUpdateLabel.Left = 16
        $script:CaiUpdateLabel.Top = 18
        $script:CaiUpdateLabel.Width = 410
        $script:CaiUpdateLabel.Height = 42
        $script:CaiUpdateLabel.Text = "Preparing CAI update..."
        $script:CaiUpdateLabel.AutoEllipsis = $true
        $script:CaiUpdateForm.Controls.Add($script:CaiUpdateLabel)

        $script:CaiUpdateProgress = [System.Windows.Forms.ProgressBar]::new()
        $script:CaiUpdateProgress.Left = 16
        $script:CaiUpdateProgress.Top = 72
        $script:CaiUpdateProgress.Width = 410
        $script:CaiUpdateProgress.Height = 20
        $script:CaiUpdateProgress.Minimum = 0
        $script:CaiUpdateProgress.Maximum = 100
        $script:CaiUpdateForm.Controls.Add($script:CaiUpdateProgress)

        [void]$script:CaiUpdateForm.Show()
        [System.Windows.Forms.Application]::DoEvents()
    } catch {
        $script:CaiUpdateForm = $null
    }
}

function Set-CaiUpdateWindow {
    param(
        [Parameter(Mandatory = $true)][string]$Message,
        [int]$Progress = 0
    )
    Start-CaiUpdateWindow
    if ($null -eq $script:CaiUpdateForm) {
        return
    }
    try {
        $script:CaiUpdateLabel.Text = $Message
        $script:CaiUpdateProgress.Value = [Math]::Max(0, [Math]::Min(100, $Progress))
        [System.Windows.Forms.Application]::DoEvents()
    } catch {
    }
}

function Close-CaiUpdateWindow {
    if ($null -eq $script:CaiUpdateForm) {
        return
    }
    try {
        $script:CaiUpdateForm.Close()
        $script:CaiUpdateForm.Dispose()
        [System.Windows.Forms.Application]::DoEvents()
    } catch {
    }
    $script:CaiUpdateForm = $null
}

function Show-CaiUpdateError {
    param([Parameter(Mandatory = $true)][string]$Message)
    try {
        Add-Type -AssemblyName System.Windows.Forms
        [void][System.Windows.Forms.MessageBox]::Show(
            $Message,
            "CAI update failed",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Warning
        )
    } catch {
    }
}

function Test-CaiPortableRoot {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (
        (Test-Path -LiteralPath (Join-Path $Path "CAI.exe")) -or
        (Test-Path -LiteralPath (Join-Path $Path "cai.exe")) -or
        (Test-Path -LiteralPath (Join-Path $Path "runtime\cai"))
    )
}

$plan = Get-Content -LiteralPath $PlanPath -Raw | ConvertFrom-Json
$root = [System.IO.Path]::GetFullPath([string]$plan.portableRoot)
$archive = [System.IO.Path]::GetFullPath([string]$plan.archivePath)
$statusPath = Join-Path $root ".cai-update\status.json"
$rollbackPath = Join-Path $root ".cai-update\rollback.json"
$stage = Join-Path $root ".cai-update\stage-portable"
$cancelPath = ""
if ($null -ne $plan.cancelPath) {
    $cancelPath = [System.IO.Path]::GetFullPath([string]$plan.cancelPath)
}
$waitTimeout = 3600
if ($null -ne $plan.waitTimeoutSeconds) {
    $waitTimeout = [int]$plan.waitTimeoutSeconds
}
$autoTerminateParent = $true
if ($null -ne $plan.autoTerminateParent) {
    $autoTerminateParent = [bool]$plan.autoTerminateParent
}

function Test-CaiUpdateCancelled {
    if ([string]::IsNullOrWhiteSpace($cancelPath)) {
        return $false
    }
    return (Test-Path -LiteralPath $cancelPath)
}

function Stop-CaiUpdateIfCancelled {
    if (-not (Test-CaiUpdateCancelled)) {
        return
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $statusPath) | Out-Null
    Write-CaiJson -Path $statusPath -Value ([ordered]@{
        checked = $true
        updated = $false
        status = "cancelled"
        phase = "cancelled"
        progress = 0
        message = "CAI portable update was cancelled by the user."
        installKind = "portable"
        canCancel = $false
        cancelRequested = $true
        restartRequired = $false
        restartScheduled = $false
        archivePath = $archive
        portableRoot = $root
        portableUpdateCancelPath = $cancelPath
        checkedAt = [DateTimeOffset]::UtcNow.ToString("o")
    })
    exit 0
}

function Format-CaiBytes {
    param([Int64]$Bytes)
    if ($Bytes -ge 1GB) {
        return ("{0:N1} GB" -f ($Bytes / 1GB))
    }
    if ($Bytes -ge 1MB) {
        return ("{0:N0} MB" -f ($Bytes / 1MB))
    }
    return ("{0:N0} bytes" -f $Bytes)
}

function Get-CaiZipUncompressedBytes {
    param([Parameter(Mandatory = $true)][string]$Path)
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::OpenRead($Path)
    try {
        [Int64]$total = 0
        foreach ($entry in $zip.Entries) {
            $total += [Int64]$entry.Length
        }
        return $total
    } finally {
        $zip.Dispose()
    }
}

function Assert-CaiPortableUpdateDiskSpace {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Archive
    )
    $archiveBytes = ([System.IO.FileInfo]::new($Archive)).Length
    $uncompressedBytes = Get-CaiZipUncompressedBytes -Path $Archive
    [Int64]$safetyBytes = 512MB
    [Int64]$requiredBytes = [Math]::Max($archiveBytes, $uncompressedBytes) + $safetyBytes
    $driveRoot = [System.IO.Path]::GetPathRoot($Root)
    $drive = [System.IO.DriveInfo]::new($driveRoot)
    [Int64]$availableBytes = $drive.AvailableFreeSpace
    if ($availableBytes -lt $requiredBytes) {
        throw (
            "Not enough free disk space to apply CAI portable update. " +
            "Required at least $(Format-CaiBytes $requiredBytes), " +
            "available $(Format-CaiBytes $availableBytes) on $driveRoot. " +
            "Free disk space and start CAI again."
        )
    }
}

function Start-CaiRelaunch {
    param([string]$Reason = "updated")
    $relaunchCommand = @($plan.relaunchCommand)
    if ($relaunchCommand.Count -le 0) {
        return
    }
    try {
        $env:CAI_AUTO_UPDATE_RESTARTED = "1"
        $env:CAI_PORTABLE_UPDATE_RESTARTED = "1"
        $env:CAI_PORTABLE_UPDATE_RESULT = $Reason
        $startArgs = @{
            FilePath = [string]$relaunchCommand[0]
            WorkingDirectory = $root
        }
        if ($relaunchCommand.Count -gt 1) {
            $startArgs["ArgumentList"] = @($relaunchCommand | Select-Object -Skip 1)
        }
        Start-Process @startArgs
    } catch {
    }
}

function Test-CaiPathInsideRoot {
    param(
        [Parameter(Mandatory = $true)][string]$Candidate,
        [Parameter(Mandatory = $true)][string]$Root
    )
    try {
        $trimChars = [char[]]@(
            [System.IO.Path]::DirectorySeparatorChar,
            [System.IO.Path]::AltDirectorySeparatorChar
        )
        $candidateFull = [System.IO.Path]::GetFullPath($Candidate).TrimEnd($trimChars)
        $rootFull = [System.IO.Path]::GetFullPath($Root).TrimEnd($trimChars)
        return (
            $candidateFull.Equals($rootFull, [System.StringComparison]::OrdinalIgnoreCase) -or
            $candidateFull.StartsWith(
                $rootFull + [System.IO.Path]::DirectorySeparatorChar,
                [System.StringComparison]::OrdinalIgnoreCase
            )
        )
    } catch {
        return $false
    }
}

function Get-CaiProcessId {
    param([Parameter(Mandatory = $true)]$Process)
    if ($null -ne $Process.ProcessId) {
        return [int]$Process.ProcessId
    }
    return [int]$Process.Id
}

function Get-CaiProcessName {
    param([Parameter(Mandatory = $true)]$Process)
    if (-not [string]::IsNullOrWhiteSpace([string]$Process.Name)) {
        return [string]$Process.Name
    }
    if (-not [string]::IsNullOrWhiteSpace([string]$Process.ProcessName)) {
        return [string]$Process.ProcessName
    }
    return ("pid-" + (Get-CaiProcessId -Process $Process))
}

function Get-CaiPortableRuntimeProcesses {
    param([Parameter(Mandatory = $true)][string]$Root)
    try {
        return @(
            Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
                $_.ProcessId -ne $PID -and
                -not [string]::IsNullOrWhiteSpace([string]$_.ExecutablePath) -and
                (Test-CaiPathInsideRoot -Candidate ([string]$_.ExecutablePath) -Root $Root)
            }
        )
    } catch {
        try {
            return @(
                Get-Process -ErrorAction SilentlyContinue | Where-Object {
                    $_.Id -ne $PID -and
                    -not [string]::IsNullOrWhiteSpace([string]$_.Path) -and
                    (Test-CaiPathInsideRoot -Candidate ([string]$_.Path) -Root $Root)
                }
            )
        } catch {
            return @()
        }
    }
}

function Get-CaiProcessSummary {
    param([object[]]$Processes)
    $labels = @()
    foreach ($item in @($Processes)) {
        try {
            $labels += ((Get-CaiProcessName -Process $item) + "(" + (Get-CaiProcessId -Process $item) + ")")
        } catch {
        }
    }
    if ($labels.Count -le 0) {
        return "<unknown>"
    }
    return ($labels | Select-Object -Unique) -join ", "
}

function Stop-CaiPortableRuntimeProcesses {
    param([Parameter(Mandatory = $true)][string]$Root)
    $running = @(Get-CaiPortableRuntimeProcesses -Root $Root)
    foreach ($item in $running) {
        try {
            $runtimeProcess = Get-Process -Id (Get-CaiProcessId -Process $item) -ErrorAction SilentlyContinue
            if ($null -ne $runtimeProcess -and -not $runtimeProcess.HasExited) {
                [void]$runtimeProcess.CloseMainWindow()
            }
        } catch {
        }
    }
    Start-Sleep -Seconds 3
    $running = @(Get-CaiPortableRuntimeProcesses -Root $Root)
    foreach ($item in $running) {
        try {
            Stop-Process -Id (Get-CaiProcessId -Process $item) -Force -ErrorAction SilentlyContinue
        } catch {
        }
    }
}

function Wait-CaiPortableRuntimeProcesses {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [int]$TimeoutSeconds = 30,
        [bool]$AllowTerminate = $false
    )
    $deadline = (Get-Date).AddSeconds([Math]::Max(1, $TimeoutSeconds))
    while ((Get-Date) -lt $deadline) {
        Stop-CaiUpdateIfCancelled
        $running = @(Get-CaiPortableRuntimeProcesses -Root $Root)
        if ($running.Count -le 0) {
            return
        }
        $summary = Get-CaiProcessSummary -Processes $running
        Set-CaiUpdateWindow -Message "Waiting for CAI runtime files to unlock: $summary" -Progress 69
        Write-CaiJson -Path $statusPath -Value ([ordered]@{
            checked = $true
            updated = $false
            status = "applying"
            phase = "waiting_for_processes"
            progress = 69
            message = "Waiting for CAI runtime files to unlock: $summary"
            installKind = "portable"
            canCancel = $true
            cancelRequested = $false
            restartRequired = $true
            restartScheduled = $false
            archivePath = $archive
            portableRoot = $Root
            checkedAt = [DateTimeOffset]::UtcNow.ToString("o")
        })
        Start-Sleep -Seconds 1
    }

    $running = @(Get-CaiPortableRuntimeProcesses -Root $Root)
    if ($running.Count -le 0) {
        return
    }
    if ($AllowTerminate) {
        $summary = Get-CaiProcessSummary -Processes $running
        Set-CaiUpdateWindow -Message "Closing stale CAI runtime processes: $summary" -Progress 69
        Stop-CaiPortableRuntimeProcesses -Root $Root
        $deadline = (Get-Date).AddSeconds(20)
        while ((Get-Date) -lt $deadline) {
            Stop-CaiUpdateIfCancelled
            $running = @(Get-CaiPortableRuntimeProcesses -Root $Root)
            if ($running.Count -le 0) {
                return
            }
            Start-Sleep -Milliseconds 500
        }
    }

    $running = @(Get-CaiPortableRuntimeProcesses -Root $Root)
    if ($running.Count -gt 0) {
        throw (
            "CAI portable update cannot replace files because CAI is still running: " +
            (Get-CaiProcessSummary -Processes $running) +
            ". Close CAI completely and start it again."
        )
    }
}

function Invoke-CaiFileOperationWithRetry {
    param(
        [Parameter(Mandatory = $true)][string]$Description,
        [Parameter(Mandatory = $true)][scriptblock]$Action,
        [int]$Attempts = 80,
        [int]$DelayMilliseconds = 500
    )
    $lastError = $null
    for ($attempt = 1; $attempt -le $Attempts; $attempt += 1) {
        Stop-CaiUpdateIfCancelled
        try {
            & $Action
            return
        } catch {
            $lastError = $_
            if ($attempt -ge $Attempts) {
                break
            }
            Start-Sleep -Milliseconds $DelayMilliseconds
        }
    }
    $message = "unknown error"
    if ($null -ne $lastError) {
        $message = $lastError.Exception.Message
    }
    throw "$Description failed after waiting for file locks to clear: $message"
}

try {
    $parentPid = 0
    if ($null -ne $plan.parentPid) {
        $parentPid = [int]$plan.parentPid
    }
    if ($parentPid -gt 0) {
        if ($autoTerminateParent) {
            Set-CaiUpdateWindow -Message "Closing CAI to apply update..." -Progress 68
            Write-CaiJson -Path $statusPath -Value ([ordered]@{
                checked = $true
                updated = $false
                status = "applying"
                phase = "closing"
                progress = 68
                message = "Closing CAI to apply update..."
                installKind = "portable"
                canCancel = $true
                cancelRequested = $false
                restartRequired = $true
                restartScheduled = $false
                archivePath = $archive
                portableRoot = $root
                checkedAt = [DateTimeOffset]::UtcNow.ToString("o")
            })
            Stop-CaiPortableRuntimeProcesses -Root $root
            $deadline = (Get-Date).AddSeconds(30)
            while ((Get-Date) -lt $deadline) {
                Stop-CaiUpdateIfCancelled
                $parent = Get-Process -Id $parentPid -ErrorAction SilentlyContinue
                if ($null -eq $parent) {
                    break
                }
                Start-Sleep -Milliseconds 500
            }
            $parent = Get-Process -Id $parentPid -ErrorAction SilentlyContinue
            if ($null -ne $parent) {
                Stop-Process -Id $parentPid -Force -ErrorAction SilentlyContinue
            }
        } else {
            Set-CaiUpdateWindow -Message "Waiting for CAI to close before applying update..." -Progress 68
            $deadline = (Get-Date).AddSeconds($waitTimeout)
            while ((Get-Date) -lt $deadline) {
                Stop-CaiUpdateIfCancelled
                $parent = Get-Process -Id $parentPid -ErrorAction SilentlyContinue
                if ($null -eq $parent) {
                    break
                }
                Start-Sleep -Milliseconds 750
            }
            $parent = Get-Process -Id $parentPid -ErrorAction SilentlyContinue
            if ($null -ne $parent) {
                Write-CaiJson -Path $statusPath -Value ([ordered]@{
                    checked = $true
                    updated = $false
                    status = "restart_pending"
                    phase = "restart_pending"
                    progress = 68
                    message = "CAI portable update is downloaded and waiting for CAI to close."
                    installKind = "portable"
                    canCancel = $true
                    cancelRequested = $false
                    restartRequired = $true
                    restartScheduled = $false
                    archivePath = $archive
                    portableRoot = $root
                    checkedAt = [DateTimeOffset]::UtcNow.ToString("o")
                })
                Close-CaiUpdateWindow
                exit 0
            }
        }
    }
    Stop-CaiUpdateIfCancelled
    Start-Sleep -Milliseconds 750
    Stop-CaiUpdateIfCancelled
    Wait-CaiPortableRuntimeProcesses -Root $root -TimeoutSeconds 30 -AllowTerminate $true

    if (-not (Test-CaiPortableRoot -Path $root)) {
        throw "CAI portable runtime not found at $root."
    }
    if (-not (Test-Path -LiteralPath $archive)) {
        throw "CAI portable update archive not found at $archive."
    }
    Assert-CaiPortableUpdateDiskSpace -Root $root -Archive $archive

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $statusPath) | Out-Null
    Set-CaiUpdateWindow -Message "Applying CAI portable update..." -Progress 70
    Write-CaiJson -Path $statusPath -Value ([ordered]@{
        checked = $true
        updated = $false
        status = "applying"
        phase = "applying"
        progress = 70
        message = "Applying CAI portable update..."
        installKind = "portable"
        canCancel = $false
        cancelRequested = $false
        restartRequired = $true
        restartScheduled = $false
        archivePath = $archive
        portableRoot = $root
        checkedAt = [DateTimeOffset]::UtcNow.ToString("o")
    })
    Write-CaiJson -Path $rollbackPath -Value ([ordered]@{
        status = "pending"
        updateKind = "portable"
        createdAt = [DateTimeOffset]::UtcNow.ToString("o")
        archivePath = $archive
        portableRoot = $root
    })

    if (Test-Path -LiteralPath $stage) {
        Invoke-CaiFileOperationWithRetry -Description "Remove previous portable update stage" -Action {
            Remove-Item -LiteralPath $stage -Recurse -Force
        }
    }
    New-Item -ItemType Directory -Force -Path $stage | Out-Null
    Set-CaiUpdateWindow -Message "Extracting CAI portable update archive..." -Progress 74
    Write-CaiJson -Path $statusPath -Value ([ordered]@{
        checked = $true
        updated = $false
        status = "applying"
        phase = "applying"
        progress = 74
        message = "Extracting CAI portable update archive..."
        installKind = "portable"
        canCancel = $false
        cancelRequested = $false
        restartRequired = $true
        restartScheduled = $false
        archivePath = $archive
        portableRoot = $root
        checkedAt = [DateTimeOffset]::UtcNow.ToString("o")
    })
    Invoke-CaiFileOperationWithRetry -Description "Extract portable update archive" -Attempts 5 -DelayMilliseconds 1000 -Action {
        [System.IO.Compression.ZipFile]::ExtractToDirectory($archive, $stage)
    }
    Set-CaiUpdateWindow -Message "Preparing CAI file replacement..." -Progress 82
    Write-CaiJson -Path $statusPath -Value ([ordered]@{
        checked = $true
        updated = $false
        status = "applying"
        phase = "applying"
        progress = 82
        message = "CAI portable update extracted; preparing file replacement..."
        installKind = "portable"
        canCancel = $false
        cancelRequested = $false
        restartRequired = $true
        restartScheduled = $false
        archivePath = $archive
        portableRoot = $root
        checkedAt = [DateTimeOffset]::UtcNow.ToString("o")
    })

    $payloadRoot = $stage
    $stageChildren = @(Get-ChildItem -LiteralPath $stage -Force)
    if ($stageChildren.Count -eq 1 -and $stageChildren[0].PSIsContainer) {
        $candidate = $stageChildren[0].FullName
        if (Test-CaiPortableRoot -Path $candidate) {
            $payloadRoot = $candidate
        }
    }
    if (-not (Test-CaiPortableRoot -Path $payloadRoot)) {
        throw "Portable update archive does not contain a CAI portable runtime."
    }

    $preserve = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::OrdinalIgnoreCase
    )
    foreach ($name in @(
        ".cai",
        ".cai-local",
        ".cai-local-testnet",
        ".cai-update",
        ".cai-update-cache",
        ".cai-api-token",
        ".cai-peer-book.json",
        "data",
        "cai_log",
        "desktop.log",
        "event_log",
        "wallets.json",
        "session.json",
        "ledger.json",
        "chain.json",
        "chain-index.json",
        "chain-snapshots.json",
        "journal.jsonl",
        "node-config.json",
        "settlements.json",
        "worker-payouts.json",
        "job-intents.json",
        "execution-receipts.json",
        "unlocked-wallet-signing-key.json"
    )) {
        [void]$preserve.Add($name)
    }

    $deletedEntries = 0
    foreach ($child in @(Get-ChildItem -LiteralPath $root -Force)) {
        if ($preserve.Contains($child.Name)) {
            continue
        }
        $childPath = $child.FullName
        Invoke-CaiFileOperationWithRetry -Description "Remove old CAI portable entry $($child.Name)" -Action {
            Remove-Item -LiteralPath $childPath -Recurse -Force
        }
        $deletedEntries += 1
    }
    Set-CaiUpdateWindow -Message "Copying updated CAI runtime files..." -Progress 88
    Write-CaiJson -Path $statusPath -Value ([ordered]@{
        checked = $true
        updated = $false
        status = "applying"
        phase = "applying"
        progress = 88
        message = "Old CAI portable runtime files removed; copying update..."
        installKind = "portable"
        canCancel = $false
        cancelRequested = $false
        restartRequired = $true
        restartScheduled = $false
        archivePath = $archive
        portableRoot = $root
        deletedEntryCount = $deletedEntries
        checkedAt = [DateTimeOffset]::UtcNow.ToString("o")
    })

    $writtenEntries = 0
    foreach ($child in @(Get-ChildItem -LiteralPath $payloadRoot -Force)) {
        if ($preserve.Contains($child.Name)) {
            continue
        }
        $sourcePath = $child.FullName
        $destinationPath = Join-Path $root $child.Name
        Invoke-CaiFileOperationWithRetry -Description "Copy updated CAI portable entry $($child.Name)" -Action {
            Copy-Item -LiteralPath $sourcePath -Destination $destinationPath -Recurse -Force
        }
        $writtenEntries += 1
    }
    Set-CaiUpdateWindow -Message "Finalizing CAI portable update..." -Progress 96
    Write-CaiJson -Path $statusPath -Value ([ordered]@{
        checked = $true
        updated = $false
        status = "applying"
        phase = "applying"
        progress = 96
        message = "Finalizing CAI portable update..."
        installKind = "portable"
        canCancel = $false
        cancelRequested = $false
        restartRequired = $true
        restartScheduled = $false
        archivePath = $archive
        portableRoot = $root
        writtenEntryCount = $writtenEntries
        deletedEntryCount = $deletedEntries
        checkedAt = [DateTimeOffset]::UtcNow.ToString("o")
    })

    if (Test-Path -LiteralPath $stage) {
        Invoke-CaiFileOperationWithRetry -Description "Remove portable update stage" -Action {
            Remove-Item -LiteralPath $stage -Recurse -Force
        }
    }

    Write-CaiJson -Path $rollbackPath -Value ([ordered]@{
        status = "applied"
        updateKind = "portable"
        appliedAt = [DateTimeOffset]::UtcNow.ToString("o")
        archivePath = $archive
        portableRoot = $root
        writtenEntryCount = $writtenEntries
        deletedEntryCount = $deletedEntries
    })
    Write-CaiJson -Path $statusPath -Value ([ordered]@{
        checked = $true
        updated = $true
        status = "updated"
        phase = "updated"
        progress = 100
        message = "CAI portable runtime updated from release package."
        installKind = "portable"
        canCancel = $false
        cancelRequested = $false
        restartRequired = $false
        restartScheduled = $false
        archivePath = $archive
        writtenEntryCount = $writtenEntries
        deletedEntryCount = $deletedEntries
        lastUpdatedAt = [DateTimeOffset]::UtcNow.ToString("o")
    })

    Set-CaiUpdateWindow -Message "CAI update complete. Starting CAI..." -Progress 100
    Start-CaiRelaunch -Reason "updated"
    Start-Sleep -Milliseconds 750
    Close-CaiUpdateWindow
    exit 0
} catch {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $statusPath) | Out-Null
    $message = $_.Exception.Message
    Set-CaiUpdateWindow -Message "CAI update failed. Starting previous CAI..." -Progress 0
    Write-CaiJson -Path $rollbackPath -Value ([ordered]@{
        status = "failed"
        updateKind = "portable"
        failedAt = [DateTimeOffset]::UtcNow.ToString("o")
        error = $message
        archivePath = $archive
        portableRoot = $root
    })
    Write-CaiJson -Path $statusPath -Value ([ordered]@{
        checked = $true
        updated = $false
        status = "error"
        phase = "error"
        progress = 0
        message = "CAI portable auto-update failed: $message"
        installKind = "portable"
        canCancel = $false
        cancelRequested = $false
        restartRequired = $true
        restartScheduled = $false
    })
    Start-CaiRelaunch -Reason "failed"
    Show-CaiUpdateError -Message ("CAI update failed. " + $message)
    Close-CaiUpdateWindow
    exit 1
}
'''


def apply_update_archive(repo_root: Path, archive_path: Path) -> dict[str, Any]:
    root = resolve_repo_root(repo_root)
    local_state = collect_local_update_state(root)
    can_apply, apply_reason = _can_apply_update(local_state)
    if not can_apply:
        raise UpdateError(apply_reason)

    package_metadata = load_package_metadata_from_archive(archive_path)
    validate_update_package_metadata(package_metadata)
    raw_files = package_metadata.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise UpdateError(f"Update package {archive_path} does not contain any files.")

    tracked_files = {
        str(path).replace("\\", "/")
        for path in (package_metadata.get("trackedFiles") or [])
        if str(path).strip()
    }
    generated_roots = [
        _normalize_update_package_path(str(path).replace("\\", "/"))
        for path in (package_metadata.get("generatedRoots") or [])
        if str(path).strip()
    ]

    written_files = 0
    deleted_files: list[str] = []
    stage_root = _prepare_update_stage_dir(root)
    backup_root = _prepare_update_rollback_backup_dir(root)
    backed_up_files: list[str] = []
    created_files: list[str] = []
    _write_update_rollback_marker(
        root,
        {
            "status": "pending",
            "createdAt": datetime.now(tz=UTC).isoformat(),
            "archivePath": str(archive_path),
            "repoRoot": str(root),
            "previousGitCommit": local_state.git_commit,
            "previousGitBranch": local_state.git_branch,
            "packageGitCommit": package_metadata.get("gitCommit"),
            "packageGitBranch": package_metadata.get("gitBranch"),
            "packageVersion": package_metadata.get("version"),
            "trackedFileCount": len(tracked_files),
            "generatedRoots": generated_roots,
            "backupPath": str(backup_root),
        },
    )
    try:
        with zipfile.ZipFile(archive_path, mode="r") as bundle:
            _validate_update_archive_members(bundle)
            bundle.extractall(stage_root)

        current_tracked = set(local_state.tracked_files)
        for relative_path in sorted(current_tracked.difference(tracked_files)):
            if _is_forbidden_package_path(relative_path):
                continue
            target_path = root / Path(relative_path)
            if target_path.exists():
                _backup_update_target(root, target_path, backup_root, backed_up_files)
                target_path.unlink()
                deleted_files.append(relative_path)

        for relative_root in generated_roots:
            target_root = root / Path(relative_root)
            if target_root.exists():
                shutil.rmtree(target_root)

        written_files = 0
        for file_entry in raw_files:
            if not isinstance(file_entry, dict):
                continue
            raw_relative_path = str(file_entry.get("path") or "").replace("\\", "/").strip()
            if not raw_relative_path:
                continue
            relative_path = _normalize_update_package_path(raw_relative_path)
            _raise_if_forbidden_package_path(relative_path)
            source_path = stage_root / Path(relative_path)
            if not source_path.is_file():
                raise UpdateError(f"Archive payload is missing {relative_path}.")

            target_path = root / Path(relative_path)
            if target_path.exists():
                _backup_update_target(root, target_path, backup_root, backed_up_files)
            else:
                created_files.append(relative_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            try:
                target_path.chmod(int(file_entry.get("mode") or 0o644))
            except Exception:
                pass
            written_files += 1
    except Exception as exc:
        rollback_status = "failed"
        rollback_error: str | None = None
        try:
            _restore_update_backup(root, backup_root, backed_up_files, created_files)
            rollback_status = "rolled_back"
        except Exception as rollback_exc:  # noqa: BLE001
            rollback_status = "rollback_failed"
            rollback_error = str(rollback_exc)
        _update_rollback_marker(
            root,
            {
                "status": rollback_status,
                "failedAt": datetime.now(tz=UTC).isoformat(),
                "error": str(exc),
                "rollbackError": rollback_error,
                "backedUpFiles": backed_up_files,
                "createdFiles": created_files,
                "writtenFileCount": written_files,
                "deletedFileCount": len(deleted_files),
            },
        )
        raise
    else:
        _update_rollback_marker(
            root,
            {
                "status": "applied",
                "appliedAt": datetime.now(tz=UTC).isoformat(),
                "backedUpFileCount": len(backed_up_files),
                "createdFileCount": len(created_files),
                "writtenFileCount": written_files,
                "deletedFileCount": len(deleted_files),
            },
        )
        _clear_update_stage_dir(root)
        _clear_update_rollback_backup_dir(root)

    return {
        "message": "CAI source checkout updated from validator package.",
        "archivePath": str(archive_path),
        "localGitCommit": package_metadata.get("gitCommit"),
        "writtenFileCount": written_files,
        "deletedFileCount": len(deleted_files),
        "rollbackMarkerPath": str(update_rollback_marker_path(root)),
    }


def apply_portable_update_archive(portable_root: Path, archive_path: Path) -> dict[str, Any]:
    root = resolve_repo_root(portable_root)
    if not portable_install_looks_valid(root):
        raise UpdateError(f"CAI portable runtime not found at {root}.")

    written_entries = 0
    deleted_entries: list[str] = []
    stage_root = _prepare_update_stage_dir(root)
    _write_update_rollback_marker(
        root,
        {
            "status": "pending",
            "updateKind": "portable",
            "createdAt": datetime.now(tz=UTC).isoformat(),
            "archivePath": str(archive_path),
            "portableRoot": str(root),
        },
    )
    try:
        with zipfile.ZipFile(archive_path, mode="r") as bundle:
            _validate_update_archive_members(bundle)
            bundle.extractall(stage_root)

        payload_root = _resolve_portable_update_payload_root(stage_root)
        for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            relative_name = child.name
            if _is_forbidden_package_path(relative_name):
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            deleted_entries.append(relative_name)

        for child in sorted(payload_root.iterdir(), key=lambda item: item.name.lower()):
            relative_name = child.name
            if _is_forbidden_package_path(relative_name):
                continue
            destination = root / relative_name
            if child.is_dir():
                shutil.copytree(child, destination)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(child, destination)
            written_entries += 1
    except Exception as exc:
        _update_rollback_marker(
            root,
            {
                "status": "failed",
                "failedAt": datetime.now(tz=UTC).isoformat(),
                "error": str(exc),
                "writtenEntryCount": written_entries,
                "deletedEntryCount": len(deleted_entries),
            },
        )
        raise
    else:
        _update_rollback_marker(
            root,
            {
                "status": "applied",
                "appliedAt": datetime.now(tz=UTC).isoformat(),
                "writtenEntryCount": written_entries,
                "deletedEntryCount": len(deleted_entries),
            },
        )
        _clear_update_stage_dir(root)

    return {
        "message": "CAI portable runtime updated from release package.",
        "archivePath": str(archive_path),
        "updated": True,
        "restartRequired": True,
        "writtenEntryCount": written_entries,
        "deletedEntryCount": len(deleted_entries),
        "rollbackMarkerPath": str(update_rollback_marker_path(root)),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_update_manifest(
    manifest: dict[str, Any],
    *,
    require_archive_hash: bool = False,
    require_signature: bool | None = None,
) -> None:
    if not isinstance(manifest, dict):
        raise UpdateError("Update manifest must be a JSON object.")
    _validate_update_protocol_range(manifest, payload_name="Update manifest")
    signature_ok, signature_error = verify_update_manifest_signature(
        manifest,
        require_signature=require_signature,
    )
    if not signature_ok:
        raise UpdateError(signature_error or "Update manifest release signature is invalid.")
    if require_archive_hash:
        expected_sha256 = str(manifest.get("archiveSha256") or "").strip().lower()
        if not expected_sha256:
            raise UpdateError("Update manifest does not include archiveSha256.")
        if not _is_sha256_hex(expected_sha256):
            raise UpdateError("Update manifest archiveSha256 is not a valid SHA-256 hex digest.")

    raw_size = manifest.get("archiveSizeBytes")
    if raw_size not in (None, ""):
        try:
            archive_size = int(raw_size)
        except (TypeError, ValueError) as exc:
            raise UpdateError("Update manifest archiveSizeBytes must be an integer.") from exc
        if archive_size <= 0:
            raise UpdateError("Update manifest archiveSizeBytes must be positive.")


def validate_update_package_metadata(metadata: dict[str, Any]) -> None:
    if not isinstance(metadata, dict):
        raise UpdateError("Update package metadata must be a JSON object.")
    _validate_update_protocol_range(metadata, payload_name="Update package")
    raw_files = metadata.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise UpdateError("Update package metadata does not contain any files.")
    for file_entry in raw_files:
        if not isinstance(file_entry, dict):
            raise UpdateError("Update package metadata contains a non-object file entry.")
        raw_relative_path = str(file_entry.get("path") or "").replace("\\", "/").strip()
        if not raw_relative_path:
            raise UpdateError("Update package metadata contains an empty file path.")
        relative_path = _normalize_update_package_path(raw_relative_path)
        _raise_if_forbidden_package_path(relative_path)
    for key in ("trackedFiles", "generatedRoots"):
        raw_paths = metadata.get(key) or []
        if not isinstance(raw_paths, list):
            raise UpdateError(f"Update package metadata {key} must be a list.")
        for raw_path in raw_paths:
            relative_path = _normalize_update_package_path(str(raw_path).replace("\\", "/"))
            _raise_if_forbidden_package_path(relative_path)


def _derive_update_manifest_pq_seed(signing_seed: bytes) -> bytes:
    return hashlib.sha256(
        b"cai-update-manifest-ml-dsa-65-v1:" + bytes(signing_seed)
    ).digest()


def maybe_sign_update_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    signing_seed_b64 = str(os.getenv(UPDATE_SIGNING_SEED_ENV) or "").strip()
    if not signing_seed_b64:
        return manifest
    public_key_b64 = str(os.getenv(UPDATE_SIGNING_PUBLIC_KEY_ENV) or "").strip() or None
    return sign_update_manifest(
        manifest,
        signing_seed_b64=signing_seed_b64,
        public_key_b64=public_key_b64,
    )


def sign_update_manifest(
    manifest: dict[str, Any],
    *,
    signing_seed_b64: str,
    public_key_b64: str | None = None,
    signed_at: str | None = None,
) -> dict[str, Any]:
    signing_seed = decode_bytes(str(signing_seed_b64 or "").strip())
    normalized_public_key = str(public_key_b64 or "").strip() or public_key_b64_from_seed(
        signing_seed
    )
    signed_manifest = dict(manifest)
    signing_body = update_manifest_signing_body(signed_manifest)
    signature_b64 = sign_payload_b64(signing_seed, signing_body)
    pq_public_key_b64, pq_private_key_b64 = mldsa65_keypair_b64_from_seed(
        _derive_update_manifest_pq_seed(signing_seed)
    )
    signed_manifest["signature"] = {
        "scheme": UPDATE_SIGNATURE_SCHEME,
        "public_key_b64": normalized_public_key,
        "public_key_address": hybrid_address_from_public_keys_b64(
            ed25519_public_key_b64=normalized_public_key,
            pq_public_key_b64=pq_public_key_b64,
        ),
        "signature_b64": signature_b64,
        "pq_scheme": SIGNING_SCHEME_ML_DSA_65,
        "pq_public_key_b64": pq_public_key_b64,
        "pq_signature_b64": sign_payload_mldsa65_b64(pq_private_key_b64, signing_body),
        "signed_at": signed_at or datetime.now(tz=UTC).isoformat(),
    }
    return signed_manifest


def verify_update_manifest_signature(
    manifest: dict[str, Any],
    *,
    require_signature: bool | None = None,
) -> tuple[bool, str | None]:
    if not isinstance(manifest, dict):
        return True, None
    signature_required = (
        update_manifest_signature_required()
        if require_signature is None
        else bool(require_signature)
    )
    signature = manifest.get("signature")
    if not isinstance(signature, dict):
        if signature_required:
            return False, "Update manifest release signature is missing."
        return True, None

    scheme = str(signature.get("scheme") or "").strip()
    if scheme not in {UPDATE_SIGNATURE_SCHEME_ED25519, UPDATE_SIGNATURE_SCHEME_HYBRID}:
        return False, f"Update manifest has unsupported signature scheme '{scheme}'."
    public_key_b64 = str(signature.get("public_key_b64") or "").strip()
    signature_b64 = str(signature.get("signature_b64") or "").strip()
    if not public_key_b64 or not signature_b64:
        return False, "Update manifest release signature is incomplete."

    trusted_keys = trusted_update_public_keys()
    if trusted_keys and public_key_b64 not in trusted_keys:
        return False, "Update manifest release signature key is not trusted."

    try:
        if scheme == UPDATE_SIGNATURE_SCHEME_HYBRID:
            if str(signature.get("pq_scheme") or "") != SIGNING_SCHEME_ML_DSA_65:
                return False, "Update manifest release PQ signature scheme is unsupported."
            pq_public_key_b64 = str(signature.get("pq_public_key_b64") or "").strip()
            pq_signature_b64 = str(signature.get("pq_signature_b64") or "").strip()
            if not pq_public_key_b64 or not pq_signature_b64:
                return False, "Update manifest release hybrid signature is incomplete."
            expected_public_key_address = hybrid_address_from_public_keys_b64(
                ed25519_public_key_b64=public_key_b64,
                pq_public_key_b64=pq_public_key_b64,
            )
        else:
            expected_public_key_address = address_from_public_key_b64(public_key_b64)
    except Exception:
        return False, "Update manifest release signature public key is invalid."
    declared_public_key_address = str(
        signature.get("public_key_address") or ""
    ).strip().lower()
    if (
        declared_public_key_address
        and declared_public_key_address != expected_public_key_address
    ):
        return False, "Update manifest release signature public key address mismatch."

    signing_body = update_manifest_signing_body(manifest)
    if scheme == UPDATE_SIGNATURE_SCHEME_HYBRID:
        if not verify_hybrid_payload_signature(
            ed25519_public_key_b64=public_key_b64,
            ed25519_signature_b64=signature_b64,
            pq_public_key_b64=str(signature.get("pq_public_key_b64") or "").strip(),
            pq_signature_b64=str(signature.get("pq_signature_b64") or "").strip(),
            payload=signing_body,
        ):
            return False, "Update manifest release hybrid signature is invalid."
        return True, None

    if not verify_payload_signature(
        public_key_b64=public_key_b64,
        signature_b64=signature_b64,
        payload=signing_body,
    ):
        return False, "Update manifest release signature is invalid."
    return True, None


def update_manifest_signature_required(value: str | None = None) -> bool:
    raw = str(
        value
        if value is not None
        else os.getenv(REQUIRE_SIGNED_UPDATES_ENV, "")
    ).strip().lower()
    if raw in _TRUTHY or raw in {"strict", "required"}:
        return True
    if raw in _FALSEY:
        return False
    return bool(trusted_update_public_keys())


def trusted_update_public_keys(value: str | None = None) -> set[str]:
    raw_value = str(
        value
        if value is not None
        else os.getenv(UPDATE_TRUSTED_PUBLIC_KEYS_ENV, "")
    )
    keys: set[str] = set()
    for item in re.split(r"[,;\s]+", raw_value):
        normalized = item.strip()
        if normalized:
            keys.add(normalized)
    return keys


def update_manifest_signing_body(manifest: dict[str, Any]) -> dict[str, Any]:
    body = dict(manifest)
    body.pop("signature", None)
    return body


def _can_apply_update(
    local_state: LocalUpdateState,
    *,
    target_channel: str | None = None,
    target_branch: str | None = None,
    target_install_kind: str | None = None,
) -> tuple[bool, str]:
    normalized_target_install_kind: str | None = None
    if target_install_kind:
        try:
            normalized_target_install_kind = normalize_update_install_kind(target_install_kind)
        except UpdateError as exc:
            return False, str(exc)

    if local_state.install_kind == "portable":
        if not portable_install_looks_valid(local_state.repo_root):
            return False, "Local CAI install is not a portable runtime."
        if target_channel == "github":
            return False, "Portable CAI installs can only update from a validator package."
        if normalized_target_install_kind and normalized_target_install_kind != "portable":
            return False, "Validator update package kind does not match the local portable install."
        return True, "ok"

    if local_state.install_kind != "source" or not source_repo_looks_valid(local_state.repo_root):
        return False, "Local CAI install is neither a source repository nor a portable runtime."
    if local_state.git_commit is None:
        return False, "Local CAI install is not a git checkout, so safe auto-apply is disabled."
    if local_state.git_dirty:
        return False, "Local CAI checkout has uncommitted changes, so auto-update is skipped."
    if normalized_target_install_kind and normalized_target_install_kind != "source":
        return False, "Validator update package kind does not match the local source install."
    if target_channel == "github":
        if not local_state.git_branch or local_state.git_branch == "HEAD":
            return False, "Local CAI checkout is in detached HEAD state, so GitHub auto-update is skipped."
        if target_branch and local_state.git_branch != target_branch:
            return (
                False,
                "Local CAI checkout branch does not match the configured GitHub update branch, "
                "so auto-update is skipped.",
            )
    return True, "ok"


def _download_update_archive(
    manifest: dict[str, Any],
    download_dir: Path,
    *,
    timeout_sec: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> Path:
    archive_url = str(manifest.get("archiveUrl") or "").strip()
    if not archive_url:
        raise UpdateError("Update manifest does not include archiveUrl.")
    expected_sha256 = str(manifest.get("archiveSha256") or "").strip().lower()
    if not expected_sha256:
        raise UpdateError("Update manifest does not include archiveSha256.")
    if not _is_sha256_hex(expected_sha256):
        raise UpdateError("Update manifest archiveSha256 is not a valid SHA-256 hex digest.")
    parsed = urlparse(archive_url)
    archive_name = _download_archive_name(manifest, parsed.path)
    archive_path = download_dir / archive_name
    partial_path = archive_path.with_name(f"{archive_path.name}.tmp")
    manifest_size = _manifest_archive_size_bytes(manifest)
    downloaded_bytes = 0
    started_at = time.monotonic()
    last_emit_at = 0.0
    last_emit_progress = -1

    def emit_progress(total_bytes: int | None, *, force: bool = False) -> None:
        nonlocal last_emit_at, last_emit_progress
        if progress_callback is None:
            return
        now = time.monotonic()
        download_percent = (
            min(100.0, max(0.0, (downloaded_bytes / total_bytes) * 100.0))
            if total_bytes and total_bytes > 0
            else None
        )
        progress = _download_phase_progress(downloaded_bytes, total_bytes)
        if (
            not force
            and now - last_emit_at < 0.5
            and int(progress) == int(last_emit_progress)
        ):
            return
        elapsed = max(now - started_at, 0.001)
        progress_callback(
            {
                "downloadedBytes": downloaded_bytes,
                "totalBytes": total_bytes,
                "downloadPercent": download_percent,
                "downloadSpeedBytesPerSec": downloaded_bytes / elapsed,
                "progress": progress,
            }
        )
        last_emit_at = now
        last_emit_progress = int(progress)

    expected_size = _optional_int(manifest.get("archiveSizeBytes"), default=0)
    if expected_size <= 0:
        expected_size = manifest_size or 0
    expected_size_value = expected_size if expected_size > 0 else None

    if archive_path.exists():
        archive_size = archive_path.stat().st_size
        if expected_size_value is not None and archive_size < expected_size_value:
            archive_path.replace(partial_path)
        elif expected_size_value is not None and archive_size != expected_size_value:
            archive_path.unlink(missing_ok=True)
        elif sha256_file(archive_path).lower() == expected_sha256:
            return archive_path
        else:
            archive_path.unlink(missing_ok=True)

    if expected_size_value is not None and parsed.scheme.lower() in {"http", "https"}:
        return _download_update_archive_by_bounded_ranges(
            archive_url=archive_url,
            archive_path=archive_path,
            partial_path=partial_path,
            expected_sha256=expected_sha256,
            expected_size=expected_size_value,
            timeout_sec=timeout_sec,
            progress_callback=progress_callback,
            cancel_requested=cancel_requested,
        )

    last_error: Exception | None = None
    for attempt in range(1, _DOWNLOAD_MAX_ATTEMPTS + 1):
        resume_from = partial_path.stat().st_size if partial_path.exists() else 0
        if expected_size_value is not None and resume_from > expected_size_value:
            partial_path.unlink(missing_ok=True)
            resume_from = 0
        downloaded_bytes = resume_from
        request = Request(archive_url)
        if resume_from > 0:
            request.add_header("Range", f"bytes={resume_from}-")
        try:
            with urlopen(request, timeout=timeout_sec) as response:
                response_status = _response_status_code(response)
                append_to_partial = resume_from > 0 and response_status == 206
                if resume_from > 0 and not append_to_partial:
                    partial_path.unlink(missing_ok=True)
                    downloaded_bytes = 0
                mode = "ab" if append_to_partial else "wb"
                with partial_path.open(mode) as handle:
                    response_size = _response_content_length_bytes(response)
                    total_bytes = expected_size_value or response_size
                    emit_progress(total_bytes, force=True)
                    while True:
                        if cancel_requested is not None and cancel_requested():
                            raise UpdateCancelled(
                                "CAI portable update was cancelled during download."
                            )
                        chunk = response.read(_DOWNLOAD_CHUNK_SIZE_BYTES)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded_bytes += len(chunk)
                        emit_progress(total_bytes)
                    emit_progress(total_bytes, force=True)

            actual_size = partial_path.stat().st_size
            if expected_size_value is not None and actual_size != expected_size_value:
                last_error = UpdateError(
                    f"Update archive size mismatch: expected {expected_size_value}, got {actual_size}."
                )
                if attempt < _DOWNLOAD_MAX_ATTEMPTS:
                    time.sleep(min(2.0 * attempt, 8.0))
                    continue
                raise last_error

            partial_path.replace(archive_path)
            actual_sha256 = sha256_file(archive_path).lower()
            if actual_sha256 == expected_sha256:
                return archive_path

            archive_path.unlink(missing_ok=True)
            partial_path.unlink(missing_ok=True)
            last_error = UpdateError(
                f"Update archive checksum mismatch: expected {expected_sha256}, got {actual_sha256}."
            )
            if attempt < _DOWNLOAD_MAX_ATTEMPTS:
                time.sleep(min(2.0 * attempt, 8.0))
                continue
            raise last_error
        except UpdateCancelled:
            partial_path.unlink(missing_ok=True)
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc if isinstance(exc, Exception) else UpdateError(str(exc))
            if attempt >= _DOWNLOAD_MAX_ATTEMPTS:
                break
            time.sleep(min(2.0 * attempt, 8.0))

    if last_error is not None:
        try:
            if expected_size_value is not None and partial_path.exists():
                partial_size = partial_path.stat().st_size
                if partial_size <= 0 or partial_size > expected_size_value:
                    partial_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise last_error
    raise UpdateError("Update archive download failed.")


def _download_update_archive_by_bounded_ranges(
    *,
    archive_url: str,
    archive_path: Path,
    partial_path: Path,
    expected_sha256: str,
    expected_size: int,
    timeout_sec: int,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    cancel_requested: Callable[[], bool] | None,
) -> Path:
    started_at = time.monotonic()
    downloaded_bytes = partial_path.stat().st_size if partial_path.exists() else 0
    if downloaded_bytes < 0 or downloaded_bytes > expected_size:
        partial_path.unlink(missing_ok=True)
        downloaded_bytes = 0

    range_chunk_size = _positive_int_from_env(
        "CAI_UPDATE_DOWNLOAD_RANGE_CHUNK_BYTES",
        _DOWNLOAD_RANGE_CHUNK_SIZE_BYTES,
    )
    range_chunk_size = max(1024 * 1024, range_chunk_size)
    range_attempts = _positive_int_from_env(
        "CAI_UPDATE_DOWNLOAD_RANGE_CHUNK_ATTEMPTS",
        _DOWNLOAD_RANGE_CHUNK_MAX_ATTEMPTS,
    )
    last_emit_at = 0.0
    last_emit_progress = -1

    def emit_progress(*, force: bool = False) -> None:
        nonlocal last_emit_at, last_emit_progress
        if progress_callback is None:
            return
        now = time.monotonic()
        download_percent = min(
            100.0,
            max(0.0, (downloaded_bytes / expected_size) * 100.0),
        )
        progress = _download_phase_progress(downloaded_bytes, expected_size)
        if (
            not force
            and now - last_emit_at < 0.5
            and int(progress) == int(last_emit_progress)
        ):
            return
        elapsed = max(now - started_at, 0.001)
        progress_callback(
            {
                "downloadedBytes": downloaded_bytes,
                "totalBytes": expected_size,
                "downloadPercent": download_percent,
                "downloadSpeedBytesPerSec": downloaded_bytes / elapsed,
                "progress": progress,
            }
        )
        last_emit_at = now
        last_emit_progress = int(progress)

    emit_progress(force=True)
    while downloaded_bytes < expected_size:
        if cancel_requested is not None and cancel_requested():
            partial_path.unlink(missing_ok=True)
            raise UpdateCancelled("CAI portable update was cancelled during download.")
        chunk_start = downloaded_bytes
        chunk_end = min(expected_size - 1, chunk_start + range_chunk_size - 1)
        expected_chunk_size = chunk_end - chunk_start + 1
        last_error: Exception | None = None

        for attempt in range(1, range_attempts + 1):
            request = Request(archive_url)
            request.add_header("Range", f"bytes={chunk_start}-{chunk_end}")
            try:
                with urlopen(request, timeout=timeout_sec) as response:
                    response_status = _response_status_code(response)
                    if response_status != 206:
                        raise UpdateError(
                            "Update archive server did not honor a bounded Range "
                            f"request; status={response_status}."
                        )
                    remaining = expected_chunk_size
                    with partial_path.open("ab") as handle:
                        while remaining > 0:
                            if cancel_requested is not None and cancel_requested():
                                raise UpdateCancelled(
                                    "CAI portable update was cancelled during download."
                                )
                            payload = response.read(
                                min(_DOWNLOAD_CHUNK_SIZE_BYTES, remaining)
                            )
                            if not payload:
                                raise UpdateError(
                                    "Update archive range response ended before the "
                                    "requested chunk was complete."
                                )
                            handle.write(payload)
                            downloaded_bytes += len(payload)
                            remaining -= len(payload)
                            emit_progress()
                    emit_progress(force=True)
                break
            except UpdateCancelled:
                partial_path.unlink(missing_ok=True)
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc if isinstance(exc, Exception) else UpdateError(str(exc))
                downloaded_bytes = chunk_start
                try:
                    with partial_path.open("ab") as handle:
                        handle.truncate(chunk_start)
                except OSError:
                    partial_path.unlink(missing_ok=True)
                    downloaded_bytes = 0
                if attempt >= range_attempts:
                    raise last_error
                time.sleep(min(2.0 * attempt, 8.0))

    partial_path.replace(archive_path)
    actual_sha256 = sha256_file(archive_path).lower()
    if actual_sha256 == expected_sha256:
        emit_progress(force=True)
        return archive_path

    archive_path.unlink(missing_ok=True)
    partial_path.unlink(missing_ok=True)
    raise UpdateError(
        f"Update archive checksum mismatch: expected {expected_sha256}, got {actual_sha256}."
    )


def _download_archive_name(manifest: dict[str, Any], url_path: str | None) -> str:
    archive_name = Path(url_path or "cai-update.zip").name or "cai-update.zip"
    archive_path = Path(archive_name)
    if archive_path.suffix.lower() == ".zip":
        return archive_name

    build_id = str(manifest.get("buildId") or "").strip()
    archive_sha256 = str(manifest.get("archiveSha256") or "").strip().lower()
    if build_id:
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", build_id).strip(".-")
    elif _is_sha256_hex(archive_sha256):
        safe_label = archive_sha256[:12]
    else:
        safe_label = archive_path.name or "cai-update"
    if not safe_label:
        safe_label = "cai-update"
    return f"{safe_label}.zip"


def _manifest_archive_size_bytes(manifest: dict[str, Any]) -> int | None:
    raw_size = manifest.get("archiveSizeBytes")
    if raw_size in (None, ""):
        return None
    expected_size = _optional_int(raw_size, default=0)
    return expected_size if expected_size > 0 else None


def _response_content_length_bytes(response: Any) -> int | None:
    try:
        raw_length = response.headers.get("Content-Length")
    except Exception:
        raw_length = None
    if raw_length in (None, ""):
        try:
            raw_length = response.getheader("Content-Length")
        except Exception:
            raw_length = None
    if raw_length in (None, ""):
        return None
    try:
        length = int(str(raw_length).strip())
    except (TypeError, ValueError):
        return None
    return length if length > 0 else None


def _response_status_code(response: Any) -> int | None:
    try:
        status = getattr(response, "status")
    except Exception:
        status = None
    if isinstance(status, int):
        return status
    try:
        code = response.getcode()
    except Exception:
        return None
    return code if isinstance(code, int) else None


def _download_phase_progress(downloaded_bytes: int, total_bytes: int | None) -> int:
    if not total_bytes or total_bytes <= 0:
        return _DOWNLOAD_PROGRESS_START
    ratio = min(1.0, max(0.0, downloaded_bytes / total_bytes))
    span = _DOWNLOAD_PROGRESS_END - _DOWNLOAD_PROGRESS_START
    return int(round(_DOWNLOAD_PROGRESS_START + (ratio * span)))


def _apply_github_update(
    local_state: LocalUpdateState,
    *,
    source: GitHubUpdateSource,
    remote_manifest: dict[str, Any],
) -> dict[str, Any]:
    remote_commit = str(remote_manifest.get("gitCommit") or "").strip() or None
    if not remote_commit:
        raise UpdateError("GitHub update manifest did not include a remote commit.")

    _run_git(
        local_state.repo_root,
        "fetch",
        "--depth",
        "1",
        "--no-tags",
        source.repo_url,
        source.branch,
    )
    _run_git(local_state.repo_root, "merge", "--ff-only", "FETCH_HEAD")

    updated_state = collect_local_update_state(local_state.repo_root)
    build_result = _rebuild_dashboard_bundle(local_state.repo_root)
    message = "CAI source checkout updated from GitHub."
    if build_result["status"] == "rebuilt":
        message += " Dashboard build refreshed."
    elif build_result["status"] in {"failed", "skipped"}:
        message += f" {build_result['message']}"

    return {
        "checked": True,
        "updated": True,
        "status": "updated",
        "message": message,
        "channel": "github",
        "provider": "github",
        "sourceUrl": f"{source.repo_url.removesuffix('.git')}/tree/{source.branch}",
        "repository": source.repository,
        "targetBranch": source.branch,
        "baseUrl": None,
        "localVersion": __version__,
        "localGitCommit": updated_state.git_commit,
        "localGitBranch": updated_state.git_branch,
        "localGitDirty": updated_state.git_dirty,
        "remoteGitCommit": remote_commit,
        "remoteGitBranch": source.branch,
        "remoteVersion": remote_manifest.get("version"),
        "updateAvailable": False,
        "canApply": True,
        "applyReason": "ok",
        "dashboardBuildStatus": build_result["status"],
        "dashboardBuildMessage": build_result["message"],
    }


def _rebuild_dashboard_bundle(repo_root: Path) -> dict[str, str]:
    dashboard_root = repo_root / "cai" / "dashboard"
    if not (dashboard_root / "package.json").is_file():
        return {
            "status": "skipped",
            "message": "Dashboard source was not found, so no frontend rebuild was needed.",
        }

    npm_executable = shutil.which("npm") or shutil.which("npm.cmd")
    if not npm_executable:
        return {
            "status": "skipped",
            "message": "npm is not installed, so the dashboard build was not refreshed.",
        }

    try:
        subprocess.run(
            [npm_executable, "run", "build"],
            cwd=dashboard_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "failed",
            "message": f"Dashboard rebuild timed out after {int(exc.timeout)} seconds.",
        }
    except subprocess.CalledProcessError as exc:
        error_output = (exc.stderr or exc.stdout or "").strip()
        if error_output:
            error_output = error_output.splitlines()[-1].strip()
        return {
            "status": "failed",
            "message": (
                "Dashboard rebuild failed."
                if not error_output
                else f"Dashboard rebuild failed: {error_output}"
            ),
        }

    return {
        "status": "rebuilt",
        "message": "Dashboard build refreshed from local source.",
    }


def _read_update_status(repo_root: Path) -> dict[str, Any]:
    status_path = update_status_path(repo_root)
    if not status_path.is_file():
        return {}
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_update_activity(repo_root: Path) -> dict[str, Any]:
    activity_path = update_activity_path(repo_root)
    if not activity_path.is_file():
        return {}
    try:
        payload = json.loads(activity_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_iso_seconds(value: object) -> float | None:
    if not value:
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.timestamp()
    except Exception:
        return None


def _positive_int_from_env(env_name: str, default: int) -> int:
    try:
        value = int(str(os.getenv(env_name) or "").strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def auto_update_check_timeout_seconds() -> int:
    return _positive_int_from_env(
        AUTO_UPDATE_CHECK_TIMEOUT_SECONDS_ENV,
        _DEFAULT_AUTO_UPDATE_CHECK_TIMEOUT_SECONDS,
    )


def auto_update_idle_seconds() -> int:
    return _positive_int_from_env(
        AUTO_UPDATE_IDLE_SECONDS_ENV,
        _DEFAULT_AUTO_UPDATE_IDLE_SECONDS,
    )


def auto_update_idle_timeout_seconds() -> int:
    return _positive_int_from_env(
        AUTO_UPDATE_IDLE_TIMEOUT_SECONDS_ENV,
        _DEFAULT_AUTO_UPDATE_IDLE_TIMEOUT_SECONDS,
    )


def record_portable_update_activity(
    repo_root: Path | None = None,
    *,
    source: str = "dashboard",
    active_request_count: int | None = None,
    user_active: bool | None = None,
    last_user_activity_at: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = resolve_repo_root(repo_root)
    activity_path = update_activity_path(root)
    activity_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_update_activity(root)
    now = datetime.now(tz=UTC).isoformat()

    payload: dict[str, Any] = {
        **existing,
        "schemaVersion": 1,
        "source": str(source or "unknown"),
        "updatedAt": now,
    }
    if active_request_count is not None:
        payload["activeRequestCount"] = max(0, int(active_request_count))
        payload["activeRequestUpdatedAt"] = now
    if last_user_activity_at:
        payload["lastUserActivityAt"] = str(last_user_activity_at)
    elif user_active:
        payload["lastUserActivityAt"] = now
    if user_active is not None:
        payload["userActive"] = bool(user_active)
    if metadata:
        payload["metadata"] = dict(metadata)

    activity_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def portable_update_activity_snapshot(
    repo_root: Path | None = None,
    *,
    idle_seconds: int | None = None,
) -> dict[str, Any]:
    root = resolve_repo_root(repo_root)
    idle_required = int(idle_seconds or auto_update_idle_seconds())
    activity = _read_update_activity(root)
    now_seconds = time.time()
    updated_seconds = _parse_iso_seconds(activity.get("updatedAt"))
    request_updated_seconds = _parse_iso_seconds(activity.get("activeRequestUpdatedAt"))
    active_request_count = int(activity.get("activeRequestCount") or 0)
    if (
        active_request_count > 0
        and request_updated_seconds is not None
        and now_seconds - request_updated_seconds > _STALE_ACTIVE_REQUEST_SECONDS
    ):
        active_request_count = 0

    last_user_activity_seconds = _parse_iso_seconds(activity.get("lastUserActivityAt"))
    if last_user_activity_seconds is None:
        idle_for_seconds = None
        user_idle = True
    else:
        idle_for_seconds = max(0.0, now_seconds - last_user_activity_seconds)
        user_idle = idle_for_seconds >= idle_required

    idle = active_request_count <= 0 and user_idle
    if active_request_count > 0:
        reason = "active_request"
    elif not user_idle:
        reason = "recent_user_activity"
    else:
        reason = "idle"

    return {
        "idle": idle,
        "reason": reason,
        "idleRequiredSeconds": idle_required,
        "idleForSeconds": idle_for_seconds,
        "activeRequestCount": active_request_count,
        "lastUserActivityAt": activity.get("lastUserActivityAt"),
        "activeRequestUpdatedAt": activity.get("activeRequestUpdatedAt"),
        "activityUpdatedAt": activity.get("updatedAt"),
        "activityStale": (
            updated_seconds is not None
            and now_seconds - updated_seconds > _STALE_ACTIVE_REQUEST_SECONDS
        ),
    }


def _portable_update_cancel_requested(repo_root: Path) -> bool:
    return portable_update_cancel_path(repo_root).is_file()


def _clear_portable_update_cancel_marker(repo_root: Path) -> None:
    cancel_path = portable_update_cancel_path(repo_root)
    try:
        if cancel_path.is_file():
            cancel_path.unlink()
    except OSError:
        pass


def _portable_update_cancelled_status(
    repo_root: Path,
    base: dict[str, Any] | None = None,
    *,
    message: str,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    root = resolve_repo_root(repo_root)
    payload = dict(base or {})
    payload.update(
        {
            "checked": bool(payload.get("checked", True)),
            "updated": False,
            "status": "cancelled",
            "phase": "cancelled",
            "progress": 0,
            "message": message,
            "installKind": payload.get("installKind") or "portable",
            "canCancel": False,
            "cancelRequested": True,
            "restartRequired": False,
            "restartScheduled": False,
            "portableUpdateCancelPath": str(portable_update_cancel_path(root)),
        }
    )
    if archive_path is not None:
        payload["archivePath"] = str(archive_path.expanduser().resolve())
    return payload


def _write_update_status(repo_root: Path, payload: dict[str, Any]) -> None:
    status_path = update_status_path(repo_root)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_update_status(repo_root)
    persisted = {
        **existing,
        **payload,
        "checkedAt": datetime.now(tz=UTC).isoformat(),
    }
    if bool(persisted.get("updated")):
        persisted["lastUpdatedAt"] = persisted["checkedAt"]
    elif existing.get("lastUpdatedAt") and "lastUpdatedAt" not in payload:
        persisted["lastUpdatedAt"] = existing["lastUpdatedAt"]

    status_path.write_text(
        json.dumps(persisted, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _prepare_update_stage_dir(repo_root: Path) -> Path:
    root = resolve_repo_root(repo_root)
    stage_dir = update_stage_dir(root)
    _assert_repo_internal_path(root, stage_dir, label="update stage")
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir


def _clear_update_stage_dir(repo_root: Path) -> None:
    root = resolve_repo_root(repo_root)
    stage_dir = update_stage_dir(root)
    _assert_repo_internal_path(root, stage_dir, label="update stage")
    if stage_dir.exists():
        shutil.rmtree(stage_dir)


def _prepare_update_rollback_backup_dir(repo_root: Path) -> Path:
    root = resolve_repo_root(repo_root)
    backup_dir = update_rollback_backup_dir(root)
    _assert_repo_internal_path(root, backup_dir, label="rollback backup")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir


def _clear_update_rollback_backup_dir(repo_root: Path) -> None:
    root = resolve_repo_root(repo_root)
    backup_dir = update_rollback_backup_dir(root)
    _assert_repo_internal_path(root, backup_dir, label="rollback backup")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def _write_update_rollback_marker(repo_root: Path, payload: dict[str, Any]) -> None:
    marker_path = update_rollback_marker_path(repo_root)
    _assert_repo_internal_path(resolve_repo_root(repo_root), marker_path, label="rollback marker")
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _update_rollback_marker(repo_root: Path, updates: dict[str, Any]) -> None:
    marker_path = update_rollback_marker_path(repo_root)
    try:
        current = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        current = {}
    if not isinstance(current, dict):
        current = {}
    _write_update_rollback_marker(repo_root, {**current, **updates})


def _backup_update_target(
    repo_root: Path,
    target_path: Path,
    backup_root: Path,
    backed_up_files: list[str],
) -> None:
    relative_path = target_path.relative_to(repo_root).as_posix()
    if relative_path in backed_up_files:
        return
    backup_path = backup_root / Path(relative_path)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.is_dir():
        shutil.copytree(target_path, backup_path)
    else:
        shutil.copy2(target_path, backup_path)
    backed_up_files.append(relative_path)


def _restore_update_backup(
    repo_root: Path,
    backup_root: Path,
    backed_up_files: list[str],
    created_files: list[str],
) -> None:
    for relative_path in reversed(created_files):
        target_path = repo_root / Path(relative_path)
        if not target_path.exists():
            continue
        if target_path.is_dir():
            shutil.rmtree(target_path)
        else:
            target_path.unlink()

    for relative_path in reversed(backed_up_files):
        backup_path = backup_root / Path(relative_path)
        target_path = repo_root / Path(relative_path)
        if not backup_path.exists():
            continue
        if target_path.exists():
            if target_path.is_dir():
                shutil.rmtree(target_path)
            else:
                target_path.unlink()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if backup_path.is_dir():
            shutil.copytree(backup_path, target_path)
        else:
            shutil.copy2(backup_path, target_path)


def _fetch_json_payload(url: str, *, timeout_sec: int) -> Any:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "cai-compute-chain-updater",
            **_github_auth_header(),
        },
    )
    with urlopen(request, timeout=timeout_sec) as response:
        return json.loads(response.read().decode("utf-8"))


def _github_auth_header() -> dict[str, str]:
    token = str(os.getenv(UPDATE_GITHUB_TOKEN_ENV) or "").strip()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _normalize_github_repository(value: str) -> str | None:
    stripped = str(value or "").strip()
    if not stripped:
        return None
    if "/" in stripped and "://" not in stripped and not stripped.startswith("git@"):
        owner, repo = stripped.split("/", 1)
        repo = repo.removesuffix(".git").strip("/")
        owner = owner.strip("/")
        if owner and repo:
            return f"{owner}/{repo}"
        return None
    match = _GITHUB_REPOSITORY_RE.match(stripped)
    if not match:
        return None
    return f"{match.group(1)}/{match.group(2)}"


def _git_remote_origin_url(repo_root: Path) -> str | None:
    if not (repo_root / ".git").exists():
        return None
    try:
        origin_url = _run_git_text(repo_root, "config", "--get", "remote.origin.url")
    except Exception:
        return None
    return origin_url or None


def _generated_roots(repo_root: Path) -> tuple[str, ...]:
    result: list[str] = []
    for relative_path in GENERATED_UPDATE_ROOTS:
        candidate = repo_root / Path(relative_path)
        if candidate.is_dir():
            result.append(relative_path)
    return tuple(result)


def _collect_package_files(repo_root: Path, tracked_files: tuple[str, ...]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for relative_path in tracked_files:
        candidate = repo_root / Path(relative_path)
        if not candidate.is_file():
            continue
        normalized_path = _normalize_update_package_path(relative_path)
        _raise_if_forbidden_package_path(normalized_path)
        if normalized_path in seen_paths:
            continue
        entries.append(
            {
                "path": normalized_path,
                "mode": candidate.stat().st_mode & 0o777,
                "tracked": True,
            }
        )
        seen_paths.add(normalized_path)

    for relative_root in _generated_roots(repo_root):
        root_path = repo_root / Path(relative_root)
        for candidate in sorted(root_path.rglob("*")):
            if not candidate.is_file():
                continue
            normalized_path = candidate.relative_to(repo_root).as_posix()
            _raise_if_forbidden_package_path(normalized_path)
            if normalized_path in seen_paths:
                continue
            entries.append(
                {
                    "path": normalized_path,
                    "mode": candidate.stat().st_mode & 0o777,
                    "tracked": False,
                }
            )
            seen_paths.add(normalized_path)

    entries.sort(key=lambda item: str(item["path"]))
    return entries


def _raise_if_forbidden_package_path(relative_path: str) -> None:
    normalized_path = _normalize_update_package_path(relative_path)
    for pattern in FORBIDDEN_PACKAGE_PATH_PATTERNS:
        if pattern.search(normalized_path):
            raise UpdateError(
                f"Refusing to package sensitive runtime path: {normalized_path}"
            )


def _is_forbidden_package_path(relative_path: str) -> bool:
    try:
        normalized_path = _normalize_update_package_path(relative_path)
    except UpdateError:
        return True
    return any(pattern.search(normalized_path) for pattern in FORBIDDEN_PACKAGE_PATH_PATTERNS)


def _validate_update_protocol_range(payload: dict[str, Any], *, payload_name: str) -> None:
    raw_manifest_version = payload.get("manifestVersion")
    if raw_manifest_version in (None, ""):
        raw_manifest_version = payload.get("packageFormatVersion")
    manifest_version = _optional_int(
        raw_manifest_version,
        default=UPDATE_MANIFEST_VERSION,
    )
    if manifest_version < 1 or manifest_version > UPDATE_MANIFEST_VERSION:
        raise UpdateError(
            f"{payload_name} version {manifest_version} is not supported by this updater."
        )

    protocol_version = _optional_int(payload.get("protocolVersion"), default=UPDATE_PROTOCOL_VERSION)
    min_protocol = _optional_int(
        payload.get("minCompatibleProtocolVersion"),
        default=UPDATE_MIN_COMPATIBLE_PROTOCOL_VERSION,
    )
    max_protocol = _optional_int(
        payload.get("maxCompatibleProtocolVersion"),
        default=protocol_version,
    )
    if min_protocol > max_protocol:
        raise UpdateError(f"{payload_name} protocol compatibility range is invalid.")
    if not (min_protocol <= UPDATE_PROTOCOL_VERSION <= max_protocol):
        raise UpdateError(
            f"{payload_name} requires update protocol {min_protocol}-{max_protocol}, "
            f"but this updater supports {UPDATE_PROTOCOL_VERSION}."
        )


def _optional_int(value: Any, *, default: int) -> int:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise UpdateError(f"Invalid integer value in update metadata: {value!r}.") from exc


def _is_sha256_hex(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(value or "").strip()))


def _normalize_update_package_path(relative_path: str) -> str:
    raw_path = str(relative_path or "").replace("\\", "/").strip()
    if not raw_path:
        raise UpdateError("Update package path must not be empty.")
    if raw_path.startswith("/") or re.match(r"^[A-Za-z]:", raw_path):
        raise UpdateError(f"Refusing absolute update package path: {raw_path}")

    parts = [part for part in raw_path.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise UpdateError(f"Refusing unsafe update package path: {raw_path}")
    normalized = "/".join(parts)
    if not normalized:
        raise UpdateError("Update package path must not be empty.")
    return normalized


def _validate_update_archive_members(bundle: zipfile.ZipFile) -> None:
    for member in bundle.infolist():
        raw_name = str(member.filename or "").replace("\\", "/")
        if not raw_name or raw_name.endswith("/"):
            continue
        normalized_name = _normalize_update_package_path(raw_name)
        if normalized_name == PACKAGE_METADATA_PATH:
            continue
        _raise_if_forbidden_package_path(normalized_name)


def _resolve_portable_update_payload_root(stage_root: Path) -> Path:
    candidates = [stage_root]
    candidates.extend(
        child
        for child in sorted(stage_root.iterdir(), key=lambda item: item.name.lower())
        if child.is_dir()
    )
    for candidate in candidates:
        if portable_install_looks_valid(candidate):
            return candidate
    raise UpdateError("Portable update archive does not contain a CAI portable runtime.")


def _assert_repo_internal_path(repo_root: Path, candidate: Path, *, label: str) -> None:
    root = repo_root.expanduser().resolve()
    resolved = candidate.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise UpdateError(f"Refusing {label} outside CAI repository: {resolved}") from exc


def _run_git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or str(exc)
        raise UpdateError(f"git {' '.join(args)} failed: {detail}") from exc


def _run_git_text(repo_root: Path, *args: str) -> str:
    completed = _run_git(repo_root, *args)
    return completed.stdout.strip()


def _run_git_tracked_files(repo_root: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z"],
        capture_output=True,
        check=True,
    )
    paths = [
        entry.decode("utf-8")
        for entry in completed.stdout.split(b"\0")
        if entry
    ]
    return tuple(sorted(path.replace("\\", "/") for path in paths))


def _write_zip_file(bundle: zipfile.ZipFile, source_path: Path, archive_name: str) -> None:
    info = zipfile.ZipInfo(archive_name.replace("\\", "/"))
    stat_result = source_path.stat()
    info.date_time = datetime.fromtimestamp(stat_result.st_mtime).timetuple()[:6]
    info.external_attr = (stat_result.st_mode & 0xFFFF) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    with source_path.open("rb") as handle:
        bundle.writestr(info, handle.read())


