#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cai_compute_chain import __version__  # noqa: E402
from cai_compute_chain.wallet_signing import (  # noqa: E402
    SIGNING_SCHEME_ML_DSA_65,
    address_from_public_key_b64,
    decode_bytes,
    encode_bytes,
    hybrid_address_from_public_keys_b64,
    mldsa65_keypair_b64_from_seed,
    public_key_b64_from_seed,
    sign_payload_mldsa65_b64,
    sign_payload_b64,
    verify_hybrid_payload_signature,
    verify_payload_signature,
)


SCHEMA_VERSION = 1
METADATA_KIND = "cai-release-artifact-metadata"
SIGNATURE_SCHEME_ED25519 = "cai-release-metadata-ed25519-v1"
SIGNATURE_SCHEME_HYBRID = "cai-release-metadata-hybrid-ed25519-ml-dsa-65-v1"
SIGNATURE_SCHEME = SIGNATURE_SCHEME_HYBRID
RELEASE_SIGNING_SEED_ENV = "CAI_RELEASE_SIGNING_SEED_B64"
RELEASE_SIGNING_PUBLIC_KEY_ENV = "CAI_RELEASE_SIGNING_PUBLIC_KEY_B64"
RELEASE_TRUSTED_PUBLIC_KEYS_ENV = "CAI_RELEASE_TRUSTED_PUBLIC_KEYS_B64"
REQUIRE_SIGNED_RELEASES_ENV = "CAI_REQUIRE_SIGNED_RELEASES"
UPDATE_SIGNING_SEED_ENV = "CAI_UPDATE_SIGNING_SEED_B64"
UPDATE_SIGNING_PUBLIC_KEY_ENV = "CAI_UPDATE_SIGNING_PUBLIC_KEY_B64"
_TRUTHY = {"1", "true", "yes", "on", "strict", "required"}
_FALSEY = {"0", "false", "no", "off"}


def build_release_metadata(
    repo_root: Path,
    *,
    artifacts: list[Path],
    version: str,
    build_id: str | None,
    sign: bool,
) -> dict[str, Any]:
    repo_root = repo_root.expanduser().resolve()
    commit = _run_git_text(repo_root, "rev-parse", "HEAD")
    branch = _run_git_text(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    dirty = bool(_run_git_text(repo_root, "status", "--porcelain", "--untracked-files=no"))
    metadata = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": METADATA_KIND,
        "version": version,
        "gitCommit": commit,
        "gitBranch": branch,
        "gitDirty": dirty,
        "buildId": build_id or _default_build_id(version=version, commit=commit),
        "generatedAt": datetime.now(tz=UTC).isoformat(),
        "artifacts": [
            _artifact_metadata(repo_root, artifact)
            for artifact in artifacts
        ],
    }
    if sign:
        return maybe_sign_release_metadata(metadata)
    return metadata


def maybe_sign_release_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    signing_seed_b64 = _env_first(RELEASE_SIGNING_SEED_ENV, UPDATE_SIGNING_SEED_ENV)
    if not signing_seed_b64:
        return metadata
    public_key_b64 = _env_first(
        RELEASE_SIGNING_PUBLIC_KEY_ENV,
        UPDATE_SIGNING_PUBLIC_KEY_ENV,
    )
    return sign_release_metadata(
        metadata,
        signing_seed_b64=signing_seed_b64,
        public_key_b64=public_key_b64 or None,
    )


def sign_release_metadata(
    metadata: dict[str, Any],
    *,
    signing_seed_b64: str,
    public_key_b64: str | None = None,
    signed_at: str | None = None,
) -> dict[str, Any]:
    signing_seed = decode_bytes(str(signing_seed_b64 or "").strip())
    normalized_public_key = str(public_key_b64 or "").strip() or public_key_b64_from_seed(
        signing_seed
    )
    signed_metadata = dict(metadata)
    signing_body = release_metadata_signing_body(signed_metadata)
    signature_b64 = sign_payload_b64(signing_seed, signing_body)
    pq_public_key_b64, pq_private_key_b64 = mldsa65_keypair_b64_from_seed(
        _derive_release_metadata_pq_seed(signing_seed)
    )
    signed_metadata["signature"] = {
        "scheme": SIGNATURE_SCHEME,
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
    return signed_metadata


def verify_release_metadata_signature(
    metadata: dict[str, Any],
    *,
    require_signature: bool | None = None,
) -> tuple[bool, str | None]:
    if not isinstance(metadata, dict):
        return False, "Release metadata must be a JSON object."
    if metadata.get("kind") != METADATA_KIND:
        return False, "Release metadata kind is invalid."
    if metadata.get("schemaVersion") != SCHEMA_VERSION:
        return False, "Release metadata schema version is unsupported."

    signature_required = (
        release_signature_required()
        if require_signature is None
        else bool(require_signature)
    )
    signature = metadata.get("signature")
    if not isinstance(signature, dict):
        if signature_required:
            return False, "Release metadata signature is missing."
        return True, None

    scheme = str(signature.get("scheme") or "").strip()
    if scheme not in {SIGNATURE_SCHEME_ED25519, SIGNATURE_SCHEME_HYBRID}:
        return False, f"Release metadata has unsupported signature scheme '{scheme}'."

    public_key_b64 = str(signature.get("public_key_b64") or "").strip()
    signature_b64 = str(signature.get("signature_b64") or "").strip()
    if not public_key_b64 or not signature_b64:
        return False, "Release metadata signature is incomplete."

    trusted_keys = trusted_release_public_keys()
    if trusted_keys and public_key_b64 not in trusted_keys:
        return False, "Release metadata signature key is not trusted."

    declared_public_key_address = str(
        signature.get("public_key_address") or ""
    ).strip().lower()
    try:
        if scheme == SIGNATURE_SCHEME_HYBRID:
            if str(signature.get("pq_scheme") or "") != SIGNING_SCHEME_ML_DSA_65:
                return False, "Release metadata PQ signature scheme is unsupported."
            pq_public_key_b64 = str(signature.get("pq_public_key_b64") or "").strip()
            pq_signature_b64 = str(signature.get("pq_signature_b64") or "").strip()
            if not pq_public_key_b64 or not pq_signature_b64:
                return False, "Release metadata hybrid signature is incomplete."
            expected_public_key_address = hybrid_address_from_public_keys_b64(
                ed25519_public_key_b64=public_key_b64,
                pq_public_key_b64=pq_public_key_b64,
            )
        else:
            expected_public_key_address = address_from_public_key_b64(public_key_b64)
    except Exception:
        return False, "Release metadata signature public key is invalid."
    if (
        declared_public_key_address
        and declared_public_key_address != expected_public_key_address
    ):
        return False, "Release metadata signature public key address mismatch."

    signing_body = release_metadata_signing_body(metadata)
    if scheme == SIGNATURE_SCHEME_HYBRID:
        if not verify_hybrid_payload_signature(
            ed25519_public_key_b64=public_key_b64,
            ed25519_signature_b64=signature_b64,
            pq_public_key_b64=str(signature.get("pq_public_key_b64") or "").strip(),
            pq_signature_b64=str(signature.get("pq_signature_b64") or "").strip(),
            payload=signing_body,
        ):
            return False, "Release metadata hybrid signature is invalid."
        return True, None

    if not verify_payload_signature(
        public_key_b64=public_key_b64,
        signature_b64=signature_b64,
        payload=signing_body,
    ):
        return False, "Release metadata signature is invalid."
    return True, None


def release_metadata_signing_body(metadata: dict[str, Any]) -> dict[str, Any]:
    body = dict(metadata)
    body.pop("signature", None)
    return body


def _derive_release_metadata_pq_seed(signing_seed: bytes) -> bytes:
    return hashlib.sha256(
        b"cai-release-metadata-ml-dsa-65-v1:" + bytes(signing_seed)
    ).digest()


def release_signature_required(value: str | None = None) -> bool:
    raw = str(
        value
        if value is not None
        else os.getenv(REQUIRE_SIGNED_RELEASES_ENV, "")
    ).strip().lower()
    if raw in _TRUTHY:
        return True
    if raw in _FALSEY:
        return False
    return bool(trusted_release_public_keys())


def trusted_release_public_keys(value: str | None = None) -> set[str]:
    raw_value = str(
        value
        if value is not None
        else os.getenv(RELEASE_TRUSTED_PUBLIC_KEYS_ENV, "")
    )
    keys: set[str] = set()
    for raw_key in raw_value.replace(";", ",").replace("\n", ",").split(","):
        key = raw_key.strip()
        if key:
            keys.add(key)
    return keys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate or verify CAI release artifact metadata."
    )
    parser.add_argument("--repo-root", default=".", help="CAI repository root.")
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Release artifact path. Can be used multiple times.",
    )
    parser.add_argument("--output", default=".dist/release-metadata.json")
    parser.add_argument("--version", default=__version__, help="Release version label.")
    parser.add_argument("--build-id", default=None, help="Optional build id.")
    parser.add_argument("--no-sign", action="store_true", help="Do not sign metadata.")
    parser.add_argument(
        "--verify",
        default=None,
        help="Verify an existing release metadata JSON instead of generating one.",
    )
    parser.add_argument(
        "--require-signature",
        action="store_true",
        help="Fail verification if the metadata is unsigned.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    if args.verify:
        metadata_path = Path(args.verify)
        if not metadata_path.is_absolute():
            metadata_path = repo_root / metadata_path
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        ok, error = verify_release_metadata_signature(
            metadata,
            require_signature=True if args.require_signature else None,
        )
        if not ok:
            print(error or "Release metadata signature is invalid.", file=sys.stderr)
            return 1
        print(f"Release metadata is valid: {metadata_path}")
        return 0

    if not args.artifact:
        parser.error("at least one --artifact is required unless --verify is used")

    artifacts = [
        _resolve_artifact(repo_root, Path(raw_artifact))
        for raw_artifact in args.artifact
    ]
    metadata = build_release_metadata(
        repo_root,
        artifacts=artifacts,
        version=args.version,
        build_id=args.build_id,
        sign=not args.no_sign,
    )
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Release metadata written: {output_path}")
    return 0


def _artifact_metadata(repo_root: Path, artifact: Path) -> dict[str, Any]:
    artifact = artifact.expanduser().resolve()
    if not artifact.exists() or not artifact.is_file():
        raise FileNotFoundError(artifact)
    return {
        "name": artifact.name,
        "path": _display_path(repo_root, artifact),
        "sizeBytes": artifact.stat().st_size,
        "sha256": _sha256_file(artifact),
    }


def _resolve_artifact(repo_root: Path, artifact: Path) -> Path:
    if artifact.is_absolute():
        return artifact.expanduser().resolve()
    return (repo_root / artifact).expanduser().resolve()


def _display_path(repo_root: Path, artifact: Path) -> str:
    try:
        return artifact.relative_to(repo_root).as_posix()
    except ValueError:
        return artifact.name


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _run_git_text(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _default_build_id(*, version: str, commit: str) -> str:
    commit_short = commit[:12] if commit else "unknown"
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{version}-{commit_short}-{timestamp}"


def _env_first(*names: str) -> str:
    for name in names:
        value = str(os.getenv(name) or "").strip()
        if value:
            return value
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
