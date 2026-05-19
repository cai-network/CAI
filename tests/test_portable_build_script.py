# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = REPO_ROOT / "tools" / "build-portable-win.ps1"
PUBLISH_SCRIPT = REPO_ROOT / "tools" / "publish-validator-update.ps1"
PUBLISH_BAT = REPO_ROOT / "tools" / "publish-validator-update.bat"
INSTALL_VALIDATOR_SCRIPT = REPO_ROOT / "tools" / "install-mainnet-validator-vps.ps1"
VPS_SSH_SCRIPT = REPO_ROOT / "tools" / "cai-vps-ssh.ps1"


class PortableBuildScriptTests(unittest.TestCase):
    def test_portable_build_does_not_package_runtime_data_by_default(self) -> None:
        script = BUILD_SCRIPT.read_text(encoding="utf-8")
        spec = (REPO_ROOT / "packaging" / "pyinstaller" / "cai-portable.spec").read_text(
            encoding="utf-8"
        )

        self.assertIn("[switch]$PreserveData", script)
        self.assertIn("function Remove-PortableRuntimeData", script)
        self.assertIn("function Remove-PortableRuntimeState", script)
        self.assertIn("function Remove-PortableDirectory", script)
        self.assertIn('".cai-local"', script)
        self.assertIn('".cai-peer-book.json"', script)
        self.assertIn("Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop", script)
        self.assertIn('throw "Unable to clean $Description directory: $Path"', script)
        self.assertIn("if ($PreserveData -and (Test-Path $existingDataDir))", script)
        self.assertNotIn("Copy-Item -LiteralPath $SourceIconPath", script)
        self.assertIn("--write-icon", script)
        self.assertIn("$env:CAI_ICON_PATH = $IconPath", script)
        self.assertIn('DATAS.append((str(ICON_PATH), "assets"))', spec)
        self.assertIn("Move-Item -LiteralPath $PortableStageDir -Destination $OutputDir", script)

        smoke_index = script.index("Portable cai.exe --doctor failed")
        clean_index = script.index(
            "Remove-PortableRuntimeState -PortableRoot $OutputDir",
            smoke_index,
        )
        zip_index = script.index("if ($Zip)", clean_index)
        self.assertLess(smoke_index, clean_index)
        self.assertLess(clean_index, zip_index)

        finalizer_index = script.index("} finally {")
        finalizer = script[finalizer_index:]
        self.assertIn("Remove-PortableRuntimeState -PortableRoot $OutputDir", finalizer)

    def test_clean_portable_check_blocks_api_token_file(self) -> None:
        checker = (REPO_ROOT / "tools" / "check-portable-clean.ps1").read_text(
            encoding="utf-8"
        )

        self.assertIn("^\\.cai-api-token$", checker)
        self.assertIn("^\\.cai-peer-book\\.json$", checker)
        self.assertIn("unlocked-wallet-signing-key", checker)

    def test_publish_validator_update_has_no_arg_vps_defaults(self) -> None:
        script = PUBLISH_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('$DefaultRemoteHost = "192.145.29.212"', script)
        self.assertIn(
            '$DefaultHostKey = "ssh-ed25519 255 SHA256:5erRiM/RocLfH1VPaK+ZvMynOZtt4supMAG6vVY68Nk"',
            script,
        )
        self.assertIn('$DefaultUser = "root"', script)
        self.assertIn('$DefaultApiPort = 52415', script)
        self.assertIn('$DefaultServiceName = "exo"', script)
        self.assertIn("CAI_VPS_PASSWORD", script)
        self.assertIn(".cai-local\\secrets\\vps-ssh-password.txt", script)
        self.assertIn("$EffectiveBuildDashboard = -not $NoBuildDashboard", script)
        self.assertIn(
            "$EffectiveRebuildPortable = (-not $NoRebuildPortable) -and (-not $SkipPortable)",
            script,
        )
        self.assertIn(
            "dashboard=$EffectiveBuildDashboard portable=$EffectiveRebuildPortable",
            script,
        )

    def test_publish_validator_update_bat_calls_powershell_script(self) -> None:
        bat = PUBLISH_BAT.read_text(encoding="utf-8")

        self.assertIn("publish-validator-update.ps1", bat)
        self.assertIn("powershell -NoProfile -ExecutionPolicy Bypass", bat)
        self.assertIn("%*", bat)

    def test_install_validator_does_not_copy_owner_seed_by_default(self) -> None:
        script = INSTALL_VALIDATOR_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('[string]$RemoteHost = "192.145.29.212"', script)
        self.assertIn(
            '[string]$HostKey = "ssh-ed25519 255 SHA256:5erRiM/RocLfH1VPaK+ZvMynOZtt4supMAG6vVY68Nk"',
            script,
        )
        self.assertIn("[switch]$ProvisionOwnerWalletOnVps", script)
        self.assertIn("derive_seed_wallet_identity", script)
        self.assertIn("Get-ConfiguredOwnerTreasuryIdentity", script)
        self.assertIn("if ($ProvisionOwnerWalletOnVps) {", script)
        self.assertIn("Owner treasury seed/password were not copied to this VPS", script)
        self.assertIn("Validator mode auto-enable skipped", script)
        self.assertIn("owner seed on VPS", script)
        self.assertIn('mv -f "$PORTABLE_ZIP" "$CURRENT_LINK/.dist/CAI-portable.zip"', script)
        self.assertIn('/tmp/cai-validator-install-*) rm -rf "$REMOTE_STAGE"', script)
        self.assertIn("python3-venv", script)
        self.assertIn("uv python install 3.14", script)
        self.assertIn("sys.version_info >= (3, 13)", script)
        self.assertIn("rm -rf \"$SERVICE_VENV\"", script)
        self.assertIn('rm -rf "$CURRENT_LINK/cai/target" "$HOME/.cargo/registry" "$HOME/.cache/pip"', script)

    def test_large_vps_upload_streams_chunks_on_remote(self) -> None:
        script = VPS_SSH_SCRIPT.read_text(encoding="utf-8")

        self.assertIn(">> ", script)
        self.assertIn("rm -f", script)
        self.assertNotIn("/part-*", script)


if __name__ == "__main__":
    unittest.main()
