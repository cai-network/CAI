# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for XDG Base Directory Specification compliance."""

import os
import sys
from pathlib import Path
from unittest import mock


def test_xdg_paths_on_linux():
    """Test that XDG paths are used on Linux when XDG env vars are set."""
    with (
        mock.patch.dict(
            os.environ,
            {
                "XDG_CONFIG_HOME": "/tmp/test-config",
                "XDG_DATA_HOME": "/tmp/test-data",
                "XDG_CACHE_HOME": "/tmp/test-cache",
            },
            clear=False,
        ),
        mock.patch.object(sys, "platform", "linux"),
    ):
        # Re-import to pick up mocked values
        import importlib

        import cai.shared.constants as constants

        importlib.reload(constants)

        assert Path("/tmp/test-config/cai") == constants.CAI_CONFIG_HOME
        assert Path("/tmp/test-data/cai") == constants.CAI_DATA_HOME
        assert Path("/tmp/test-cache/cai") == constants.CAI_CACHE_HOME


def test_xdg_default_paths_on_linux():
    """Test that XDG default paths are used on Linux when env vars are not set."""
    # Remove XDG env vars and CAI_HOME
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("XDG_") and k != "CAI_HOME"
    }
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.object(sys, "platform", "linux"),
    ):
        import importlib

        import cai.shared.constants as constants

        importlib.reload(constants)

        home = Path.home()
        assert home / ".config" / "cai" == constants.CAI_CONFIG_HOME
        assert home / ".local/share" / "cai" == constants.CAI_DATA_HOME
        assert home / ".cache" / "cai" == constants.CAI_CACHE_HOME


def test_legacy_cai_home_takes_precedence():
    """Test that CAI_HOME environment variable takes precedence for backward compatibility."""
    with mock.patch.dict(
        os.environ,
        {
            "CAI_HOME": ".custom-cai",
            "XDG_CONFIG_HOME": "/tmp/test-config",
        },
        clear=False,
    ):
        import importlib

        import cai.shared.constants as constants

        importlib.reload(constants)

        home = Path.home()
        assert home / ".custom-cai" == constants.CAI_CONFIG_HOME
        assert home / ".custom-cai" == constants.CAI_DATA_HOME


def test_macos_uses_traditional_paths():
    """Test that macOS uses traditional ~/.cai directory."""
    # Remove CAI_HOME to ensure we test the default behavior
    env = {k: v for k, v in os.environ.items() if k != "CAI_HOME"}
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.object(sys, "platform", "darwin"),
    ):
        import importlib

        import cai.shared.constants as constants

        importlib.reload(constants)

        home = Path.home()
        assert home / ".cai" == constants.CAI_CONFIG_HOME
        assert home / ".cai" == constants.CAI_DATA_HOME
        assert home / ".cai" == constants.CAI_CACHE_HOME


def test_node_id_in_config_dir():
    """Test that node ID keypair is in the config directory."""
    import cai.shared.constants as constants

    assert constants.CAI_NODE_ID_KEYPAIR.parent == constants.CAI_CONFIG_HOME


def test_models_in_data_dir():
    """Test that default models directory is in the data directory."""
    # Clear CAI_MODELS_DIRS to test default behavior
    env = {k: v for k, v in os.environ.items() if k != "CAI_MODELS_DIRS"}
    with mock.patch.dict(os.environ, env, clear=True):
        import importlib

        import cai.shared.constants as constants

        importlib.reload(constants)

        assert constants.CAI_DEFAULT_MODELS_DIR.parent == constants.CAI_DATA_HOME


def test_default_dir_always_prepended_to_models_dirs():
    """Test that the default models dir is always the first entry in CAI_MODELS_DIRS."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("CAI_MODELS_DIRS", "CAI_MODELS_READ_ONLY_DIRS", "CAI_HOME")
    }
    env["CAI_MODELS_DIRS"] = "/tmp/custom-models"
    with mock.patch.dict(os.environ, env, clear=True):
        import importlib

        import cai.shared.constants as constants

        importlib.reload(constants)

        assert constants.CAI_MODELS_DIRS[0] == constants.CAI_DEFAULT_MODELS_DIR
        assert Path("/tmp/custom-models") in constants.CAI_MODELS_DIRS


def test_default_models_dir_override():
    """Test that CAI_DEFAULT_MODELS_DIR can be overridden via env var."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in (
            "CAI_MODELS_DIRS",
            "CAI_MODELS_READ_ONLY_DIRS",
            "CAI_HOME",
            "CAI_DEFAULT_MODELS_DIR",
        )
    }
    env["CAI_DEFAULT_MODELS_DIR"] = "/Volumes/FastSSD/CAI-models"
    with mock.patch.dict(os.environ, env, clear=True):
        import importlib

        import cai.shared.constants as constants

        importlib.reload(constants)

        assert Path("/Volumes/FastSSD/CAI-models") == constants.CAI_DEFAULT_MODELS_DIR
        assert constants.CAI_MODELS_DIRS[0] == constants.CAI_DEFAULT_MODELS_DIR


def test_default_dir_only_entry_when_env_unset():
    """Test that CAI_MODELS_DIRS contains only the default when env var is not set."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("CAI_MODELS_DIRS", "CAI_MODELS_READ_ONLY_DIRS", "CAI_HOME")
    }
    with mock.patch.dict(os.environ, env, clear=True):
        import importlib

        import cai.shared.constants as constants

        importlib.reload(constants)

        assert constants.CAI_MODELS_DIRS == (constants.CAI_DEFAULT_MODELS_DIR,)


def test_overlap_between_dirs_and_read_only_dirs():
    """Test that a directory in both lists is excluded from writable dirs."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("CAI_MODELS_DIRS", "CAI_MODELS_READ_ONLY_DIRS", "CAI_HOME")
    }
    env["CAI_MODELS_DIRS"] = os.pathsep.join(["/tmp/shared", "/tmp/writable-only"])
    env["CAI_MODELS_READ_ONLY_DIRS"] = os.pathsep.join(["/tmp/shared", "/tmp/ro-only"])
    with mock.patch.dict(os.environ, env, clear=True):
        import importlib

        import cai.shared.constants as constants

        importlib.reload(constants)

        # /tmp/shared should be excluded from writable dirs
        assert Path("/tmp/shared") not in constants.CAI_MODELS_DIRS
        assert Path("/tmp/writable-only") in constants.CAI_MODELS_DIRS
        # /tmp/shared should still be in read-only dirs
        assert Path("/tmp/shared") in constants.CAI_MODELS_READ_ONLY_DIRS
        assert Path("/tmp/ro-only") in constants.CAI_MODELS_READ_ONLY_DIRS


def test_empty_read_only_dirs_when_unset():
    """Test that CAI_MODELS_READ_ONLY_DIRS is empty when env var is not set."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("CAI_MODELS_DIRS", "CAI_MODELS_READ_ONLY_DIRS", "CAI_HOME")
    }
    with mock.patch.dict(os.environ, env, clear=True):
        import importlib

        import cai.shared.constants as constants

        importlib.reload(constants)

        assert constants.CAI_MODELS_READ_ONLY_DIRS == ()


def test_models_dirs_use_platform_path_separator():
    """Test that model dir env vars use the OS path separator, not a hard-coded colon."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("CAI_MODELS_DIRS", "CAI_MODELS_READ_ONLY_DIRS", "CAI_HOME")
    }
    env["CAI_MODELS_DIRS"] = r"C:\models-writable;D:\models-extra"
    env["CAI_MODELS_READ_ONLY_DIRS"] = r"E:\models-ro"
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch("os.pathsep", ";"),
    ):
        import importlib

        import cai.shared.constants as constants

        importlib.reload(constants)

        assert Path(r"C:\models-writable") in constants.CAI_MODELS_DIRS
        assert Path(r"D:\models-extra") in constants.CAI_MODELS_DIRS
        assert Path(r"E:\models-ro") in constants.CAI_MODELS_READ_ONLY_DIRS

