# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from cai_compute_chain.cai_desktop_app import (
    CaiDesktopConfig,
    CaiDesktopController,
    _maybe_schedule_embedded_portable_auto_update,
    _portable_auto_update_error_interval_seconds,
    _portable_auto_update_interval_seconds,
    _prepare_embedded_runtime_env,
    build_parser,
    build_cai_desktop_command,
    build_cai_desktop_env,
    cai_home_path,
    handle_existing_instance,
    load_peer_book_for_repo,
    peer_book_path_for_repo,
    main,
    resolve_language,
    resolve_repo_root,
    save_desktop_language,
    sync_peer_book_for_repo_from_bootstrap,
    should_enforce_single_instance,
    write_desktop_icon,
)


def make_repo(root: Path) -> Path:
    (root / "cai" / "src" / "cai").mkdir(parents=True)
    (root / "cai" / "src" / "cai" / "main.py").write_text("", encoding="utf-8")
    (root / "cai" / ".venv-win" / "Scripts").mkdir(parents=True)
    (root / "cai" / ".venv-win" / "Scripts" / "python.exe").write_text(
        "", encoding="utf-8"
    )
    (root / "tools").mkdir()
    (root / "tools" / "run-cai-main.py").write_text("", encoding="utf-8")
    (root / "src").mkdir()
    return root


def add_patched_shard_engine(repo: Path) -> Path:
    engine = repo / "_internal" / "llama.cpp" / "llama-cai-shard-engine.exe"
    engine.parent.mkdir(parents=True, exist_ok=True)
    engine.write_text("", encoding="utf-8")
    return engine


class FakePopen:
    instances: list["FakePopen"] = []

    def __init__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        self.returncode = None
        self.terminated = False
        self.killed = False
        FakePopen.instances.append(self)

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class CaiDesktopAppTests(unittest.TestCase):
    def setUp(self) -> None:
        FakePopen.instances.clear()

    def test_resolve_repo_root_from_explicit_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(Path(tmp))

            self.assertEqual(resolve_repo_root(str(repo)), repo.resolve())

    def test_load_peer_book_for_repo_reads_multiaddrs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(Path(tmp))
            peer = "/ip4/127.0.0.1/tcp/52416/p2p/node"
            (repo / ".cai-peer-book.json").write_text(
                json.dumps([peer, "", peer]), encoding="utf-8"
            )

            self.assertEqual(load_peer_book_for_repo(repo), [peer])

    def test_load_peer_book_for_repo_accepts_windows_utf8_bom(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(Path(tmp))
            peer = "/ip4/85.137.164.250/tcp/52416"
            (repo / ".cai-peer-book.json").write_text(
                "\ufeff" + json.dumps([peer]), encoding="utf-8"
            )

            self.assertEqual(load_peer_book_for_repo(repo), [peer])

    def test_build_command_uses_runtime_main_and_peer_book(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(Path(tmp))
            peer = "/ip4/127.0.0.1/tcp/52416/p2p/local"
            (repo / ".cai-peer-book.json").write_text(json.dumps([peer]), encoding="utf-8")
            config = CaiDesktopConfig(
                repo_root=repo,
                api_port=1111,
                libp2p_port=2222,
                no_downloads=True,
                force_master=True,
            )

            command = build_cai_desktop_command(config)

            self.assertEqual(command[0], str(repo / "cai" / ".venv-win" / "Scripts" / "python.exe"))
            self.assertEqual(command[1], str(repo / "tools" / "run-cai-main.py"))
            self.assertIn("-m", command)
            self.assertIn("--no-downloads", command)
            self.assertIn("--api-port", command)
            self.assertIn("1111", command)
            self.assertIn("--libp2p-port", command)
            self.assertIn("2222", command)
            self.assertIn("--bootstrap-peers", command)
            self.assertIn(peer, command[-1])

    def test_offline_command_skips_bootstrap_peers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(Path(tmp))
            config = CaiDesktopConfig(repo_root=repo, offline=True)

            command = build_cai_desktop_command(config)

            self.assertNotIn("--bootstrap-peers", command)
            self.assertIn("--offline", command)

    def test_env_sets_runtime_paths_and_keeps_dashboard_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(Path(tmp))
            engine = add_patched_shard_engine(repo)
            config = CaiDesktopConfig(repo_root=repo)
            desktop_state_dir = str(Path(tmp) / "desktop-state")

            with patch.dict(
                os.environ,
                {
                    "CAI_DESKTOP_STATE_DIR": desktop_state_dir,
                    "CAI_DISABLE_DASHBOARD": "1",
                    "PYTHONPATH": "old-path",
                },
                clear=True,
            ):
                env = build_cai_desktop_env(config)

            self.assertEqual(env["CAI_RUNTIME_REPO"], str(repo))
            self.assertEqual(env["CAI_RUNTIME_SRC"], str(repo / "src"))
            self.assertEqual(env["CAI_HOME"], str(cai_home_path(config)))
            self.assertEqual(env["CAI_HOME"], str(cai_home_path(config)))
            self.assertEqual(env["EXO_LIBP2P_NAMESPACE"], "cai-ai-net")
            self.assertEqual(env["CAI_LANG"], "en")
            self.assertEqual(env["CAI_ENABLE_TASK_LEVEL_TRANSPORT_JOBS"], "1")
            self.assertEqual(env["CAI_REQUIRE_TASK_LEVEL_TRANSPORT_JOBS"], "1")
            self.assertEqual(env["CAI_ALLOW_TASK_LEVEL_TRANSPORT_PRIVATE_MODELS"], "1")
            self.assertEqual(env["CAI_TASK_LEVEL_TRANSPORT_REQUIRE_RUNTIME_READY"], "1")
            self.assertEqual(env["CAI_TASK_LEVEL_TRANSPORT_REQUIRE_SHARD_READINESS"], "1")
            self.assertEqual(env["CAI_TASK_LEVEL_TRANSPORT_REQUIRE_DATA_PLANE_ROUTE"], "1")
            self.assertEqual(
                env["CAI_TASK_LEVEL_TRANSPORT_REQUIRE_PROVEN_DATA_PLANE_ROUTE"],
                "1",
            )
            self.assertEqual(env["CAI_TASK_LEVEL_TRANSPORT_EXECUTOR_COUNT"], "2")
            self.assertEqual(env["CAI_OWNED_TRANSPORT_GENERATION_ENABLED"], "1")
            self.assertEqual(env["CAI_OWNED_TRANSPORT_GENERATION_REQUIRED"], "1")
            self.assertEqual(
                env["CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_RUNTIME_READY"],
                "1",
            )
            self.assertEqual(
                env["CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_SHARD_READINESS"],
                "1",
            )
            self.assertEqual(
                env["CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_DATA_PLANE_ROUTE"],
                "1",
            )
            self.assertEqual(
                env["CAI_OWNED_TRANSPORT_GENERATION_REQUIRE_PROVEN_DATA_PLANE_ROUTE"],
                "1",
            )
            self.assertEqual(env["CAI_JOB_EXECUTION_FIRST_RESPONSE_TIMEOUT_SECONDS"], "120")
            self.assertNotIn("CAI_EXECUTION_CAI_URL", env)
            self.assertEqual(env["CAI_PRIVATE_NETWORK_MODEL_MIN_NODES"], "2")
            self.assertEqual(env["CAI_LLM_SHARD_ADAPTER"], "native_bridge")
            self.assertEqual(env["CAI_LLM_PATCHED_BINARY_REQUIRE_REAL_LAYER_EXECUTION"], "1")
            self.assertIn(str(engine), env["CAI_LLM_PATCHED_BINARY_COMMAND"])
            self.assertNotIn("CAI_DISABLE_DASHBOARD", env)
            self.assertIn(str(repo / "cai" / "src"), env["PYTHONPATH"])
            self.assertIn(str(repo / "src"), env["PYTHONPATH"])
            self.assertIn("old-path", env["PYTHONPATH"])

    def test_env_preserves_explicit_task_level_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(Path(tmp))
            config = CaiDesktopConfig(repo_root=repo)

            with patch.dict(
                os.environ,
                {
                    "LOCALAPPDATA": str(Path(tmp) / "appdata"),
                    "CAI_EXECUTION_CAI_URL": "http://validator.example:52415",
                    "CAI_TASK_LEVEL_TRANSPORT_EXECUTOR_COUNT": "3",
                    "CAI_TASK_LEVEL_TRANSPORT_REQUIRE_RUNTIME_READY": "1",
                    "CAI_JOB_EXECUTION_FIRST_RESPONSE_TIMEOUT_SECONDS": "50",
                },
                clear=True,
            ):
                env = build_cai_desktop_env(config)

            self.assertEqual(env["CAI_EXECUTION_CAI_URL"], "http://validator.example:52415")
            self.assertEqual(env["CAI_TASK_LEVEL_TRANSPORT_EXECUTOR_COUNT"], "3")
            self.assertEqual(env["CAI_TASK_LEVEL_TRANSPORT_REQUIRE_RUNTIME_READY"], "1")
            self.assertEqual(env["CAI_JOB_EXECUTION_FIRST_RESPONSE_TIMEOUT_SECONDS"], "50")

    def test_embedded_runtime_env_sets_network_defaults_for_direct_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(Path(tmp))
            engine = add_patched_shard_engine(repo)

            with (
                patch.dict(
                    os.environ,
                    {"LOCALAPPDATA": str(Path(tmp) / "appdata")},
                    clear=True,
                ),
                patch(
                    "cai_compute_chain.cai_desktop_app.resolve_repo_root",
                    return_value=repo,
                ),
            ):
                _prepare_embedded_runtime_env()

                self.assertEqual(os.environ["CAI_LIBP2P_NAMESPACE"], "cai-ai-net")
                self.assertEqual(os.environ["EXO_LIBP2P_NAMESPACE"], "cai-ai-net")
                self.assertEqual(os.environ["CAI_RUNTIME_REPO"], str(repo))
                self.assertEqual(
                    os.environ["CAI_HOME"],
                    str(cai_home_path(CaiDesktopConfig(repo_root=repo))),
                )
                self.assertEqual(os.environ["CAI_LLM_SHARD_ADAPTER"], "native_bridge")
                self.assertIn(
                    str(engine),
                    os.environ["CAI_LLM_PATCHED_BINARY_COMMAND"],
                )

    def test_embedded_portable_auto_update_schedules_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(Path(tmp))
            config = CaiDesktopConfig(repo_root=repo)

            with (
                patch(
                    "cai_compute_chain.cai_desktop_app._is_embedded_portable_app",
                    return_value=True,
                ),
                patch.dict(os.environ, {}, clear=True),
                patch(
                    "cai_compute_chain.update_channel.maybe_stage_portable_auto_update_on_launch",
                    return_value={
                        "restartScheduled": True,
                        "message": "scheduled",
                    },
                ) as schedule_mock,
            ):
                result = _maybe_schedule_embedded_portable_auto_update(
                    config,
                    ["--no-tray"],
                )
                deadline = time.monotonic() + 2
                while not schedule_mock.called and time.monotonic() < deadline:
                    time.sleep(0.01)

            self.assertIsNotNone(result)
            self.assertTrue(result["started"])
            schedule_mock.assert_called_once()
            self.assertEqual(schedule_mock.call_args.args[0], repo)
            self.assertEqual(
                schedule_mock.call_args.kwargs["relaunch_command"][1:],
                ["--no-tray"],
            )
            self.assertEqual(schedule_mock.call_args.kwargs["timeout_sec"], 60)

    def test_embedded_portable_auto_update_uses_short_polling_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_portable_auto_update_interval_seconds(), 120)
            self.assertEqual(_portable_auto_update_error_interval_seconds(), 60)

    def test_embedded_portable_auto_update_interval_env_overrides_default(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CAI_AUTO_UPDATE_INTERVAL_SECONDS": "7",
                "CAI_AUTO_UPDATE_ERROR_INTERVAL_SECONDS": "11",
            },
            clear=True,
        ):
            self.assertEqual(_portable_auto_update_interval_seconds(), 7)
            self.assertEqual(_portable_auto_update_error_interval_seconds(), 11)

    def test_resolve_language_defaults_to_english_and_honors_saved_preference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {"CAI_DESKTOP_STATE_DIR": tmp},
                clear=True,
            ):
                self.assertEqual(resolve_language("auto"), "en")
                save_desktop_language("ru")
                self.assertEqual(resolve_language("auto"), "ru")
                save_desktop_language("es")
                self.assertEqual(resolve_language("auto"), "es")
                self.assertEqual(resolve_language("de"), "de")
                self.assertEqual(resolve_language("en"), "en")

    def test_desktop_parser_defaults_to_english_language(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.language, "en")

    def test_desktop_parser_accepts_additional_languages(self) -> None:
        for language in ("es", "de", "fr", "zh"):
            args = build_parser().parse_args(["--language", language])
            self.assertEqual(args.language, language)

    def test_handle_existing_instance_opens_saved_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(Path(tmp))
            config = CaiDesktopConfig(repo_root=repo, open_browser=True)
            state_dir = Path(tmp) / "desktop-state"
            state_dir.mkdir(parents=True, exist_ok=True)
            (state_dir / "desktop-instance.json").write_text(
                json.dumps({"dashboardUrl": "http://127.0.0.1:59999/"}),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"CAI_DESKTOP_STATE_DIR": str(state_dir)}, clear=True):
                with patch("builtins.print") as print_mock:
                    with patch("webbrowser.open") as open_mock:
                        result = handle_existing_instance(config)

            self.assertEqual(result, 0)
            open_mock.assert_called_once_with("http://127.0.0.1:59999/")
            self.assertGreaterEqual(print_mock.call_count, 1)

    def test_single_instance_is_default_but_can_be_overridden(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(sys, "frozen", False, create=True):
                self.assertTrue(should_enforce_single_instance())
            with patch.object(sys, "frozen", True, create=True):
                self.assertTrue(should_enforce_single_instance())

        with patch.dict(
            os.environ,
            {"CAI_ALLOW_MULTIPLE_DESKTOP_INSTANCES": "1"},
            clear=True,
        ):
            with patch.object(sys, "frozen", True, create=True):
                self.assertFalse(should_enforce_single_instance())

        with patch.dict(
            os.environ,
            {"CAI_ENFORCE_SINGLE_INSTANCE": "1"},
            clear=True,
        ):
            with patch.object(sys, "frozen", False, create=True):
                self.assertTrue(should_enforce_single_instance())

    def test_controller_starts_and_stops_runtime_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(Path(tmp))
            config = CaiDesktopConfig(repo_root=repo, open_browser=False)
            controller = CaiDesktopController(config)

            with (
                patch("subprocess.Popen", FakePopen),
                patch(
                    "cai_compute_chain.cai_desktop_app.sync_peer_book_for_repo_from_bootstrap",
                    return_value=(peer_book_path_for_repo(repo), [], []),
                ) as sync_mock,
                patch(
                    "cai_compute_chain.cai_desktop_app.ensure_desktop_firewall",
                    return_value=None,
                ) as firewall_mock,
            ):
                controller.start()
                self.assertTrue(controller.running)
                controller.start()
                self.assertEqual(len(FakePopen.instances), 1)
                controller.stop()
            firewall_mock.assert_called_once_with(config)
            sync_mock.assert_called_once()

            process = FakePopen.instances[0]
            self.assertEqual(process.command[1], str(repo / "tools" / "run-cai-main.py"))
            self.assertEqual(process.kwargs["cwd"], str(repo))
            self.assertEqual(process.kwargs["env"]["CAI_RUNTIME_REPO"], str(repo))
            self.assertTrue(process.terminated)
            self.assertTrue((repo / ".cai" / "desktop.log").exists())

    def test_sync_peer_book_for_repo_from_bootstrap_crawls_overlay_advertised_peers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(Path(tmp))
            peer = "/ip4/85.137.164.250/tcp/52416/p2p/bootstrap"
            peer_a = "/ip4/26.97.29.153/tcp/52426/p2p/node-a"
            peer_b = "/dns4/node-b.example/tcp/52426/p2p/node-b"
            save_path = peer_book_path_for_repo(repo)
            save_path.write_text(json.dumps([peer]), encoding="utf-8")
            config = CaiDesktopConfig(repo_root=repo)
            payload_by_url = {
                "http://192.145.29.212:52415/state": {
                    "overlayAdvertisedPeers": {},
                },
                "http://85.137.164.250:52415/state": {
                    "overlayAdvertisedPeers": {
                        "node-a": [{"address": peer_a}],
                    },
                },
                "http://26.97.29.153:52415/state": {
                    "overlayAdvertisedPeers": {
                        "node-b": [peer_b],
                    },
                },
                "http://node-b.example:52415/state": {
                    "overlayAdvertisedPeers": {},
                },
            }

            class _FakeResponse:
                def __init__(self, payload: dict[str, object]) -> None:
                    self.payload = payload

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def read(self) -> bytes:
                    return json.dumps(self.payload).encode("utf-8")

            def _fake_urlopen(state_url: str, timeout: float | None = None):
                _ = timeout
                return _FakeResponse(payload_by_url[state_url])

            with patch(
                "urllib.request.urlopen",
                side_effect=_fake_urlopen,
            ):
                path, imported, tried = sync_peer_book_for_repo_from_bootstrap(config)

            self.assertEqual(path, save_path)
            self.assertEqual(imported, [peer_a, peer_b])
            self.assertEqual(
                tried,
                [
                    "http://192.145.29.212:52415/state",
                    "http://85.137.164.250:52415/state",
                    "http://26.97.29.153:52415/state",
                    "http://node-b.example:52415/state",
                ],
            )
            self.assertEqual(
                load_peer_book_for_repo(repo),
                [
                    peer,
                    peer_a,
                    peer_b,
                ],
            )

    def test_open_dashboard_uses_local_cai_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(Path(tmp))
            controller = CaiDesktopController(CaiDesktopConfig(repo_root=repo, api_port=3333))

            with patch("webbrowser.open") as open_mock:
                controller.open_dashboard()

            open_mock.assert_called_once_with("http://127.0.0.1:3333/")

    def test_write_desktop_icon_creates_square_multi_size_ico(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            icon_path = Path(tmp) / "cai.ico"

            write_desktop_icon(icon_path, repo_root=Path(__file__).resolve().parents[1])

            with Image.open(icon_path) as image:
                self.assertEqual(
                    sorted(image.ico.sizes()),
                    [
                        (16, 16),
                        (20, 20),
                        (24, 24),
                        (32, 32),
                        (40, 40),
                        (48, 48),
                        (64, 64),
                        (128, 128),
                        (256, 256),
                    ],
                )

    def test_main_locks_wallet_session_on_start_and_exit_for_single_instance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(Path(tmp))
            with (
                patch("cai_compute_chain.cai_desktop_app.resolve_repo_root", return_value=repo),
                patch("cai_compute_chain.cai_desktop_app.should_enforce_single_instance", return_value=True),
                patch("cai_compute_chain.cai_desktop_app._find_available_port", side_effect=lambda port, **_: port),
                patch("cai_compute_chain.cai_desktop_app.run_console", return_value=0),
                patch("cai_compute_chain.cai_desktop_app.lock_wallet") as lock_wallet_mock,
                patch("cai_compute_chain.cai_desktop_app.DesktopSingleInstanceGuard.acquire", return_value=True),
                patch("cai_compute_chain.cai_desktop_app.DesktopSingleInstanceGuard.write_state"),
                patch("cai_compute_chain.cai_desktop_app.DesktopSingleInstanceGuard.clear_state"),
                patch("cai_compute_chain.cai_desktop_app.DesktopSingleInstanceGuard.release"),
            ):
                result = main(["--no-tray", "--no-browser"])

            self.assertEqual(result, 0)
            self.assertEqual(lock_wallet_mock.call_count, 2)

    def test_main_continues_to_console_when_auto_update_starts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(Path(tmp))
            with (
                patch("cai_compute_chain.cai_desktop_app.resolve_repo_root", return_value=repo),
                patch("cai_compute_chain.cai_desktop_app.should_enforce_single_instance", return_value=True),
                patch("cai_compute_chain.cai_desktop_app._find_available_port", side_effect=lambda port, **_: port),
                patch("cai_compute_chain.cai_desktop_app.run_console", return_value=42) as run_console_mock,
                patch(
                    "cai_compute_chain.cai_desktop_app._maybe_schedule_embedded_portable_auto_update",
                    return_value={
                        "started": True,
                        "message": "checking",
                    },
                ),
                patch("cai_compute_chain.cai_desktop_app.lock_wallet"),
                patch("cai_compute_chain.cai_desktop_app.DesktopSingleInstanceGuard.acquire", return_value=True),
                patch("cai_compute_chain.cai_desktop_app.DesktopSingleInstanceGuard.write_state"),
                patch("cai_compute_chain.cai_desktop_app.DesktopSingleInstanceGuard.clear_state"),
                patch("cai_compute_chain.cai_desktop_app.DesktopSingleInstanceGuard.release"),
            ):
                result = main(["--no-tray", "--no-browser"])

            self.assertEqual(result, 42)
            run_console_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()

