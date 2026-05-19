# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import logging
import os
import sys
import webbrowser

from cai.shared.constants import CAI_CONFIG_HOME

logger = logging.getLogger(__name__)

_FIRST_RUN_MARKER = CAI_CONFIG_HOME / ".dashboard_opened"


def _is_first_run() -> bool:
    return not _FIRST_RUN_MARKER.exists()


def _mark_first_run_done() -> None:
    _FIRST_RUN_MARKER.parent.mkdir(parents=True, exist_ok=True)
    _FIRST_RUN_MARKER.touch()


def _safe_print_stderr(text: str) -> None:
    try:
        print(text, file=sys.stderr)
    except (OSError, ValueError, UnicodeError):
        logger.debug("Could not print startup banner", exc_info=True)


def print_startup_banner(port: int) -> None:
    dashboard_url = f"http://localhost:{port}"
    try:
        first_run = _is_first_run()
    except OSError:
        first_run = False
        logger.debug("Could not read first-run marker", exc_info=True)

    banner = f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   ███████╗██╗  ██╗ ██████╗                                            ║
║   ██╔════╝╚██╗██╔╝██╔═══██╗                                           ║
║   █████╗   ╚███╔╝ ██║   ██║                                           ║
║   ██╔══╝   ██╔██╗ ██║   ██║                                           ║
║   ███████╗██╔╝ ██╗╚██████╔╝                                           ║
║   ╚══════╝╚═╝  ╚═╝ ╚═════╝                                            ║
║                                                                       ║
║   Distributed AI Inference Cluster                                    ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║  🌐 Dashboard & API Ready                                             ║
║                                                                       ║
║  {dashboard_url}{" " * (69 - len(dashboard_url))}║
║                                                                       ║
║  Click the URL above to open the dashboard in your browser            ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝

"""

    _safe_print_stderr(banner)

    if first_run:
        # Skip browser open when running inside the native macOS app —
        # FirstLaunchPopout.swift handles the auto-open with a countdown.
        if not os.environ.get("CAI_RUNTIME_DIR"):
            try:
                webbrowser.open(dashboard_url)
                logger.info("First run detected — opening dashboard in browser")
            except Exception:
                logger.debug("Could not auto-open browser", exc_info=True)
        try:
            _mark_first_run_done()
        except OSError:
            logger.debug("Could not update first-run marker", exc_info=True)

