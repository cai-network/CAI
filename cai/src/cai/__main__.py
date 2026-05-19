# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from collections.abc import Sequence
from multiprocessing import freeze_support
from typing import Final

from cai.main import main

INLINE_CODE_FLAG: Final[str] = "-c"


def _maybe_run_inline_code(argv: Sequence[str]) -> bool:
    """
    Reproduce the bare minimum of Python's `-c` flag so multiprocessing
    helper processes (for example the resource tracker) can execute.
    """

    try:
        flag_index = argv.index(INLINE_CODE_FLAG)
    except ValueError:
        return False

    code_index = flag_index + 1
    if code_index >= len(argv):
        return False

    inline_code = argv[code_index]
    sys.argv = ["-c", *argv[code_index + 1 :]]
    namespace: dict[str, object] = {"__name__": "__main__"}
    exec(inline_code, namespace, namespace)
    return True


if __name__ == "__main__":
    if _maybe_run_inline_code(sys.argv):
        sys.exit(0)
    freeze_support()
    main()
