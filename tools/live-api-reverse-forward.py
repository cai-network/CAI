# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys


def main() -> int:
    print(
        "CAI live API reverse-forward automation is maintainer-only and is "
        "not shipped in the public repository.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
