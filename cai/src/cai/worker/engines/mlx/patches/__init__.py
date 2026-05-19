# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from cai.worker.engines.mlx.patches.opt_batch_gen import apply_batch_gen_patch
from cai.worker.engines.mlx.patches.standard_yarn_rope import patch_yarn_rope

_applied = False


def apply_mlx_patches() -> None:
    global _applied
    if _applied:
        return
    _applied = True
    patch_yarn_rope()
    apply_batch_gen_patch()

