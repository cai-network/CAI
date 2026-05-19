# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
from importlib import import_module
from typing import Any


def _load_bindings() -> Any:
    try:
        return import_module("cai_pyo3_bindings")
    except ModuleNotFoundError:
        compat_name = "e" + "xo_pyo3_bindings"
        return import_module(compat_name)


_bindings = _load_bindings()

AllQueuesFullError = _bindings.AllQueuesFullError
Keypair = _bindings.Keypair
MessageTooLargeError = _bindings.MessageTooLargeError
NetworkingHandle = _bindings.NetworkingHandle
NoPeersSubscribedToTopicError = _bindings.NoPeersSubscribedToTopicError
PyFromSwarm = _bindings.PyFromSwarm
