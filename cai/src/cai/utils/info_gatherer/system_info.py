# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import platform
import shutil
import socket
import subprocess
import sys
import os
from subprocess import CalledProcessError

import psutil
from anyio import run_process, to_thread

from cai.shared.types.profiling import InterfaceType, NetworkInterfaceInfo


def _windows_hidden_process_kwargs() -> dict[str, object]:
    if not sys.platform.startswith("win"):
        return {}

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return {
        "startupinfo": startupinfo,
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
    }


def _run_hidden_subprocess(
    command: list[str],
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **_windows_hidden_process_kwargs(),
    )


def get_os_version() -> str:
    """Return the OS version string for this node.

    On macOS this is the macOS version (e.g. ``"15.3"``).
    On other platforms it falls back to the platform name (e.g. ``"Linux"``).
    """
    if sys.platform == "darwin":
        version = platform.mac_ver()[0]
        return version if version else "Unknown"
    return platform.system() or "Unknown"


async def get_os_build_version() -> str:
    """Return the macOS build version string (e.g. ``"24D5055b"``).

    On non-macOS platforms, returns ``"Unknown"``.
    """
    if sys.platform != "darwin":
        return "Unknown"

    try:
        process = await run_process(["sw_vers", "-buildVersion"])
    except CalledProcessError:
        return "Unknown"

    return process.stdout.decode("utf-8", errors="replace").strip() or "Unknown"


async def get_friendly_name() -> str:
    """
    Asynchronously gets the 'Computer Name' (friendly name) of a Mac.
    e.g., "John's MacBook Pro"
    Returns the name as a string, or None if an error occurs or not on macOS.
    """
    hostname = socket.gethostname()

    if sys.platform != "darwin":
        return hostname

    try:
        process = await run_process(["scutil", "--get", "ComputerName"])
    except CalledProcessError:
        return hostname

    return process.stdout.decode("utf-8", errors="replace").strip() or hostname


async def _get_interface_types_from_networksetup() -> dict[str, InterfaceType]:
    """Parse networksetup -listallhardwareports to get interface types."""
    if sys.platform != "darwin":
        return {}

    try:
        result = await run_process(["networksetup", "-listallhardwareports"])
    except CalledProcessError:
        return {}

    types: dict[str, InterfaceType] = {}
    current_type: InterfaceType = "unknown"

    for line in result.stdout.decode().splitlines():
        if line.startswith("Hardware Port:"):
            port_name = line.split(":", 1)[1].strip()
            if "Wi-Fi" in port_name:
                current_type = "wifi"
            elif "Ethernet" in port_name or "LAN" in port_name:
                current_type = "ethernet"
            elif port_name.startswith("Thunderbolt"):
                current_type = "thunderbolt"
            else:
                current_type = "unknown"
        elif line.startswith("Device:"):
            device = line.split(":", 1)[1].strip()
            # enX is ethernet adapters or thunderbolt - these must be deprioritised
            if device.startswith("en") and device not in ["en0", "en1"]:
                current_type = "maybe_ethernet"
            types[device] = current_type

    return types


async def get_network_interfaces() -> list[NetworkInterfaceInfo]:
    """
    Retrieves detailed network interface information on macOS.
    Parses output from 'networksetup -listallhardwareports' and 'ifconfig'
    to determine interface names, IP addresses, and types (ethernet, wifi, vpn, other).
    Returns a list of NetworkInterfaceInfo objects.
    """
    interfaces_info: list[NetworkInterfaceInfo] = []
    interface_types = await _get_interface_types_from_networksetup()

    for iface, services in psutil.net_if_addrs().items():
        for service in services:
            match service.family:
                case socket.AF_INET | socket.AF_INET6:
                    interfaces_info.append(
                        NetworkInterfaceInfo(
                            name=iface,
                            ip_address=service.address,
                            interface_type=interface_types.get(iface, "unknown"),
                        )
                    )
                case _:
                    pass

    return interfaces_info


async def get_model_and_chip() -> tuple[str, str]:
    """Get Mac system information using system_profiler."""
    model = "Unknown Model"
    chip = "Unknown Chip"

    # TODO: better non mac support
    if sys.platform != "darwin":
        return (model, chip)

    try:
        process = await run_process(
            [
                "system_profiler",
                "SPHardwareDataType",
            ]
        )
    except CalledProcessError:
        return (model, chip)

    # less interested in errors here because this value should be hard coded
    output = process.stdout.decode().strip()

    model_line = next(
        (line for line in output.split("\n") if "Model Name" in line), None
    )
    model = model_line.split(": ")[1] if model_line else "Unknown Model"

    chip_line = next((line for line in output.split("\n") if "Chip" in line), None)
    chip = chip_line.split(": ")[1] if chip_line else "Unknown Chip"

    return (model, chip)


def get_cpu_core_counts() -> tuple[int | None, int | None]:
    """Return physical and logical CPU core counts."""

    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)
    return (
        int(physical) if isinstance(physical, int) and physical > 0 else None,
        int(logical) if isinstance(logical, int) and logical > 0 else None,
    )


def _get_total_vram_bytes_dxgi() -> int | None:
    """Best-effort dedicated VRAM total via DXGI without spawning subprocesses."""

    if not sys.platform.startswith("win"):
        return None

    try:
        import ctypes
        from ctypes import POINTER, Structure, byref, c_size_t, c_void_p
        from ctypes.wintypes import DWORD, LONG, UINT, ULONG, WCHAR
    except ImportError:
        return None

    HRESULT = LONG
    DXGI_ERROR_NOT_FOUND = 0x887A0002
    DXGI_ADAPTER_FLAG_SOFTWARE = 0x2

    class GUID(Structure):
        _fields_ = [
            ("Data1", DWORD),
            ("Data2", ctypes.c_ushort),
            ("Data3", ctypes.c_ushort),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    def _guid(value: str) -> GUID:
        cleaned = value.strip().strip("{}")
        data1, data2, data3, data4a, data4b = cleaned.split("-")
        data4 = bytes.fromhex(data4a + data4b)
        return GUID(
            Data1=int(data1, 16),
            Data2=int(data2, 16),
            Data3=int(data3, 16),
            Data4=(ctypes.c_ubyte * 8)(*data4),
        )

    class LUID(Structure):
        _fields_ = [("LowPart", DWORD), ("HighPart", LONG)]

    class DXGI_ADAPTER_DESC1(Structure):
        _fields_ = [
            ("Description", WCHAR * 128),
            ("VendorId", DWORD),
            ("DeviceId", DWORD),
            ("SubSysId", DWORD),
            ("Revision", DWORD),
            ("DedicatedVideoMemory", c_size_t),
            ("DedicatedSystemMemory", c_size_t),
            ("SharedSystemMemory", c_size_t),
            ("AdapterLuid", LUID),
            ("Flags", UINT),
        ]

    class IDXGIFactory1(Structure):
        pass

    class IDXGIAdapter1(Structure):
        pass

    ReleaseFactoryFunc = ctypes.WINFUNCTYPE(ULONG, POINTER(IDXGIFactory1))
    EnumAdapters1Func = ctypes.WINFUNCTYPE(
        HRESULT,
        POINTER(IDXGIFactory1),
        UINT,
        POINTER(POINTER(IDXGIAdapter1)),
    )
    ReleaseAdapterFunc = ctypes.WINFUNCTYPE(ULONG, POINTER(IDXGIAdapter1))
    GetDesc1Func = ctypes.WINFUNCTYPE(
        HRESULT,
        POINTER(IDXGIAdapter1),
        POINTER(DXGI_ADAPTER_DESC1),
    )

    class IDXGIFactory1Vtbl(Structure):
        _fields_ = [
            ("QueryInterface", c_void_p),
            ("AddRef", c_void_p),
            ("Release", ReleaseFactoryFunc),
            ("SetPrivateData", c_void_p),
            ("SetPrivateDataInterface", c_void_p),
            ("GetPrivateData", c_void_p),
            ("GetParent", c_void_p),
            ("EnumAdapters", c_void_p),
            ("MakeWindowAssociation", c_void_p),
            ("GetWindowAssociation", c_void_p),
            ("CreateSwapChain", c_void_p),
            ("CreateSoftwareAdapter", c_void_p),
            ("EnumAdapters1", EnumAdapters1Func),
            ("IsCurrent", c_void_p),
        ]

    class IDXGIAdapter1Vtbl(Structure):
        _fields_ = [
            ("QueryInterface", c_void_p),
            ("AddRef", c_void_p),
            ("Release", ReleaseAdapterFunc),
            ("SetPrivateData", c_void_p),
            ("SetPrivateDataInterface", c_void_p),
            ("GetPrivateData", c_void_p),
            ("GetParent", c_void_p),
            ("EnumOutputs", c_void_p),
            ("GetDesc", c_void_p),
            ("CheckInterfaceSupport", c_void_p),
            ("GetDesc1", GetDesc1Func),
        ]

    IDXGIFactory1._fields_ = [("lpVtbl", POINTER(IDXGIFactory1Vtbl))]
    IDXGIAdapter1._fields_ = [("lpVtbl", POINTER(IDXGIAdapter1Vtbl))]

    try:
        dxgi = ctypes.WinDLL("dxgi.dll")
    except OSError:
        return None

    create_factory = dxgi.CreateDXGIFactory1
    create_factory.argtypes = [POINTER(GUID), POINTER(c_void_p)]
    create_factory.restype = HRESULT

    factory_ptr = c_void_p()
    hr = int(create_factory(byref(_guid("770aae78-f26f-4dba-a829-253c83d1b387")), byref(factory_ptr)))
    if hr != 0 or not factory_ptr.value:
        return None

    factory = ctypes.cast(factory_ptr, POINTER(IDXGIFactory1))
    total_vram_bytes = 0
    adapter_index = 0

    try:
        while True:
            adapter = POINTER(IDXGIAdapter1)()
            result = int(factory.contents.lpVtbl.contents.EnumAdapters1(factory, adapter_index, byref(adapter)))
            if result == 0:
                desc = DXGI_ADAPTER_DESC1()
                if (
                    adapter
                    and int(adapter.contents.lpVtbl.contents.GetDesc1(adapter, byref(desc))) == 0
                    and not (int(desc.Flags) & DXGI_ADAPTER_FLAG_SOFTWARE)
                ):
                    total_vram_bytes += int(desc.DedicatedVideoMemory or 0)
                if adapter:
                    adapter.contents.lpVtbl.contents.Release(adapter)
                adapter_index += 1
                continue
            if (result & 0xFFFFFFFF) == DXGI_ERROR_NOT_FOUND:
                break
            return None
    finally:
        factory.contents.lpVtbl.contents.Release(factory)

    return total_vram_bytes if total_vram_bytes > 0 else None


async def get_total_vram_bytes() -> int | None:
    """Best-effort total dedicated VRAM across visible GPUs."""

    if sys.platform.startswith("win"):
        try:
            dxgi_total = await to_thread.run_sync(_get_total_vram_bytes_dxgi)
        except RuntimeError:
            dxgi_total = None
        if dxgi_total is not None:
            return dxgi_total

    if str(
        os.getenv("CAI_DISABLE_NVIDIA_SMI_VRAM_PROBE")
        or ""
    ).strip().lower() in {"1", "true", "yes", "on"}:
        return None

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None

    try:
        process = await to_thread.run_sync(
            _run_hidden_subprocess,
            [
                nvidia_smi,
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
        )
    except (CalledProcessError, FileNotFoundError, OSError, RuntimeError):
        return None

    if process.returncode != 0:
        return None

    total_vram_bytes = 0
    for raw_line in process.stdout.decode("utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            total_vram_bytes += int(float(line)) * 1024 * 1024
        except ValueError:
            continue

    return total_vram_bytes if total_vram_bytes > 0 else None

