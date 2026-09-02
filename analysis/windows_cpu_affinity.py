"""Benchmark-only CPU-set diagnostics; never changes global power settings."""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import os
import struct


def cpu_sets():
    if os.name != "nt":
        return []
    from ctypes import wintypes as w
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    function = api.GetSystemCpuSetInformation
    function.argtypes = [w.LPVOID, w.ULONG, ctypes.POINTER(w.ULONG), w.HANDLE, w.ULONG]
    function.restype = w.BOOL
    size = w.ULONG()
    function(None, 0, ctypes.byref(size), None, 0)
    if size.value == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_string_buffer(size.value)
    if not function(buffer, size, ctypes.byref(size), None, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    raw = buffer.raw
    result = []
    offset = 0
    while offset < size.value:
        length, kind = struct.unpack_from("<II", raw, offset)
        if length < 8 or offset + length > size.value:
            raise RuntimeError("Invalid CPU-set descriptor")
        if kind == 0 and length >= 32:
            identity, group, logical, core, cache, numa, efficiency, flags = struct.unpack_from("<IHBBBBBB", raw, offset + 8)
            result.append({"id": identity, "group": group, "logical": logical, "core": core,
                           "efficiency_class": efficiency, "flags": flags})
        offset += length
    return result


@contextmanager
def performance_core_benchmark(enabled=False):
    information = {"pinned": False, "cpu_sets": cpu_sets()}
    if not enabled:
        yield information
        return
    if os.name != "nt":
        raise RuntimeError("Performance-core selection is only implemented for Windows benchmarks")
    sets = information["cpu_sets"]
    if {item["group"] for item in sets} != {0}:
        raise RuntimeError("This benchmark affinity helper requires a single processor group")
    classes = {item["efficiency_class"] for item in sets}
    if len(classes) < 2:
        raise RuntimeError("Windows does not identify distinct CPU performance classes")
    from ctypes import wintypes as w
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.GetCurrentProcess.restype = w.HANDLE
    api.GetProcessAffinityMask.argtypes = [w.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    api.SetProcessAffinityMask.argtypes = [w.HANDLE, ctypes.c_size_t]
    process = api.GetCurrentProcess()
    previous, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not api.GetProcessAffinityMask(process, ctypes.byref(previous), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    selected = sum(1 << item["logical"] for item in sets if item["efficiency_class"] == max(classes)) & previous.value
    if not selected or not api.SetProcessAffinityMask(process, selected):
        raise ctypes.WinError(ctypes.get_last_error())
    information.update(pinned=True, previous_mask=hex(previous.value), selected_mask=hex(selected))
    try:
        yield information
    finally:
        if not api.SetProcessAffinityMask(process, previous.value):
            raise ctypes.WinError(ctypes.get_last_error())
