"""Shared execution choices and self-only, non-elevating CPU priority policy.

Call apply_process_priority only inside a disposable simulation/render process,
never in the desktop GUI. No affinity, power plan, BLAS, I/O or memory settings
are changed. On Unix, raising priority again may require privileges; exit the
child and launch a new one instead of trying to restore it in a context manager.
"""
from __future__ import annotations

import ctypes
import os

RENDER_WORKER_CHOICES = (1, 2, 4, 6, 8, 12)
PROCESS_PRIORITY_CHOICES = ("normal", "below_normal")


def _windows_priority_api():
    from ctypes import wintypes as w
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.GetCurrentProcess.argtypes, api.GetCurrentProcess.restype = [], w.HANDLE
    api.GetPriorityClass.argtypes, api.GetPriorityClass.restype = [w.HANDLE], w.DWORD
    api.SetPriorityClass.argtypes, api.SetPriorityClass.restype = [w.HANDLE, w.DWORD], w.BOOL
    return api


def read_process_priority():
    if os.name == "nt":
        api = _windows_priority_api()
        value = api.GetPriorityClass(api.GetCurrentProcess())
        if not value:
            raise ctypes.WinError(ctypes.get_last_error())
        return {"kind": "windows_priority_class", "value": value}
    if hasattr(os, "getpriority"):
        return {"kind": "unix_nice", "value": os.getpriority(os.PRIO_PROCESS, 0)}
    return {"kind": "unavailable", "value": None}


def apply_process_priority(mode):
    """Lower this process only. 'normal' preserves the launching environment."""
    if mode not in PROCESS_PRIORITY_CHOICES:
        raise ValueError(f"Process priority must be one of {PROCESS_PRIORITY_CHOICES}")
    before = read_process_priority()
    if mode == "normal":
        return before
    if os.name == "nt":
        # Do not raise a process which was already launched with idle priority.
        if before["value"] != 0x40:
            api = _windows_priority_api()
            if not api.SetPriorityClass(api.GetCurrentProcess(), 0x4000):
                raise ctypes.WinError(ctypes.get_last_error())
    elif hasattr(os, "setpriority"):
        # Absolute, idempotent nice value: workers may inherit it from the parent.
        os.setpriority(os.PRIO_PROCESS, 0, max(5, before["value"]))
    else:
        raise RuntimeError("Lower CPU priority is not supported on this platform")
    after = read_process_priority()
    if not is_lower_priority(after):
        raise RuntimeError(f"Operating system did not apply lower CPU priority: {after}")
    return after


def is_lower_priority(priority):
    return ((priority["kind"] == "windows_priority_class" and priority["value"] in (0x4000, 0x40))
            or (priority["kind"] == "unix_nice" and priority["value"] >= 5))
