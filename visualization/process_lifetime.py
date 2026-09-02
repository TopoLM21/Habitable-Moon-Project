"""Tie render workers to their owner, including forced GUI termination."""
from __future__ import annotations

import ctypes
import os
import signal
import sys
from uuid import uuid4


def _windows_api():
    from ctypes import wintypes as w
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    for name, args, result in (
        ("CreateJobObjectW", [w.LPVOID, w.LPCWSTR], w.HANDLE),
        ("OpenJobObjectW", [w.DWORD, w.BOOL, w.LPCWSTR], w.HANDLE),
        ("SetInformationJobObject", [w.HANDLE, ctypes.c_int, w.LPVOID, w.DWORD], w.BOOL),
        ("AssignProcessToJobObject", [w.HANDLE, w.HANDLE], w.BOOL),
        ("GetCurrentProcess", [], w.HANDLE),
        ("CloseHandle", [w.HANDLE], w.BOOL),
    ):
        function = getattr(api, name)
        function.argtypes, function.restype = args, result
    return api


class WorkerLifetime:
    def __init__(self):
        self.name = None
        self.handle = None
        if os.name != "nt":
            return
        from ctypes import wintypes as w

        class Basic(ctypes.Structure):
            _fields_ = [("ProcessTime", ctypes.c_longlong), ("JobTime", ctypes.c_longlong),
                        ("Flags", w.DWORD), ("MinWorkingSet", ctypes.c_size_t),
                        ("MaxWorkingSet", ctypes.c_size_t), ("ActiveProcesses", w.DWORD),
                        ("Affinity", ctypes.c_size_t), ("Priority", w.DWORD), ("Scheduling", w.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in
                        ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]

        class Extended(ctypes.Structure):
            _fields_ = [("Basic", Basic), ("IO", IO), ("ProcessMemory", ctypes.c_size_t),
                        ("JobMemory", ctypes.c_size_t), ("PeakProcess", ctypes.c_size_t),
                        ("PeakJob", ctypes.c_size_t)]

        self.api = _windows_api()
        self.name = "Local\\MoonRender-" + uuid4().hex
        self.handle = self.api.CreateJobObjectW(None, self.name)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = Extended()
        limits.Basic.Flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def close(self):
        if self.handle is not None:
            self.api.CloseHandle(self.handle)
            self.handle = None


def bind_worker_to_owner(job_name: str | None, owner_pid: int) -> None:
    if os.name == "nt":
        api = _windows_api()
        handle = api.OpenJobObjectW(0x0001, False, job_name)  # JOB_OBJECT_ASSIGN_PROCESS
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not api.AssignProcessToJobObject(handle, api.GetCurrentProcess()):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            # Only the owner retains a handle: killing it closes the job.
            api.CloseHandle(handle)
    elif sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
            raise OSError(ctypes.get_errno(), "Could not bind render-worker lifetime")
        if os.getppid() != owner_pid:
            os._exit(1)
    else:
        # Portable fallback for other platforms; Windows/Linux use OS enforcement.
        import multiprocessing
        import threading
        import time
        parent = multiprocessing.parent_process()
        def watch():
            while parent is not None and parent.is_alive():
                time.sleep(0.2)
            os._exit(1)
        threading.Thread(target=watch, daemon=True, name="render-owner-watch").start()


def process_diagnostics():
    """Read-only execution/memory evidence for render benchmark reports."""
    if os.name != "nt":
        return {}
    from ctypes import wintypes as w
    api = _windows_api()
    process = api.GetCurrentProcess()
    api.GetProcessAffinityMask.argtypes = [w.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    affinity, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not api.GetProcessAffinityMask(process, ctypes.byref(affinity), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    class Memory(ctypes.Structure):
        _fields_ = [("cb", w.DWORD), ("PageFaults", w.DWORD)] + [(name, ctypes.c_size_t) for name in
            ("PeakWorkingSet", "WorkingSet", "PeakPaged", "Paged", "PeakNonPaged", "NonPaged", "Pagefile", "PeakPagefile")]
    memory = Memory()
    memory.cb = ctypes.sizeof(memory)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE, ctypes.POINTER(Memory), w.DWORD]
    if not psapi.GetProcessMemoryInfo(process, ctypes.byref(memory), memory.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"affinity_mask": hex(affinity.value), "peak_working_set_bytes": memory.PeakWorkingSet}
