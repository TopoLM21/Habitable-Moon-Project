"""Small, best-effort diagnostics for an isolated numerical worker.

Only stage labels and small metadata are recorded: simulation arrays and frame
locals are deliberately excluded. The watchdog is implemented by faulthandler,
so a stuck native call holding the GIL can still leave a Python stack trace.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import faulthandler
import json
from pathlib import Path
import re
import sys
import threading
import time
import traceback
import uuid
import os


EVENT_PREFIX = "@@MOON_DIAGNOSTICS@@"
MAX_LOG_BYTES = 4 * 1024 * 1024
WATCHDOG_SECONDS = 30


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _details(values: dict) -> dict:
    """Keep caller metadata bounded without invoking large object reprs."""
    result = {}
    for key, value in list(values.items())[:24]:
        if isinstance(value, str):
            value = value[:512]
        elif value is not None and not isinstance(value, (bool, int, float)):
            value = f"<{type(value).__name__}>"
        result[str(key)[:80]] = value
    return result


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _rotate(path: Path, additional_bytes: int = 0) -> None:
    if path.exists() and path.stat().st_size + additional_bytes > MAX_LOG_BYTES:
        path.replace(path.with_name(path.name + ".1"))


class WorkerDiagnostics:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory).resolve()
        self._pid = os.getpid()
        self._session_token = os.environ.get("MOON_DIAGNOSTICS_SESSION_TOKEN", "")
        self._lock = threading.Lock()
        self._publish_lock = threading.Lock()
        self._watchdog_lock = threading.Lock()
        self._dump_lock = threading.Lock()
        self._stages: list[dict] = []
        self._stage_metadata_keys: dict[int, tuple[str, ...]] = {}
        self._workers: dict[int, dict] = {}
        self._worker_metadata_keys: dict[int, tuple[str, ...]] = {}
        self._events: deque[dict] = deque(maxlen=100)
        self._failure: dict | None = None
        self._errors: set[str] = set()
        self._next_stage_id = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._watchdog_file = None
        self._started = False
        self._closed = False
        self._last_request_id = ""

    def _snapshot(self, kind: str) -> dict:
        now = time.monotonic()
        with self._lock:
            stages = [dict(stage, elapsed_seconds=max(0.0, now - stage["started_monotonic"]))
                      for stage in self._stages]
            # Keep stdout packets bounded even with 32 plate workers. Full
            # per-call metadata lives in each worker's assignment journal.
            keys = ("assignment_call", "plate_id", "time_myr", "backend", "candidates",
                    "source_count", "used_targets", "assignment_phase", "matched", "rows",
                    "searched_rows", "scanned_edges", "current_search_rows", "frontier_entries")
            workers = [dict(worker, details={key: worker["details"][key] for key in keys
                                             if key in worker["details"]},
                            elapsed_seconds=max(0.0, now - worker["started_monotonic"]))
                       for worker in list(self._workers.values())[:32]]
        current = None
        if stages:
            inherited = {}
            for stage in stages:
                inherited.update(stage["details"])
            if workers:
                inherited["active_workers"] = len(workers)
                inherited["worker_summary"] = "; ".join(
                    f"Плита {worker['details'].get('plate_id', '?')}: "
                    f"{worker['details'].get('matched', '?')}/"
                    f"{worker['details'].get('source_count', '?')}"
                    for worker in workers[:4]
                ) + (f"; ещё {len(workers) - 4}" if len(workers) > 4 else "")
            current = dict(stages[-1], details=inherited)
        return {"kind": kind, "pid": self._pid, "session_token": self._session_token,
                "timestamp": _timestamp(),
                "stage": current, "stages": stages, "workers": workers,
                "stage_path": [stage["name"] for stage in stages]}

    @staticmethod
    def _stdout(packet: dict) -> None:
        try:
            # ASCII JSON escapes also form valid UTF-8 on Windows legacy consoles.
            sys.stdout.write(EVENT_PREFIX + json.dumps(packet, ensure_ascii=True) + "\n")
            sys.stdout.flush()
        except Exception:
            pass  # A closed parent pipe must not change simulation results.

    def _error(self, operation: str, error: Exception) -> None:
        with self._lock:
            if operation in self._errors:
                return
            self._errors.add(operation)
        packet = self._snapshot("diagnostic")
        packet["error"] = f"{operation}: {type(error).__name__}: {str(error)[:512]}"
        self._stdout(packet)

    def _emit(self, kind: str, **extra) -> None:
        # Dump collection never holds this lock. Stage transitions contend only
        # with a short heartbeat publication, not with traceback/report writing.
        with self._publish_lock:
            if self._closed and kind != "finished":
                return
            packet = self._snapshot(kind)
            packet.update(extra)
            with self._lock:
                self._events.append(packet)
            self._stdout(packet)
            try:
                path = self.directory / "events.jsonl"
                line = json.dumps(packet, ensure_ascii=False) + "\n"
                _rotate(path, len(line.encode("utf-8")))
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(line)
                    stream.flush()
                _write_json(self.directory / "status.json", packet)
            except Exception as error:
                self._error("write diagnostics", error)

    def _arm_watchdog(self) -> None:
        try:
            with self._watchdog_lock:
                if self._closed:
                    return
                faulthandler.cancel_dump_traceback_later()
                path = self.directory / "stacks.txt"
                if self._watchdog_file is not None and path.stat().st_size > MAX_LOG_BYTES:
                    self._watchdog_file.close()
                    self._watchdog_file = None
                if self._watchdog_file is None:
                    _rotate(path)
                    self._watchdog_file = path.open("a", encoding="utf-8")
                packet = self._snapshot("watchdog")
                self._watchdog_file.write("\n" + json.dumps(packet, ensure_ascii=False) + "\n")
                self._watchdog_file.flush()
                faulthandler.dump_traceback_later(
                    WATCHDOG_SECONDS, repeat=False, file=self._watchdog_file,
                )
        except Exception as error:
            self._error("arm stack watchdog", error)

    def start(self) -> None:
        if self._started or self._closed:
            return
        self._started = True
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except Exception as error:
            self._error("create diagnostic directory", error)
        self._emit("stage")
        self._arm_watchdog()
        self._thread = threading.Thread(target=self._service, name="moon-diagnostics", daemon=True)
        try:
            self._thread.start()
        except Exception as error:
            self._thread = None
            self._error("start diagnostic service", error)

    @contextmanager
    def stage(self, name: str, **details):
        # Helpers may run concurrently; their frames remain visible in dumps,
        # while the progress label always follows the worker's main thread.
        if threading.current_thread() is not threading.main_thread():
            yield
            return
        self.start()
        with self._lock:
            self._next_stage_id += 1
            stage = {"name": str(name)[:256], "details": _details(details),
                     "stage_id": self._next_stage_id, "started_monotonic": time.monotonic()}
            self._stages.append(stage)
            self._stage_metadata_keys[stage["stage_id"]] = tuple(stage["details"])
        self._emit("stage")
        self._arm_watchdog()
        outcome = "completed"
        try:
            yield
        except BaseException as error:
            outcome = "interrupted" if isinstance(error, (KeyboardInterrupt, GeneratorExit)) else "failed"
            # A step context spans a generator yield. Iterator teardown during
            # another exception injects GeneratorExit; it is not the root cause.
            if not isinstance(error, GeneratorExit):
                failure = self._snapshot("failure")
                failure["exception_type"] = type(error).__name__
                failure["exception_message"] = str(error)[:1024]
                with self._lock:
                    if self._failure is None:
                        self._failure = failure
            raise
        finally:
            completed = dict(stage, outcome=outcome,
                             elapsed_seconds=max(0.0, time.monotonic() - stage["started_monotonic"]))
            with self._lock:
                self._stages = [entry for entry in self._stages if entry is not stage]
                self._stage_metadata_keys.pop(stage["stage_id"], None)
            self._emit("stage", completed_stage=completed)
            self._arm_watchdog()

    def update(self, **details) -> None:
        """Publish actual progress within the current stage without resetting its age."""
        if threading.current_thread() is not threading.main_thread():
            return
        with self._lock:
            if self._closed or not self._stages:
                return
            stage = self._stages[-1]
            incoming = _details(details)
            metadata_keys = self._stage_metadata_keys[stage["stage_id"]]
            # Preserve the stage's identity/context, while keeping at most 24
            # recent progress fields. New counters must not sit behind a full
            # old dictionary and silently disappear at the metadata limit.
            metadata = {key: incoming.get(key, stage["details"][key]) for key in metadata_keys}
            previous_progress = {
                key: value for key, value in stage["details"].items()
                if key not in incoming and key not in metadata_keys
            }
            progress = _details({**incoming, **previous_progress})
            updated = {**metadata, **progress}
            if updated == stage["details"]:
                return
            stage["details"] = updated
        self._emit("progress")
        self._arm_watchdog()

    @contextmanager
    def worker_stage(self, name: str, **details):
        """Observe a plate worker without mixing its stage into the main stack."""
        identity = threading.get_ident()
        worker = {"thread_id": identity, "name": str(name)[:256],
                  "details": _details(details), "started_monotonic": time.monotonic()}
        with self._lock:
            self._workers[identity] = worker
            self._worker_metadata_keys[identity] = tuple(worker["details"])
        self._emit("progress")
        outcome = "completed"
        try:
            yield
        except BaseException as error:
            outcome = "failed"
            failure = self._snapshot("failure")
            failure["stage"] = dict(worker, elapsed_seconds=time.monotonic() - worker["started_monotonic"])
            failure["stage_path"] = [*failure["stage_path"], worker["name"]]
            failure["exception_type"] = type(error).__name__
            failure["exception_message"] = str(error)[:1024]
            with self._lock:
                if self._failure is None:
                    self._failure = failure
            self._emit("failure", failure=failure)
            raise
        finally:
            completed = dict(worker, outcome=outcome,
                             elapsed_seconds=time.monotonic() - worker["started_monotonic"])
            with self._lock:
                self._workers.pop(identity, None)
                self._worker_metadata_keys.pop(identity, None)
            self._emit("progress", completed_worker=completed)

    def update_worker(self, **details) -> None:
        identity = threading.get_ident()
        with self._lock:
            worker = self._workers.get(identity)
            if worker is None or self._closed:
                return
            incoming = _details(details)
            keys = self._worker_metadata_keys[identity]
            metadata = {key: incoming.get(key, worker["details"][key]) for key in keys}
            previous = {key: value for key, value in worker["details"].items()
                        if key not in incoming and key not in keys}
            worker["details"] = {**metadata, **_details({**incoming, **previous})}
        self._emit("progress")
        # One active worker must not reset the main watchdog while another is
        # stuck. Its stack still dumps after the coordinator waits 30 seconds.

    def dump(self, reason: str, request_id: str = "") -> dict:
        """Write a bounded report and all Python thread stacks; never raise for I/O."""
        request_id = str(request_id)[:160] or uuid.uuid4().hex
        identifier = re.sub(r"[^A-Za-z0-9_-]", "_", request_id)[:80]
        response = {"request_id": request_id, "pid": self._pid,
                    "session_token": self._session_token, "report": "", "stack": ""}
        with self._dump_lock:
            packet = self._snapshot("diagnostic")
            packet.update(reason=str(reason)[:1024], request_id=request_id,
                          python_version=sys.version, watchdog_file=str(self.directory / "stacks.txt"))
            exception_value = sys.exc_info()[1]
            exception = traceback.format_exc()
            if exception_value is not None:
                # A function wrapper may dump before its stage context unwinds.
                # Preserve that innermost location for the final GUI summary.
                failure = self._snapshot("failure")
                failure["exception_type"] = type(exception_value).__name__
                failure["exception_message"] = str(exception_value)[:1024]
                with self._lock:
                    if self._failure is None:
                        self._failure = failure
            with self._lock:
                packet["recent_events"] = list(self._events)
                packet["failure"] = self._failure
            if exception.strip() != "NoneType: None":
                packet["exception"] = exception[-32768:]
            stack = self.directory / f"stack-{identifier}.txt"
            try:
                with stack.open("w", encoding="utf-8") as stream:
                    stream.write(f"{packet['timestamp']} | {packet['reason']}\n")
                    stream.flush()
                    faulthandler.dump_traceback(file=stream, all_threads=True)
                response["stack"] = str(stack)
            except Exception as error:
                self._error("write requested stacks", error)
            report = self.directory / f"report-{identifier}.json"
            try:
                _write_json(report, packet)
                response["report"] = str(report)
            except Exception as error:
                self._error("write requested report", error)
            try:
                _write_json(self.directory / "diagnostic-response.json", response)
            except Exception as error:
                self._error("write diagnostic response", error)
        self._emit("diagnostic", **response, reason=str(reason)[:1024], failure=packet["failure"])
        return response

    def _service(self) -> None:
        last_heartbeat = time.monotonic()
        while not self._stop.wait(0.5):
            if time.monotonic() - last_heartbeat >= 2.0:
                self._emit("heartbeat")
                last_heartbeat = time.monotonic()
            try:
                path = self.directory / "diagnostic-request.json"
                if not path.exists():
                    continue
                with path.open("r", encoding="utf-8") as stream:
                    request = json.loads(stream.read(65536))
                if not isinstance(request, dict):
                    continue
                if request.get("session_token", "") != self._session_token:
                    continue
                request_id = str(request.get("request_id", ""))[:160]
                if not request_id or request_id == self._last_request_id:
                    continue
                if request.get("pid", self._pid) != self._pid:
                    continue
                self._last_request_id = request_id
                self.dump(request.get("reason", "User requested diagnostics"), request_id)
            except (FileNotFoundError, json.JSONDecodeError):
                continue  # Parent may be replacing/writing its request file.
            except Exception as error:
                self._error("read diagnostic request", error)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        with self._watchdog_lock:
            try:
                faulthandler.cancel_dump_traceback_later()
            except Exception as error:
                self._error("cancel stack watchdog", error)
            if self._watchdog_file is not None:
                try:
                    self._watchdog_file.close()
                except Exception as error:
                    self._error("close stack watchdog", error)
                self._watchdog_file = None
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        with self._lock:
            self._stages.clear()
            self._stage_metadata_keys.clear()
            self._workers.clear()
            self._worker_metadata_keys.clear()
        self._emit("finished")
