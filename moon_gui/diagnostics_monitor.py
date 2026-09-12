"""Small, bounded GUI-side record of a worker, independent of model imports."""

from __future__ import annotations

import codecs
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4


PROTOCOL_PREFIX = "@@MOON_DIAGNOSTICS@@"
MAX_LOG_BYTES = 2 * 1024 * 1024
MAX_LINE_CHARS = 65536


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class DiagnosticsMonitor:
    """Receive UTF-8 telemetry without assuming QProcess reads whole lines."""

    def __init__(self) -> None:
        self.session_dir: Path | None = None
        self.segment_dir: Path | None = None
        self.log_tail: deque[str] = deque(maxlen=120)
        self.launcher_pid = 0
        self.worker_pid = 0
        self.session_token = ""
        self.argv: list[str] = []
        self.options: dict[str, Any] = {}
        self.latest_event: dict[str, Any] = {}
        self.last_worker_diagnostic: dict[str, Any] = {}
        self.failure: dict[str, Any] | None = None
        self.stage: dict[str, Any] = {}
        self.stage_path: list[str] = []
        self.stage_received_at: float | None = None
        self.last_heartbeat_at: float | None = None
        self.segment_started_at: float | None = None
        self.segment_ended_at: float | None = None
        self.last_report: Path | None = None
        self.last_error = ""
        self._log_error_reported = False
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._output_buffer = ""

    def start_session(self, output_dir: Path, options: dict[str, Any]) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.session_dir = output_dir / "diagnostics" / f"{stamp}-{uuid4().hex[:8]}"
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.segment_dir = None
        self.latest_event = {}
        self.stage = {}
        self.stage_path = []
        self.last_worker_diagnostic = {}
        self.failure = None
        self.options = options
        self.log_tail.clear()
        self.last_report = None
        self.last_error = ""
        self._log_error_reported = False

    def start_segment(self, index: int, argv: list[str]) -> Path:
        if self.session_dir is None:
            raise RuntimeError("Diagnostics session has not been prepared")
        self.segment_dir = self.session_dir / f"segment-{index:03d}"
        self.segment_dir.mkdir(parents=True, exist_ok=False)
        self.argv = argv
        self.launcher_pid = 0
        self.worker_pid = 0
        # Windows venv launchers can spawn a different Python worker PID.
        # Bind telemetry to this segment before learning the worker's PID.
        self.session_token = uuid4().hex
        self.latest_event = {}
        self.last_worker_diagnostic = {}
        self.failure = None
        self.stage = {"name": "Запуск Python и загрузка диагностического модуля", "details": {}}
        self.stage_path = []
        self.stage_received_at = monotonic()
        self.last_heartbeat_at = None
        self.segment_started_at = self.stage_received_at
        self.segment_ended_at = None
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._output_buffer = ""
        return self.segment_dir

    def record_log(self, line: str) -> str | None:
        entry = f"{utc_timestamp()} {line[:8000]}"
        self.log_tail.append(entry)
        if self.session_dir is None:
            return None
        try:
            path = self.session_dir / "gui.log"
            if path.exists() and path.stat().st_size >= MAX_LOG_BYTES:
                path.replace(self.session_dir / "gui.previous.log")
            with path.open("a", encoding="utf-8") as handle:
                handle.write(entry + "\n")
        except OSError as exc:
            self.last_error = f"Не удалось записать журнал диагностики: {exc}"
            if not self._log_error_reported:
                self._log_error_reported = True
                return self.last_error
        return None

    def feed(self, data: bytes = b"", *, final: bool = False) -> list[str]:
        self._output_buffer += self._decoder.decode(data, final=final)
        # A split CRLF only introduces an ignored empty line; no text is lost.
        self._output_buffer = self._output_buffer.replace("\r\n", "\n").replace("\r", "\n")
        lines = self._output_buffer.split("\n")
        self._output_buffer = lines.pop()
        if final and self._output_buffer:
            lines.append(self._output_buffer)
            self._output_buffer = ""
        # Ordinary output may never print a newline. Keep memory bounded while
        # still displaying it; valid telemetry is much smaller than this cap.
        while len(self._output_buffer) > MAX_LINE_CHARS:
            lines.append(self._output_buffer[:MAX_LINE_CHARS])
            self._output_buffer = self._output_buffer[MAX_LINE_CHARS:]
        result: list[str] = []
        for line in lines:
            if not line.strip():
                continue
            marker = line.find(PROTOCOL_PREFIX)
            if marker < 0:
                result.append(line)
                continue
            if marker:
                result.append(line[:marker])
            try:
                event = json.loads(line[marker + len(PROTOCOL_PREFIX):])
            except (ValueError, TypeError):
                result.append(line[marker:])
                continue
            if not self._is_worker_packet(event):
                result.append(line[marker:])
                continue
            message = self.accept_event(event)
            if message:
                result.append(message)
        return result

    def _is_worker_packet(self, packet: Any) -> bool:
        if not isinstance(packet, dict) or not self.session_token:
            return False
        pid = packet.get("pid")
        return (
            packet.get("session_token") == self.session_token
            and type(pid) is int and pid > 0
            and (not self.worker_pid or pid == self.worker_pid)
        )

    def accept_event(self, event: dict[str, Any]) -> str | None:
        if not self._is_worker_packet(event):
            return None
        self.worker_pid = event["pid"]
        now = monotonic()
        self.latest_event = event
        self.last_heartbeat_at = now
        if event.get("kind") == "diagnostic" and event.get("report"):
            self.last_worker_diagnostic = event
        failure = event.get("failure")
        if event.get("kind") == "failure" and not isinstance(failure, dict):
            failure = event
        if isinstance(failure, dict):
            self.failure = failure
        old_stage_id = (self.stage.get("stage_id"), self.stage.get("started_monotonic"))
        stage = event.get("stage")
        if isinstance(stage, dict):
            self.stage = stage
            self.stage_received_at = now
            path = event.get("stage_path")
            if isinstance(path, list):
                self.stage_path = [str(part) for part in path]
        new_stage_id = (self.stage.get("stage_id"), self.stage.get("started_monotonic"))
        completed = event.get("completed_stage")
        if isinstance(completed, dict):
            if stage is None:
                self.stage = {"name": "Завершение процесса", "details": {}, "elapsed_seconds": 0.0}
                self.stage_path = []
                self.stage_received_at = now
            elapsed = completed.get("elapsed_seconds", 0.0)
            duration = f"{elapsed:.1f} с" if isinstance(elapsed, (float, int)) else str(elapsed)
            label = {"failed": "Ошибка на этапе", "interrupted": "Прерван этап"}.get(
                completed.get("outcome"), "Завершён этап")
            return f"{label}: {completed.get('name', '—')} ({duration})"
        if event.get("kind") == "stage" and old_stage_id != new_stage_id:
            return "Этап: " + (" → ".join(self.stage_path) or str(self.stage.get("name", "—")))
        if event.get("error"):
            self.last_error = str(event["error"])
            return "Диагностика: " + str(event["error"])
        return None

    def finish_segment(self) -> None:
        self.segment_ended_at = monotonic()

    def stage_elapsed(self) -> float:
        now = self.segment_ended_at or monotonic()
        try:
            received_elapsed = float(self.stage.get("elapsed_seconds", 0.0))
        except (ValueError, TypeError):
            received_elapsed = 0.0
        return max(0.0, received_elapsed + now - (self.stage_received_at or now))

    def heartbeat_age(self) -> float | None:
        if self.last_heartbeat_at is None:
            return None
        return max(0.0, monotonic() - self.last_heartbeat_at)

    def save_report(
        self, reason: str, context: dict[str, Any], request_id: str | None = None
    ) -> tuple[str, Path]:
        if self.segment_dir is None:
            raise RuntimeError("Нет папки диагностики текущего сегмента")
        request_id = request_id or uuid4().hex
        report = self.segment_dir / f"gui-report-{request_id}.json"
        worker_report = self.segment_dir / f"report-{request_id}.json"
        worker_stack = self.segment_dir / f"stack-{request_id}.txt"
        write_json_atomic(report, {
            "timestamp": utc_timestamp(),
            "reason": reason,
            "request_id": request_id,
            "pid": self.worker_pid or None,
            "worker_pid": self.worker_pid or None,
            "launcher_pid": self.launcher_pid or None,
            "session_token": self.session_token,
            "argv": self.argv,
            "options": self.options,
            "context": context,
            "stage": self.stage,
            "stage_path": self.stage_path,
            "stage_elapsed_seconds": self.stage_elapsed(),
            "seconds_since_heartbeat": self.heartbeat_age(),
            "heartbeat_note": "A heartbeat indicates monitor activity, not numerical progress.",
            "last_worker_event": self.latest_event,
            "failure": self.failure,
            "diagnostic_write_error": self.last_error or None,
            "log_tail": list(self.log_tail),
            "worker_diagnostics": {
                "status": "requested" if context.get("worker_running") else "worker_not_running",
                "report": str(worker_report),
                "stack": str(worker_stack),
                "watchdog_stacks": str(self.segment_dir / "stacks.txt"),
                "latest_status": str(self.segment_dir / "status.json"),
                "events": str(self.segment_dir / "events.jsonl"),
                "last_worker_dump": self.last_worker_diagnostic,
                "note": "Requested files may be absent if the worker cannot respond; inspect watchdog_stacks.",
            },
        })
        self.last_report = report
        return request_id, report

    def request_worker_dump(self, request_id: str, reason: str) -> None:
        if self.segment_dir is None:
            raise RuntimeError("Нет папки диагностики текущего сегмента")
        write_json_atomic(self.segment_dir / "diagnostic-request.json", {
            "request_id": request_id, "reason": reason,
            "session_token": self.session_token,
            **({"pid": self.worker_pid} if self.worker_pid else {}),
        })

    def read_response(self, request_id: str) -> dict[str, Any] | None:
        if self.segment_dir is None:
            return None
        try:
            response = json.loads((self.segment_dir / "diagnostic-response.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(response, dict) or response.get("request_id") != request_id:
            return None
        if not self._is_worker_packet(response):
            return None
        self.worker_pid = response["pid"]
        return response

    def finish_report(self, report: Path, status: str, response: dict[str, Any] | None = None) -> None:
        data = json.loads(report.read_text(encoding="utf-8"))
        data["pid"] = self.worker_pid or None
        data["worker_pid"] = self.worker_pid or None
        data["launcher_pid"] = self.launcher_pid or None
        data["worker_diagnostics"]["status"] = status
        if response is not None:
            data["worker_diagnostics"]["response"] = response
        write_json_atomic(report, data)
