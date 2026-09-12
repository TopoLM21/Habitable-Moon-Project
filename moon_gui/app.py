"""PySide6 desktop application for checkpointed Moon Tectonics runs."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import os
from pathlib import Path
import sys
from time import monotonic
from typing import Any

import yaml

from PySide6.QtCore import (
    QProcess,
    QProcessEnvironment,
    QSize,
    Qt,
    QTimer,
    QUrl,
    Signal,
    QObject,
)
from PySide6.QtGui import QCloseEvent, QDesktopServices, QFont, QFontDatabase, QMovie, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .backend import (
    RunSpec,
    SUBDIVISION_CHOICES,
    build_segment_command,
    checkpoint_cell_count,
    checkpoint_name,
    discover_artifacts,
    load_run_metrics,
    preferred_preview,
    read_checkpoint_time,
    resolution_note,
    segment_targets,
    subdivision_for_cell_count,
    write_run_record,
    write_runtime_config,
)
from .genesis_schema import ORIGIN_LABELS_RU, SatelliteOrigin
from .diagnostics_monitor import DiagnosticsMonitor
from .timing import RunTiming, format_duration
from execution_policy import RENDER_WORKER_CHOICES


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "canonical_moon.yaml"


def simulation_environment(spec: RunSpec) -> QProcessEnvironment:
    """Prepare a child environment without importing or initializing CUDA."""

    environment = QProcessEnvironment.systemEnvironment()
    environment.insert("PYTHONUNBUFFERED", "1")
    environment.insert("MPLBACKEND", "Agg")
    old_pythonpath = environment.value("PYTHONPATH")
    environment.insert(
        "PYTHONPATH",
        str(spec.project_root)
        if not old_pythonpath
        else str(spec.project_root) + os.pathsep + old_pythonpath,
    )
    if spec.gpu_surface:
        for variable, directory in (
            ("CUPY_CACHE_DIR", "cupy"),
            ("CUDA_CACHE_PATH", "cuda"),
            ("MPLCONFIGDIR", "matplotlib"),
        ):
            if not environment.value(variable):
                cache = spec.output_dir / ".cache" / directory
                cache.mkdir(parents=True, exist_ok=True)
                environment.insert(variable, str(cache))
    return environment


def install_application_font(app: QApplication) -> None:
    """Install a bundled Unicode font when the host default is unavailable."""

    try:
        import matplotlib

        font_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            app.setFont(QFont(families[0], 10))
    except Exception:
        # The native platform font remains a valid fallback on normal desktops.
        pass


class SimulationController(QObject):
    log_line = Signal(str)
    state_changed = Signal(str)
    progress_changed = Signal(float, float)
    segment_completed = Signal(float, str)
    run_completed = Signal(str)
    run_failed = Signal(str)
    diagnostics_changed = Signal()
    diagnostics_notice = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.started.connect(self._process_started)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)
        self.spec: RunSpec | None = None
        self.targets: list[float] = []
        self.target_index = 0
        self.current_time = 0.0
        self.resume_checkpoint: Path | None = None
        self.active_checkpoint: Path | None = None
        self.pause_requested = False
        self.cancel_requested = False
        self.state = "Idle"
        self.timing = RunTiming()
        self.diagnostics = DiagnosticsMonitor()
        self.log_line.connect(self._record_diagnostic_log)
        self._segment_identity = 0
        self._active_segment_index = 0
        self._pending_dump: dict[str, Any] | None = None
        self._stop_started_at: float | None = None
        self._termination_sent = False
        self.diagnostics_timer = QTimer(self)
        self.diagnostics_timer.timeout.connect(self._poll_diagnostics)
        self.diagnostics_timer.start(200)

    def _set_state(self, value: str) -> None:
        self.state = value
        self.state_changed.emit(value)
        self.diagnostics_changed.emit()

    def is_active(self) -> bool:
        return self.state in {"Preparing", "Running", "Pausing", "Stopping"}

    def start(self, spec: RunSpec) -> None:
        if self.is_active():
            raise RuntimeError("A simulation is already active")
        spec = spec.normalized()
        spec.validate()
        self.timing = RunTiming(spec.start_time_myr(), spec.end_time_myr)
        self.diagnostics = DiagnosticsMonitor()
        self.spec = spec
        self._set_state("Preparing")
        try:
            if spec.resume_checkpoint is not None and spec.runtime_config.is_file():
                runtime_config = spec.runtime_config
            else:
                runtime_config = write_runtime_config(spec)
            write_run_record(spec, runtime_config)
            self.diagnostics.start_session(spec.output_dir, asdict(spec))
        except Exception:
            self._set_state("Error")
            raise
        self.current_time = spec.start_time_myr()
        self.resume_checkpoint = spec.resume_checkpoint
        self.targets = segment_targets(
            self.current_time,
            spec.end_time_myr,
            spec.checkpoint_interval_myr,
            spec.dt_myr,
        )
        self.target_index = 0
        self.pause_requested = False
        self.cancel_requested = False
        self._pending_dump = None
        self._stop_started_at = None
        self._termination_sent = False
        self.diagnostics_notice.emit(f"Диагностика: {self.diagnostics.session_dir}")
        self.progress_changed.emit(self.current_time, spec.end_time_myr)
        self.log_line.emit(
            f"Prepared v0.31 run: t={self.current_time:g} -> {spec.end_time_myr:g} Myr, "
            f"sub-{spec.subdivisions}, {len(self.targets)} checkpoint segment(s)."
        )
        mode = (
            f"CPU + GPU surface (CUDA device {spec.gpu_device})"
            if spec.gpu_surface
            else "optimized CPU" if spec.cpu_optimized else "reference CPU"
        )
        self.log_line.emit(
            f"Execution: {mode}; fast assignment={spec.assignment_optimized}, "
            f"CPU assignment columns={spec.assignment_columns}, "
            f"CPU boundary forces={spec.boundary_forces}."
        )
        self._start_next_segment()

    def _start_next_segment(self) -> None:
        if self.spec is None or self.cancel_requested:
            return
        if self.target_index >= len(self.targets):
            self._set_state("Completed")
            self.run_completed.emit(str(self.spec.output_dir))
            return
        target = self.targets[self.target_index]
        checkpoint = self.spec.output_dir / checkpoint_name(target)
        final_segment = self.target_index == len(self.targets) - 1
        command = build_segment_command(
            self.spec,
            target_time_myr=target,
            checkpoint_dir=checkpoint,
            resume_checkpoint=self.resume_checkpoint,
            final_segment=final_segment,
        )
        try:
            segment_dir = self.diagnostics.start_segment(self.target_index + 1, [])
        except OSError as exc:
            self._set_state("Error")
            self.run_failed.emit(f"Не удалось подготовить папку диагностики: {exc}")
            return
        command = [
            str(self.spec.project_root / "run_with_diagnostics.py"),
            "--diagnostics-dir", str(segment_dir),
            *(["--optimize-assignment"] if self.spec.assignment_optimized else []),
            *command,
        ]
        self.diagnostics.argv = [sys.executable, *command]
        self._segment_identity += 1
        self._active_segment_index = self.target_index + 1
        self._termination_sent = False
        self.active_checkpoint = checkpoint
        environment = simulation_environment(self.spec)
        environment.insert("PYTHONIOENCODING", "utf-8")
        environment.insert("MOON_DIAGNOSTICS_SESSION_TOKEN", self.diagnostics.session_token)
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(self.spec.project_root))
        self.process.setProgram(sys.executable)
        self.process.setArguments(command)
        self.log_line.emit(
            f"Starting segment {self.target_index + 1}/{len(self.targets)} to t={target:g} Myr"
        )
        self.log_line.emit("$ " + " ".join([sys.executable, *command]))
        self.timing.start_segment(target, monotonic())
        self._set_state("Running")
        self.process.start()

    def request_pause(self) -> None:
        if self.state != "Running":
            return
        self.pause_requested = True
        self._set_state("Pausing")
        self.log_line.emit("Pause requested; the current checkpoint segment will finish safely.")

    def resume(self) -> None:
        if self.state != "Paused":
            return
        self.pause_requested = False
        self._start_next_segment()

    def stop_now(self) -> None:
        if self.state == "Paused":
            self.cancel_requested = True
            self._set_state("Stopped")
            self.request_diagnostics("user_stop_paused")
            self.log_line.emit("Paused run stopped; select its completed checkpoint to start with new settings.")
            return
        if not self.is_active() or self.cancel_requested:
            return
        self.cancel_requested = True
        self._stop_started_at = monotonic()
        self._set_state("Stopping")
        self.log_line.emit(
            "Остановка: сохраняю диагностику; ожидание дампа — до 1,5 с. "
            "Последний завершённый checkpoint сохраняется."
        )
        self.request_diagnostics("user_stop")
        self._poll_diagnostics()

    def _process_started(self) -> None:
        self.diagnostics.launcher_pid = int(self.process.processId())
        self.diagnostics_changed.emit()
        # A stop may be requested while QProcess is still Starting (PID = 0).
        # Its first terminate was then ineffectual; target the now known PID.
        if self.cancel_requested and self._termination_sent:
            self._termination_sent = False
            self._terminate_current(self._segment_identity)

    def _terminate_current(self, identity: int) -> None:
        if identity != self._segment_identity or not self.cancel_requested or self._termination_sent:
            return
        if self.process.state() == QProcess.ProcessState.NotRunning:
            self.timing.stop_segment(monotonic())
            self.diagnostics.finish_segment()
            self._set_state("Stopped")
            return
        self._termination_sent = True
        pid = int(self.process.processId())
        self.log_line.emit(f"Прерываю процесс PID {pid or 'ожидается'}.")
        self.process.terminate()
        if pid:
            QTimer.singleShot(3000, lambda: self._kill_if_running(identity, pid))

    def _kill_if_running(self, identity: int, pid: int) -> None:
        if (
            identity == self._segment_identity
            and self.cancel_requested
            and int(self.process.processId()) == pid
            and self.process.state() != QProcess.ProcessState.NotRunning
        ):
            self.log_line.emit(f"PID {pid} не завершился после terminate; выполняю kill.")
            self.process.kill()

    def _record_diagnostic_log(self, line: str) -> None:
        error = self.diagnostics.record_log(line)
        if error:
            self.diagnostics_notice.emit(error)

    def _diagnostic_context(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "segment": self._active_segment_index,
            "segment_count": len(self.targets),
            "last_completed_time_myr": self.current_time,
            "last_completed_checkpoint": self.resume_checkpoint,
            "active_checkpoint_incomplete_until_success": self.active_checkpoint,
            "pause_requested": self.pause_requested,
            "cancel_requested": self.cancel_requested,
            "worker_running": self.process.state() != QProcess.ProcessState.NotRunning,
        }

    def request_diagnostics(self, reason: str = "user_snapshot") -> None:
        """Save GUI evidence first; ask the worker without blocking the event loop."""
        existing_id = self._pending_dump["request_id"] if self._pending_dump else None
        try:
            request_id, report = self.diagnostics.save_report(reason, self._diagnostic_context(), existing_id)
        except (OSError, ValueError, RuntimeError) as exc:
            message = f"Не удалось сохранить диагностику GUI: {exc}"
            self.diagnostics.last_error = message
            self.log_line.emit(message)
            self.diagnostics_notice.emit(message)
            return
        self.log_line.emit(f"Отчёт GUI сохранён: {report}")
        self.diagnostics_notice.emit(f"Сохранён отчёт GUI: {report}")
        if self._pending_dump:
            return
        if self.process.state() == QProcess.ProcessState.NotRunning:
            self.diagnostics_changed.emit()
            return
        try:
            self.diagnostics.request_worker_dump(request_id, reason)
        except OSError as exc:
            message = f"Отчёт GUI сохранён, запрос дампа не записан: {exc}"
            self.diagnostics.last_error = message
            self.log_line.emit(message)
            self.diagnostics_notice.emit(message)
            return
        self._pending_dump = {
            "request_id": request_id, "report": report,
            "identity": self._segment_identity, "started_at": monotonic(),
        }
        self.diagnostics_notice.emit(f"Отчёт GUI сохранён; ожидаю дамп процесса: {report}")
        self.diagnostics_changed.emit()

    def _finish_dump(self, status: str, response: dict[str, Any] | None = None) -> None:
        pending, self._pending_dump = self._pending_dump, None
        if pending is None:
            return
        complete = bool(response and response.get("report") and response.get("stack") and not response.get("error"))
        if response is not None and not complete:
            status = "partial_or_failed"
        try:
            self.diagnostics.finish_report(pending["report"], status, response)
        except (OSError, ValueError) as exc:
            self.diagnostics.last_error = f"Не удалось обновить отчёт GUI: {exc}"
            self.log_line.emit(f"Не удалось обновить отчёт GUI: {exc}")
            self.diagnostics_notice.emit(f"Не удалось обновить отчёт GUI: {exc}")
            return
        if complete:
            message = f"Диагностика сохранена: {pending['report']} · дамп: {response.get('stack', 'см. отчёт')}"
        else:
            detail = str(response.get("error") or "дамп записан не полностью") if response else "процесс не ответил вовремя"
            message = f"Отчёт GUI сохранён: {pending['report']}; {detail}. Автодамп: {self.diagnostics.segment_dir / 'stacks.txt'}"
        self.log_line.emit(message)
        self.diagnostics_notice.emit(message)

    def _poll_diagnostics(self) -> None:
        if self._pending_dump:
            pending = self._pending_dump
            if pending["identity"] != self._segment_identity:
                self._pending_dump = None
            else:
                response = self.diagnostics.read_response(pending["request_id"])
                if response is not None:
                    self._finish_dump("error" if response.get("error") else "received", response)
                elif monotonic() - pending["started_at"] >= 3.0:
                    self._finish_dump("no_response_yet")
        if self.state == "Stopping" and self._stop_started_at is not None:
            if self._pending_dump is None or monotonic() - self._stop_started_at >= 1.5:
                if self._pending_dump is not None:
                    self._finish_dump("stop_timeout")
                self._terminate_current(self._segment_identity)
        self.diagnostics_changed.emit()

    def _read_output(self) -> None:
        for line in self.diagnostics.feed(bytes(self.process.readAllStandardOutput())):
            self.log_line.emit(line)
        self.diagnostics_changed.emit()

    def _process_finished(self, exit_code: int, status: QProcess.ExitStatus) -> None:
        self._read_output()
        for line in self.diagnostics.feed(final=True):
            self.log_line.emit(line)
        self.diagnostics.finish_segment()
        if self._pending_dump:
            response = self.diagnostics.read_response(self._pending_dump["request_id"])
            self._finish_dump("received" if response and not response.get("error") else "worker_exited", response)
        if self.cancel_requested:
            self.timing.stop_segment(monotonic())
            self._set_state("Stopped")
            return
        if exit_code != 0 or status == QProcess.ExitStatus.CrashExit:
            self.timing.stop_segment(monotonic())
            message = (
                f"Simulation segment exited with code {exit_code} ({status.name}). "
                "Подробности ошибки — во вкладке «Журнал»."
            )
            self._set_state("Error")
            self.request_diagnostics("worker_failure")
            self.run_failed.emit(message)
            return
        if self.spec is None or self.active_checkpoint is None:
            return
        self.timing.finish_segment(monotonic())
        self.current_time = self.targets[self.target_index]
        self.resume_checkpoint = self.active_checkpoint
        self.target_index += 1
        self.progress_changed.emit(self.current_time, self.spec.end_time_myr)
        self.segment_completed.emit(self.current_time, str(self.active_checkpoint))
        if self.pause_requested and self.target_index < len(self.targets):
            self._set_state("Paused")
            self.log_line.emit(f"Paused safely at t={self.current_time:g} Myr.")
            return
        self._start_next_segment()

    def _process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.timing.stop_segment(monotonic())
            self.diagnostics.finish_segment()
            if self._pending_dump:
                self._finish_dump("failed_to_start")
            if self.cancel_requested:
                self._set_state("Stopped")
                return
            message = "The Python simulation process could not be started."
            self._set_state("Error")
            self.request_diagnostics("failed_to_start")
            self.run_failed.emit(message)


class PathField(QWidget):
    def __init__(self, text: str, *, directory: bool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.directory = directory
        self.edit = QLineEdit(text)
        self.button = QPushButton("Обзор…")
        self.button.setObjectName("secondaryButton")
        self.button.clicked.connect(self._browse)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)

    def path(self) -> Path:
        return Path(self.edit.text().strip()).expanduser()

    def set_path(self, path: Path) -> None:
        self.edit.setText(str(path))

    def _browse(self) -> None:
        start = str(self.path().parent if not self.directory else self.path())
        if self.directory:
            selected = QFileDialog.getExistingDirectory(self, "Выберите папку", start)
        else:
            selected, _ = QFileDialog.getOpenFileName(
                self, "Выберите YAML-конфигурацию", start, "YAML (*.yaml *.yml);;Все файлы (*)"
            )
        if selected:
            self.edit.setText(selected)


class MoonWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        app = QApplication.instance()
        if app is not None:
            install_application_font(app)
        self.setWindowTitle("Moon Tectonics — лаборатория v0.31")
        self.resize(1540, 930)
        self.setMinimumSize(1120, 720)
        self.controller = SimulationController(self)
        self.controller.log_line.connect(self._append_log)
        self.controller.state_changed.connect(self._state_changed)
        self.controller.progress_changed.connect(self._progress_changed)
        self.controller.segment_completed.connect(self._segment_completed)
        self.controller.run_completed.connect(self._run_completed)
        self.controller.run_failed.connect(self._run_failed)
        self.current_artifact: Path | None = None
        self.current_movie: QMovie | None = None
        self._close_pending = False
        self._build_ui()
        self.controller.diagnostics_changed.connect(self._refresh_diagnostics)
        self.controller.diagnostics_notice.connect(self.diagnostics_notice.setText)
        self._apply_style()
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._refresh_results)
        self.refresh_timer.timeout.connect(self._refresh_eta)
        self.refresh_timer.start(2000)
        self._resolution_changed()
        self._state_changed("Idle")

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 14, 16, 14)
        root_layout.setSpacing(10)

        title_row = QHBoxLayout()
        title = QLabel("Лаборатория тектоники спутника")
        title.setObjectName("title")
        subtitle = QLabel("v0.31 · экспериментальная CPU/GPU-ветка · отдельная рабочая папка")
        subtitle.setObjectName("subtitle")
        title_box = QVBoxLayout()
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        title_row.addLayout(title_box)
        title_row.addStretch(1)
        self.state_badge = QLabel("Ожидание")
        self.state_badge.setObjectName("stateBadge")
        self.state_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_row.addWidget(self.state_badge)
        root_layout.addLayout(title_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat("Прогон ещё не запущен")
        root_layout.addWidget(self.progress)

        self.eta_label = QLabel()
        self.eta_label.setObjectName("hint")
        self.eta_label.setWordWrap(True)
        self.eta_label.setToolTip(
            "ETA по последним пяти завершённым сегментам, начиная со второго. "
            "Учитываются запуск процесса, расчёт, кадры и checkpoint. "
            "Паузы исключены; при продолжении отсчёт начинается с выбранного checkpoint. "
            "Итоговая сборка GIF может потребовать дополнительного времени."
        )
        root_layout.addWidget(self.eta_label)
        root_layout.addWidget(self._diagnostics_panel())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        settings = self._settings_panel()
        preview = self._preview_panel()
        information = self._information_panel()
        settings.setMinimumWidth(460)
        preview.setMinimumWidth(580)
        information.setMinimumWidth(340)
        splitter.addWidget(settings)
        splitter.addWidget(preview)
        splitter.addWidget(information)
        splitter.setSizes([480, 650, 360])
        splitter.setStretchFactor(1, 1)
        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(root)

    def _diagnostics_panel(self) -> QWidget:
        group = QGroupBox("Сейчас выполняется")
        layout = QGridLayout(group)
        layout.setVerticalSpacing(3)
        self.stage_label = QLabel("Этап появится после запуска")
        self.stage_label.setTextFormat(Qt.TextFormat.PlainText)
        self.stage_label.setWordWrap(True)
        self.stage_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.stage_details = QLabel()
        self.stage_details.setTextFormat(Qt.TextFormat.PlainText)
        self.stage_details.setWordWrap(True)
        self.stage_details.setObjectName("hint")
        self.stage_clock = QLabel()
        self.stage_clock.setWordWrap(True)
        self.stage_clock.setObjectName("hint")
        self.save_diagnostics_button = QPushButton("Сохранить диагностику")
        self.save_diagnostics_button.setToolTip(
            "Сохраняет этап, параметры, последние сообщения и стеки потоков без остановки расчёта."
        )
        self.save_diagnostics_button.clicked.connect(lambda: self.controller.request_diagnostics())
        self.open_diagnostics_button = QPushButton("Открыть диагностику")
        self.open_diagnostics_button.setObjectName("secondaryButton")
        self.open_diagnostics_button.clicked.connect(self._open_diagnostics_folder)
        self.diagnostics_recent = QPlainTextEdit()
        self.diagnostics_recent.setReadOnly(True)
        self.diagnostics_recent.setMaximumBlockCount(8)
        self.diagnostics_recent.setFixedHeight(58)
        self.diagnostics_recent.setPlaceholderText("Последние этапы и сообщения процесса")
        self.diagnostics_notice = QLabel()
        self.diagnostics_notice.setTextFormat(Qt.TextFormat.PlainText)
        self.diagnostics_notice.setWordWrap(True)
        self.diagnostics_notice.setObjectName("hint")
        self.diagnostics_notice.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.stage_label, 0, 0)
        layout.addWidget(self.save_diagnostics_button, 0, 1)
        layout.addWidget(self.stage_details, 1, 0)
        layout.addWidget(self.open_diagnostics_button, 1, 1)
        layout.addWidget(self.stage_clock, 2, 0, 1, 2)
        layout.addWidget(self.diagnostics_recent, 3, 0, 1, 2)
        layout.addWidget(self.diagnostics_notice, 4, 0, 1, 2)
        layout.setColumnStretch(0, 1)
        return group

    def _refresh_diagnostics(self) -> None:
        monitor = self.controller.diagnostics
        available = monitor.segment_dir is not None
        self.save_diagnostics_button.setEnabled(available and self.controller._pending_dump is None)
        self.open_diagnostics_button.setEnabled(monitor.session_dir is not None)
        if not available:
            self.stage_label.setText("Этап появится после запуска")
            self.stage_details.setText("")
            self.stage_clock.setText("Сигнал процесса подтверждает связь; продвижение видно по этапам и счётчикам.")
            return
        stage = monitor.stage
        stage_path = monitor.stage_path
        if self.controller.state == "Error" and monitor.failure:
            stage = monitor.failure.get("stage") or stage
            stage_path = monitor.failure.get("stage_path") or stage_path
        stage_name = " → ".join(stage_path) or str(stage.get("name", "Запуск процесса"))
        endings = {"Paused": " · безопасная пауза", "Completed": " · расчёт завершён",
                   "Stopped": " · остановлено", "Error": " · ошибка"}
        suffix = endings.get(self.controller.state, "")
        self.stage_label.setText(stage_name + suffix)
        details = stage.get("details", {})
        if isinstance(details, dict):
            labels = {"step": "Шаг", "step_count": "Всего шагов", "time_myr": "t, Myr",
                      "target_time_myr": "Цель, Myr", "dt_myr": "dt, Myr", "function": "Функция",
                      "runner": "Программа", "path": "Файл", "matched": "Назначено ячеек",
                      "rows": "Всего ячеек", "scanned_edges": "Проверено связей",
                      "assignment_phase": "Фаза подбора", "searched_rows": "Просмотрено ячеек",
                      "current_search_rows": "Ячеек в текущем поиске", "frontier_entries": "Связей в очереди",
                      "plate_id": "Плита", "candidates": "Кандидатов на ячейку",
                      "active_workers": "Плит в работе", "worker_summary": "Работники",
                      "source_count": "Исходных ячеек", "used_targets": "Целевых ячеек",
                      "graph_file": "Граф подбора"}
            priority_keys = ("active_workers", "worker_summary", "plate_id", "assignment_phase", "matched", "rows", "scanned_edges",
                             "current_search_rows", "frontier_entries", "searched_rows",
                             "candidates", "source_count", "used_targets", "graph_file")
            visible_details = [(key, details[key]) for key in priority_keys if key in details]
            visible_details.extend((key, value) for key, value in details.items() if key not in priority_keys)
            phase_labels = {"prepare": "Подготовка", "warm_start": "Начальное назначение",
                            "feasibility": "Проверка возможности полного назначения",
                            "augment": "Разрешение конфликтов", "certificate": "Проверка оптимальности",
                            "complete": "Подбор завершён", "expand_candidates": "Расширение списка кандидатов"}
            detail_text = " · ".join(
                f"{labels.get(key, key)}: "
                + (phase_labels.get(str(value), str(value)) if key == "assignment_phase" else str(value))[:180]
                for key, value in visible_details[:12]
            )
        else:
            detail_text = str(details)[:1000]
        self.stage_details.setText(detail_text or "Подробности этапа пока не переданы")
        elapsed = monitor.stage_elapsed()
        if self.controller.state == "Error" and monitor.failure:
            elapsed = float(stage.get("elapsed_seconds", elapsed))
        heartbeat_age = monitor.heartbeat_age()
        if not self.controller.is_active():
            heartbeat = "процесс не активен"
        elif heartbeat_age is None:
            heartbeat = "сигнал процесса ещё не получен"
        else:
            heartbeat = f"сигнал процесса {heartbeat_age:.0f} с назад"
        warning = " · долгий этап" if elapsed >= 30 and self.controller.is_active() else ""
        self.stage_clock.setText(
            f"Этап: {format_duration(elapsed)} · PID расчёта {monitor.worker_pid or 'ожидается'} · {heartbeat}{warning}. "
            "Сигнал подтверждает связь; продвижение видно по этапам и счётчикам."
        )

    def _open_diagnostics_folder(self) -> None:
        path = self.controller.diagnostics.segment_dir or self.controller.diagnostics.session_dir
        if path is not None and path.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def _settings_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(2, 2, 8, 2)

        model_group = QGroupBox("Эксперимент")
        model_form = QFormLayout(model_group)
        self.scenario = QComboBox()
        self.scenario.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.scenario.setMinimumContentsLength(16)
        self.scenario.addItems(
            [
                "Зрелая тектоника — v0.31",
                *(f"{ORIGIN_LABELS_RU[origin]} (в плане)" for origin in SatelliteOrigin),
            ]
        )
        for index in range(1, self.scenario.count()):
            item = self.scenario.model().item(index)
            if item is not None:
                item.setEnabled(False)
        model_form.addRow("Сценарий", self.scenario)
        self.config_field = PathField(str(DEFAULT_CONFIG), directory=False)
        model_form.addRow("Конфигурация", self.config_field)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_field = PathField(
            str(PROJECT_ROOT / "results" / "gui_runs" / f"v031_{stamp}"), directory=True
        )
        model_form.addRow("Результаты", self.output_field)
        self.resume_field = PathField("", directory=True)
        self.resume_field.button.setText("Checkpoint…")
        self.resume_field.button.clicked.disconnect()
        self.resume_field.button.clicked.connect(self._browse_checkpoint)
        model_form.addRow("Продолжить", self.resume_field)
        clear_resume = QPushButton("Очистить выбранный checkpoint")
        clear_resume.setObjectName("secondaryButton")
        clear_resume.clicked.connect(lambda: self.resume_field.edit.clear())
        model_form.addRow("", clear_resume)
        layout.addWidget(model_group)

        numerical_group = QGroupBox("Численная сетка и время")
        numerical_form = QFormLayout(numerical_group)
        self.cpu_mode = QComboBox()
        self.cpu_mode.addItem("CPU — исходный", False)
        self.cpu_mode.addItem("CPU — оптимизированный", True)
        self.cpu_mode.addItem("CPU + GPU — поверхность (CUDA)", "gpu_surface")
        self.cpu_mode.setToolTip(
            "GPU-режим переносит только блок поверхностных процессов на NVIDIA CUDA; "
            "остальная физика и новые оптимизации остаются на CPU. "
            "Требуется установленное GPU-окружение. При ошибке CUDA расчёт остановится "
            "с сообщением в журнале, без скрытого переключения на CPU."
        )
        self.cpu_mode.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cpu_mode.setMinimumContentsLength(16)
        self.cpu_mode.setCurrentIndex(1)
        numerical_form.addRow("Режим расчёта", self.cpu_mode)
        self.gpu_device = QSpinBox()
        self.gpu_device.setRange(0, 31)
        self.gpu_device.setValue(0)
        self.gpu_device.setToolTip(
            "Индекс CUDA-устройства, начиная с 0. Для единственной видеокарты оставьте 0. "
            "Наличие устройства проверяется при запуске расчёта."
        )
        self.gpu_device_label = QLabel("CUDA-устройство")
        numerical_form.addRow(self.gpu_device_label, self.gpu_device)
        self.cpu_workers = QComboBox()
        self.cpu_workers.addItems(["1", "2", "4", "8"])
        self.cpu_workers.setToolTip(
            "Число работников переноса плит и пакетного пространственного поиска вулканических дуг. "
            "Это не общее число потоков всей программы. Даже 1 использует кэш геометрии "
            "и пакетные вычисления; 4 обычно разумный компромисс, а больше не всегда быстрее."
        )
        numerical_form.addRow("Работников расчёта", self.cpu_workers)
        self.render_workers = QComboBox()
        self.render_workers.addItems([str(value) for value in RENDER_WORKER_CHOICES])
        self.render_workers.setCurrentText("4")
        self.render_workers.setToolTip(
            "Отдельные процессы для карт и кадров. 1 — последовательное рисование. "
            "Численный результат и качество изображений не меняются. "
            "Больше процессов требует больше памяти и не всегда быстрее. "
            "Безопасная пауза дождётся всех кадров текущего сегмента."
        )
        numerical_form.addRow("Процессов для карт", self.render_workers)
        self.low_priority = QCheckBox("Уступать CPU другим приложениям")
        self.low_priority.setChecked(True)
        self.low_priority.setToolTip(
            "Пониженный CPU-приоритет расчёта и процессов карт; само окно GUI остаётся обычным. "
            "При нагрузке от других программ расчёт может идти дольше. "
            "Это не ограничение памяти, диска или процента загрузки CPU. "
            "Доступно в оптимизированном режиме."
        )
        numerical_form.addRow("", self.low_priority)
        self.cell_kernels = QCheckBox("Пакетный перенос осадков")
        self.cell_kernels.setChecked(True)
        self.cell_kernels.setToolTip("Экспериментальное CPU-ядро с сохранением порядка сложения и точности float64.")
        numerical_form.addRow("", self.cell_kernels)
        self.assignment_optimized = QCheckBox("Быстрый подбор ячеек")
        self.assignment_optimized.setChecked(True)
        self.assignment_optimized.setToolTip(
            "Ускоряет подбор отдельных целей для ячеек движущихся плит. "
            "Сохраняет взаимно однозначный перенос материала; при равной стоимости "
            "соответствия могут отличаться. Продолжение исходного прогона сохраняется в новую папку."
        )
        numerical_form.addRow("", self.assignment_optimized)
        self.assignment_columns = QCheckBox("Компактный подбор соответствий (CPU)")
        self.assignment_columns.setToolTip(
            "Убирает неиспользуемые столбцы в прежнем алгоритме SciPy. "
            "Доступно при выключенном быстром подборе ячеек; выключено по умолчанию."
        )
        numerical_form.addRow("", self.assignment_columns)
        self.boundary_forces = QCheckBox("Пакетные граничные силы (CPU)")
        self.boundary_forces.setToolTip(
            "Кэширует геометрию границ и пакетно вычисляет силы на CPU. "
            "Независима от GPU и подбора соответствий; выключена по умолчанию."
        )
        numerical_form.addRow("", self.boundary_forces)
        self.cpu_mode.currentIndexChanged.connect(self._refresh_execution_controls)
        self.assignment_optimized.toggled.connect(self._refresh_execution_controls)
        self.subdivisions = QComboBox()
        self.subdivisions.addItems([str(value) for value in SUBDIVISION_CHOICES])
        self.subdivisions.setCurrentText("5")
        self.subdivisions.setToolTip(
            "5 — стандартная сетка; 7/8 — экспериментальная высокая детализация. "
            "Каждый уровень увеличивает число ячеек в 4 раза. "
            "Размер в км — √средней площади, не длина ребра и не гарантия точности. "
            "Изменение сетки требует нового расчёта, не продолжения checkpoint."
        )
        self.subdivisions.currentTextChanged.connect(self._resolution_changed)
        numerical_form.addRow("Subdivision", self.subdivisions)
        self.resolution_label = QLabel()
        self.resolution_label.setWordWrap(True)
        self.resolution_label.setObjectName("hint")
        numerical_form.addRow("", self.resolution_label)
        self.config_field.edit.textChanged.connect(self._resolution_changed)
        self.end_time = QDoubleSpinBox()
        self.end_time.setRange(4.0, 20_000.0)
        self.end_time.setDecimals(1)
        self.end_time.setValue(500.0)
        self.end_time.setSuffix(" Myr")
        numerical_form.addRow("Конечное время", self.end_time)
        self.dt = QDoubleSpinBox()
        self.dt.setRange(0.25, 100.0)
        self.dt.setDecimals(2)
        self.dt.setValue(4.0)
        self.dt.setSuffix(" Myr")
        numerical_form.addRow("Шаг времени", self.dt)
        self.checkpoint_interval = QDoubleSpinBox()
        self.checkpoint_interval.setRange(1.0, 1000.0)
        self.checkpoint_interval.setDecimals(1)
        self.checkpoint_interval.setValue(20.0)
        self.checkpoint_interval.setSuffix(" Myr")
        numerical_form.addRow("Checkpoint", self.checkpoint_interval)
        self.frame_interval = QDoubleSpinBox()
        self.frame_interval.setRange(1.0, 1000.0)
        self.frame_interval.setDecimals(1)
        self.frame_interval.setValue(20.0)
        self.frame_interval.setSuffix(" Myr")
        numerical_form.addRow("Частота кадров", self.frame_interval)
        layout.addWidget(numerical_group)

        output_group = QGroupBox("Вывод")
        output_layout = QVBoxLayout(output_group)
        self.surface_only = QCheckBox("Быстрые кадры: только поверхность")
        self.surface_only.setChecked(False)
        self.finalize = QCheckBox("Итоговые карты, графики и GIF")
        self.finalize.setChecked(True)
        output_layout.addWidget(self.surface_only)
        output_layout.addWidget(self.finalize)
        note = QLabel(
            "Безопасная пауза завершает текущий сегмент. Немедленная остановка "
            "сохраняет только предыдущий готовый checkpoint."
        )
        note.setWordWrap(True)
        note.setObjectName("hint")
        output_layout.addWidget(note)
        layout.addWidget(output_group)

        buttons = QGridLayout()
        self.start_button = QPushButton("Запустить прогон")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self._start_run)
        self.pause_button = QPushButton("Безопасная пауза")
        self.pause_button.clicked.connect(self.controller.request_pause)
        self.resume_button = QPushButton("Продолжить")
        self.resume_button.clicked.connect(self.controller.resume)
        self.stop_button = QPushButton("Остановить сейчас")
        self.stop_button.setObjectName("dangerButton")
        self.stop_button.clicked.connect(self._confirm_stop)
        buttons.addWidget(self.start_button, 0, 0, 1, 2)
        buttons.addWidget(self.pause_button, 1, 0)
        buttons.addWidget(self.resume_button, 1, 1)
        buttons.addWidget(self.stop_button, 2, 0, 1, 2)
        layout.addLayout(buttons)
        layout.addStretch(1)
        scroll.setWidget(panel)
        return scroll

    def _preview_panel(self) -> QWidget:
        group = QGroupBox("Живая карта / выбранный результат")
        layout = QVBoxLayout(group)
        toolbar = QHBoxLayout()
        self.preview_name = QLabel("Ожидание первого кадра")
        self.preview_name.setObjectName("hint")
        latest_button = QPushButton("Показать последний")
        latest_button.setObjectName("secondaryButton")
        latest_button.clicked.connect(self._show_latest)
        folder_button = QPushButton("Открыть папку")
        folder_button.setObjectName("secondaryButton")
        folder_button.clicked.connect(self._open_output_folder)
        toolbar.addWidget(self.preview_name, 1)
        toolbar.addWidget(latest_button)
        toolbar.addWidget(folder_button)
        layout.addLayout(toolbar)
        self.preview = QLabel("Кадры и GIF появятся здесь во время прогона.")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(560, 430)
        self.preview.setObjectName("preview")
        layout.addWidget(self.preview, 1)
        return group

    def _information_panel(self) -> QWidget:
        tabs = QTabWidget()
        metrics_page = QWidget()
        metrics_layout = QVBoxLayout(metrics_page)
        self.metrics = QTableWidget(0, 2)
        self.metrics.setHorizontalHeaderLabels(["Показатель", "Значение"])
        self.metrics.horizontalHeader().setStretchLastSection(True)
        self.metrics.verticalHeader().setVisible(False)
        self.metrics.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        metrics_layout.addWidget(self.metrics)
        tabs.addTab(metrics_page, "Метрики")

        artifacts_page = QWidget()
        artifacts_layout = QVBoxLayout(artifacts_page)
        artifact_note = QLabel("Двойной щелчок показывает PNG или GIF.")
        artifact_note.setObjectName("hint")
        self.artifacts = QListWidget()
        self.artifacts.itemDoubleClicked.connect(self._artifact_activated)
        artifacts_layout.addWidget(artifact_note)
        artifacts_layout.addWidget(self.artifacts, 1)
        tabs.addTab(artifacts_page, "Файлы")

        log_page = QWidget()
        log_layout = QVBoxLayout(log_page)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        log_layout.addWidget(self.log)
        tabs.addTab(log_page, "Журнал")
        return tabs

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #11161d; color: #d9e2ec; font-size: 13px; }
            QLabel#title { font-size: 25px; font-weight: 700; color: #f3f7fb; }
            QLabel#subtitle, QLabel#hint { color: #8fa3b8; }
            QLabel#stateBadge { background: #253244; color: #dbeafe; border-radius: 12px; padding: 6px 15px; font-weight: 700; }
            QLabel#preview { background: #080b10; border: 1px solid #263343; border-radius: 8px; color: #738396; }
            QGroupBox { border: 1px solid #263343; border-radius: 8px; margin-top: 12px; padding-top: 12px; font-weight: 650; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #d5e7f7; }
            QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QPlainTextEdit, QListWidget, QTableWidget {
                background: #0b1016; border: 1px solid #2a394b; border-radius: 5px; padding: 5px; selection-background-color: #176b87;
            }
            QPushButton { background: #26384b; border: 1px solid #38516c; border-radius: 6px; padding: 7px 10px; }
            QPushButton:hover { background: #304a64; }
            QPushButton:disabled { color: #617083; background: #1a222c; border-color: #27313c; }
            QPushButton#primaryButton { background: #087f8c; border-color: #16a6b5; color: white; font-weight: 700; padding: 10px; }
            QPushButton#primaryButton:hover { background: #0b96a5; }
            QPushButton#secondaryButton { background: #1b2734; }
            QPushButton#dangerButton { background: #563039; border-color: #804653; }
            QProgressBar { background: #0b1016; border: 1px solid #27384a; border-radius: 5px; text-align: center; min-height: 18px; }
            QProgressBar::chunk { background: #118d9a; border-radius: 4px; }
            QTabWidget::pane { border: 1px solid #263343; border-radius: 6px; }
            QTabBar::tab { background: #18222d; padding: 8px 11px; margin-right: 2px; }
            QTabBar::tab:selected { background: #284258; color: white; }
            QHeaderView::section { background: #1a2734; color: #bfd0df; padding: 6px; border: 0; }
            QSplitter::handle { background: #1c2733; width: 4px; }
            """
        )

    def _browse_checkpoint(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "Select checkpoint folder", str(PROJECT_ROOT / "results")
        )
        if not selected:
            return
        checkpoint = Path(selected)
        try:
            time_myr = read_checkpoint_time(checkpoint)
            subdivisions = subdivision_for_cell_count(checkpoint_cell_count(checkpoint))
            if subdivisions not in SUBDIVISION_CHOICES:
                raise ValueError(f"Checkpoint subdivision {subdivisions} is not supported by this GUI")
        except Exception as exc:
            QMessageBox.warning(self, "Invalid checkpoint", str(exc))
            return
        self.resume_field.set_path(checkpoint)
        source_output = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
        if source_output.resolve().is_relative_to((PROJECT_ROOT / "results").resolve()):
            self.output_field.set_path(source_output)
        else:
            # Never default an experiment to writing into the stable run or backup.
            self.output_field.set_path(PROJECT_ROOT / "results" / "gui_runs" / datetime.now().strftime("cpu_resume_%Y%m%d_%H%M%S"))
            self._append_log("External checkpoint is read-only; new frames will start in this workspace. Copy the run here first to retain old animation frames.")
        saved_config = source_output / "gui_runtime_config.yaml"
        if saved_config.is_file():
            self.config_field.set_path(saved_config)
        self.subdivisions.setCurrentText(str(subdivisions))
        self._append_log(f"Selected checkpoint at t={time_myr:g} Myr (sub-{subdivisions}).")

    def _make_spec(self) -> RunSpec:
        resume_text = self.resume_field.edit.text().strip()
        gpu_surface = self.cpu_mode.currentData() == "gpu_surface"
        optimized = self.cpu_mode.currentData() is True or gpu_surface
        return RunSpec(
            project_root=PROJECT_ROOT,
            source_config=self.config_field.path(),
            output_dir=self.output_field.path(),
            subdivisions=int(self.subdivisions.currentText()),
            end_time_myr=self.end_time.value(),
            dt_myr=self.dt.value(),
            checkpoint_interval_myr=self.checkpoint_interval.value(),
            frame_interval_myr=self.frame_interval.value(),
            surface_only_frames=self.surface_only.isChecked(),
            finalize=self.finalize.isChecked(),
            resume_checkpoint=Path(resume_text) if resume_text else None,
            cpu_optimized=optimized,
            cpu_workers=int(self.cpu_workers.currentText()),
            render_workers=int(self.render_workers.currentText()) if optimized else 1,
            cell_kernels=self.cell_kernels.isChecked() if optimized else False,
            process_priority="below_normal" if optimized and self.low_priority.isChecked() else "normal",
            gpu_surface=gpu_surface,
            gpu_device=self.gpu_device.value() if gpu_surface else 0,
            assignment_columns=optimized and not self.assignment_optimized.isChecked() and self.assignment_columns.isChecked(),
            boundary_forces=optimized and self.boundary_forces.isChecked(),
            assignment_optimized=self.assignment_optimized.isChecked(),
        )

    def _start_run(self) -> None:
        try:
            spec = self._make_spec().normalized()
            if spec.assignment_optimized and spec.resume_checkpoint is not None:
                source_run = spec.resume_checkpoint.parent
                if source_run.name == "checkpoints":
                    source_run = source_run.parent
                if spec.output_dir == source_run:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    output_dir = PROJECT_ROOT / "results" / "gui_runs" / f"assignment_{stamp}"
                    spec = replace(spec, output_dir=output_dir)
                    self.output_field.set_path(output_dir)
                    self._append_log(f"Продолжение с оптимизацией будет сохранено в новую папку: {output_dir}")
                saved_config = source_run / "gui_runtime_config.yaml"
                if saved_config.is_file():
                    spec = replace(spec, source_config=saved_config)
                    self.config_field.set_path(saved_config)
                    self._append_log(f"Используется сохранённая конфигурация исходного прогона: {saved_config}")
            if spec.resume_checkpoint is None and spec.output_dir.exists():
                existing = list(spec.output_dir.iterdir())
                if existing:
                    raise ValueError(
                        "Fresh runs require an empty output folder. Choose a new folder or select a checkpoint to resume."
                    )
            if spec.subdivisions >= 7:
                spec.validate()
                answer = QMessageBox.question(
                    self,
                    "Высокая детализация сетки",
                    self.resolution_label.text()
                    + "\n\nДлительный расчёт и построение карт могут потребовать много RAM и времени. "
                    "Тесты оптимизации остаются на subdivision 5. Запустить выбранную сетку?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
            self.controller.start(spec)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot start simulation", str(exc))

    def _confirm_stop(self) -> None:
        paused = self.controller.state == "Paused"
        answer = QMessageBox.question(
            self,
            "Остановить расчёт и сохранить диагностику?",
            ("Прогон уже на безопасной паузе. Можно будет выбрать его чекпойнт и запустить продолжение с новыми настройками."
             if paused else "Будут сохранены текущий этап и доступный дамп, затем сегмент будет прерван. "
             "Предыдущий готовый чекпойнт останется доступен."),
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.controller.stop_now()

    def _state_changed(self, state: str) -> None:
        translated = {
            "Idle": "Ожидание",
            "Preparing": "Подготовка",
            "Running": "Расчёт",
            "Pausing": "Завершение сегмента",
            "Paused": "Пауза",
            "Stopping": "Остановка",
            "Stopped": "Остановлено",
            "Completed": "Готово",
            "Error": "Ошибка",
        }
        self.state_badge.setText(translated.get(state, state))
        self.start_button.setEnabled(state in {"Idle", "Completed", "Error", "Stopped"})
        self.pause_button.setEnabled(state == "Running")
        self.resume_button.setEnabled(state == "Paused")
        self.stop_button.setEnabled(state in {"Running", "Pausing", "Preparing", "Paused"})
        self._refresh_execution_controls()
        self._refresh_eta()
        self._refresh_diagnostics()
        if self._close_pending and not self.controller.is_active():
            QTimer.singleShot(0, self.close)

    def _refresh_execution_controls(self) -> None:
        # RunSpec is captured at Start. Do not imply that editing a selector
        # can reconfigure an existing worker pool, including across safe pause.
        locked = self.controller.is_active() or self.controller.state == "Paused"
        self.cpu_mode.setEnabled(not locked)
        gpu_surface = self.cpu_mode.currentData() == "gpu_surface"
        enabled = not locked and (self.cpu_mode.currentData() is True or gpu_surface)
        for control in (
            self.cpu_workers, self.render_workers, self.low_priority, self.cell_kernels,
            self.boundary_forces,
        ):
            control.setEnabled(enabled)
        self.assignment_optimized.setEnabled(not locked)
        self.assignment_columns.setEnabled(enabled and not self.assignment_optimized.isChecked())
        self.gpu_device.setVisible(gpu_surface)
        self.gpu_device_label.setVisible(gpu_surface)
        self.gpu_device.setEnabled(not locked and gpu_surface)

    def _progress_changed(self, current: float, end: float) -> None:
        value = 0 if end <= 0 else int(max(0.0, min(1.0, current / end)) * 1000)
        self.progress.setValue(value)
        self.progress.setFormat(f"t = {current:g} / {end:g} Myr   ·   {100 * current / end:.1f}%")
        self._refresh_eta()

    def _refresh_eta(self) -> None:
        state = self.controller.state
        if state == "Idle":
            self.eta_label.setText("ETA появится после двух завершённых сегментов.")
            return
        estimate = self.controller.timing.estimate(monotonic())
        elapsed = f"Прошло без пауз: {format_duration(estimate.elapsed_seconds)}"
        if state == "Completed":
            detail = "Готово"
        elif state in {"Stopped", "Stopping", "Error"}:
            detail = "ETA недоступно: расчёт остановлен или прерван"
        elif estimate.segment_overdue:
            detail = "ETA уточняется: текущий сегмент длится дольше прогноза"
        elif estimate.remaining_seconds is None:
            detail = f"ETA: собираю статистику ({estimate.sample_count}/2 сегмента)"
        else:
            prefix = "После возобновления" if state == "Paused" else "ETA расчёта"
            detail = f"{prefix}: ≈ {format_duration(estimate.remaining_seconds)}"
            if self.controller.spec is not None and self.controller.spec.finalize:
                detail += " + итоговая сборка карт/GIF"
        if state == "Paused" and estimate.remaining_seconds is None:
            detail = "Пауза · " + detail
        self.eta_label.setText(f"{elapsed} · {detail}")

    def _segment_completed(self, time_myr: float, checkpoint: str) -> None:
        self._append_log(f"Safe checkpoint completed at t={time_myr:g} Myr: {checkpoint}")
        self._refresh_results()
        self._show_latest()

    def _run_completed(self, output: str) -> None:
        self._append_log(f"Run complete: {output}")
        self._refresh_results()
        self._show_latest()
        if not self._close_pending:
            QMessageBox.information(self, "Run complete", f"All requested segments completed.\n\n{output}")

    def _run_failed(self, message: str) -> None:
        self._append_log("ERROR: " + message)
        if not self._close_pending:
            QMessageBox.critical(self, "Simulation failed", message)

    def _append_log(self, line: str) -> None:
        self.log.appendPlainText(line)
        self.diagnostics_recent.appendPlainText(line)
        scrollbar = self.log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _resolution_changed(self) -> None:
        subdivisions = int(self.subdivisions.currentText())
        try:
            with self.config_field.path().open("r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            note = resolution_note(subdivisions, float(config["moon"]["radius_km"]))
        except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError):
            note = resolution_note(subdivisions)
        self.resolution_label.setText(note)

    def _refresh_results(self) -> None:
        output = self.output_field.path()
        metrics = load_run_metrics(output)
        self.metrics.setRowCount(len(metrics))
        for row, (key, value) in enumerate(metrics.items()):
            self.metrics.setItem(row, 0, QTableWidgetItem(str(key)))
            if isinstance(value, float):
                display = f"{value:.6g}"
            else:
                display = "—" if value is None else str(value)
            self.metrics.setItem(row, 1, QTableWidgetItem(display))

        selected = None
        if self.artifacts.currentItem() is not None:
            selected = self.artifacts.currentItem().data(Qt.ItemDataRole.UserRole)
        artifacts = discover_artifacts(output)
        current_paths = [
            self.artifacts.item(index).data(Qt.ItemDataRole.UserRole)
            for index in range(self.artifacts.count())
        ]
        artifact_paths = [str(path) for path in artifacts]
        if current_paths != artifact_paths:
            self.artifacts.clear()
            for path in artifacts:
                item = QListWidgetItem(str(path.relative_to(output)))
                item.setToolTip(str(path))
                item.setData(Qt.ItemDataRole.UserRole, str(path))
                self.artifacts.addItem(item)
                if selected == str(path):
                    self.artifacts.setCurrentItem(item)

        if self.current_artifact is None:
            latest = preferred_preview(output)
            if latest is not None:
                self._display_artifact(latest)

    def _artifact_activated(self, item: QListWidgetItem) -> None:
        path = Path(item.data(Qt.ItemDataRole.UserRole))
        self._display_artifact(path)

    def _show_latest(self) -> None:
        path = preferred_preview(self.output_field.path())
        if path is not None:
            self._display_artifact(path)

    def _display_artifact(self, path: Path) -> None:
        if not path.is_file():
            return
        self.current_artifact = path
        self.preview_name.setText(path.name)
        if self.current_movie is not None:
            self.current_movie.stop()
            self.current_movie.deleteLater()
            self.current_movie = None
        if path.suffix.lower() == ".gif":
            movie = QMovie(str(path))
            movie.setCacheMode(QMovie.CacheMode.CacheAll)
            size = self.preview.size() - QSize(20, 20)
            movie.setScaledSize(size)
            self.preview.setMovie(movie)
            self.current_movie = movie
            movie.start()
        else:
            pixmap = QPixmap(str(path))
            if not pixmap.isNull():
                self.preview.setPixmap(
                    pixmap.scaled(
                        self.preview.size() - QSize(20, 20),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )

    def _open_output_folder(self) -> None:
        path = self.output_field.path()
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        if self.current_artifact is not None and self.current_artifact.suffix.lower() != ".gif":
            QTimer.singleShot(100, lambda: self._display_artifact(self.current_artifact))

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.controller.is_active():
            if self._close_pending:
                event.ignore()
                return
            answer = QMessageBox.question(
                self,
                "Simulation is running",
                "Сохранить диагностику, остановить сегмент и закрыть окно? "
                "Последний завершённый checkpoint останется доступен.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            if not self.controller.is_active():
                event.accept()
                return
            self._close_pending = True
            self.controller.stop_now()
            event.ignore()
            return
        event.accept()


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    install_application_font(app)
    app.setApplicationName("Moon Tectonics Laboratory")
    app.setOrganizationName("Habitable Moon Project")
    window = MoonWindow()
    window.show()
    return app.exec()
