"""PySide6 desktop application for checkpointed Moon Tectonics runs."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import sys
from time import monotonic
from typing import Any

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
from .timing import RunTiming, format_duration
from execution_policy import RENDER_WORKER_CHOICES


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "canonical_moon.yaml"


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

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
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

    def _set_state(self, value: str) -> None:
        self.state = value
        self.state_changed.emit(value)

    def is_active(self) -> bool:
        return self.state in {"Preparing", "Running", "Pausing", "Stopping"}

    def start(self, spec: RunSpec) -> None:
        if self.is_active():
            raise RuntimeError("A simulation is already active")
        spec = spec.normalized()
        spec.validate()
        self.timing = RunTiming(spec.start_time_myr(), spec.end_time_myr)
        self._set_state("Preparing")
        if spec.resume_checkpoint is not None and spec.runtime_config.is_file():
            runtime_config = spec.runtime_config
        else:
            runtime_config = write_runtime_config(spec)
        write_run_record(spec, runtime_config)
        self.spec = spec
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
        self.progress_changed.emit(self.current_time, spec.end_time_myr)
        self.log_line.emit(
            f"Prepared v0.31 run: t={self.current_time:g} -> {spec.end_time_myr:g} Myr, "
            f"sub-{spec.subdivisions}, {len(self.targets)} checkpoint segment(s)."
        )
        self._start_next_segment()

    def _start_next_segment(self) -> None:
        if self.spec is None:
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
        self.active_checkpoint = checkpoint
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("MPLBACKEND", "Agg")
        old_pythonpath = environment.value("PYTHONPATH")
        environment.insert(
            "PYTHONPATH",
            str(self.spec.project_root)
            if not old_pythonpath
            else str(self.spec.project_root) + os.pathsep + old_pythonpath,
        )
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
            self.log_line.emit("Paused run stopped; select its completed checkpoint to start with new settings.")
            return
        if not self.is_active():
            return
        self.cancel_requested = True
        self._set_state("Stopping")
        self.log_line.emit("Stopping the active segment; the last completed checkpoint is preserved.")
        process_id = int(self.process.processId())
        self.process.terminate()
        QTimer.singleShot(3000, lambda: self._kill_if_running(process_id))

    def _kill_if_running(self, process_id: int) -> None:
        if (self.process.state() != QProcess.ProcessState.NotRunning
                and int(self.process.processId()) == process_id):
            self.process.kill()

    def _read_output(self) -> None:
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
            if line.strip():
                self.log_line.emit(line)

    def _process_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._read_output()
        if self.cancel_requested:
            self.timing.stop_segment(monotonic())
            self._set_state("Stopped")
            return
        if exit_code != 0:
            self.timing.stop_segment(monotonic())
            message = f"Simulation segment exited with code {exit_code}."
            self._set_state("Error")
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
        if self.cancel_requested:
            return
        if error == QProcess.ProcessError.FailedToStart:
            self.timing.stop_segment(monotonic())
            message = "The Python simulation process could not be started."
            self._set_state("Error")
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
        self._build_ui()
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
        subtitle = QLabel("v0.31 · экспериментальная CPU-ветка · отдельная рабочая папка")
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
        self.cpu_mode.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cpu_mode.setMinimumContentsLength(16)
        self.cpu_mode.setCurrentIndex(1)
        numerical_form.addRow("Режим расчёта", self.cpu_mode)
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
        self.cpu_mode.currentIndexChanged.connect(self._refresh_execution_controls)
        numerical_form.addRow("", self.cell_kernels)
        self.subdivisions = QComboBox()
        self.subdivisions.addItems(["3", "4", "5", "6"])
        self.subdivisions.setCurrentText("5")
        self.subdivisions.currentTextChanged.connect(self._resolution_changed)
        numerical_form.addRow("Subdivision", self.subdivisions)
        self.resolution_label = QLabel()
        self.resolution_label.setWordWrap(True)
        self.resolution_label.setObjectName("hint")
        numerical_form.addRow("", self.resolution_label)
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
            cpu_optimized=bool(self.cpu_mode.currentData()),
            cpu_workers=int(self.cpu_workers.currentText()),
            render_workers=int(self.render_workers.currentText()) if self.cpu_mode.currentData() else 1,
            cell_kernels=self.cell_kernels.isChecked() if self.cpu_mode.currentData() else False,
            process_priority="below_normal" if self.cpu_mode.currentData() and self.low_priority.isChecked() else "normal",
        )

    def _start_run(self) -> None:
        try:
            spec = self._make_spec().normalized()
            if spec.resume_checkpoint is None and spec.output_dir.exists():
                existing = list(spec.output_dir.iterdir())
                if existing:
                    raise ValueError(
                        "Fresh runs require an empty output folder. Choose a new folder or select a checkpoint to resume."
                    )
            self.controller.start(spec)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot start simulation", str(exc))

    def _confirm_stop(self) -> None:
        paused = self.controller.state == "Paused"
        answer = QMessageBox.question(
            self,
            "Остановить прогон?",
            ("Прогон уже на безопасной паузе. Можно будет выбрать его чекпойнт и запустить продолжение с новыми настройками."
             if paused else "Текущий сегмент будет прерван. Предыдущий готовый чекпойнт останется целым."),
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

    def _refresh_execution_controls(self) -> None:
        # RunSpec is captured at Start. Do not imply that editing a selector
        # can reconfigure an existing worker pool, including across safe pause.
        locked = self.controller.is_active() or self.controller.state == "Paused"
        self.cpu_mode.setEnabled(not locked)
        enabled = not locked and bool(self.cpu_mode.currentData())
        for control in (self.cpu_workers, self.render_workers, self.low_priority, self.cell_kernels):
            control.setEnabled(enabled)

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
        QMessageBox.information(self, "Run complete", f"All requested segments completed.\n\n{output}")

    def _run_failed(self, message: str) -> None:
        self._append_log("ERROR: " + message)
        QMessageBox.critical(self, "Simulation failed", message)

    def _append_log(self, line: str) -> None:
        self.log.appendPlainText(line)
        scrollbar = self.log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _resolution_changed(self) -> None:
        self.resolution_label.setText(resolution_note(int(self.subdivisions.currentText())))

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
            answer = QMessageBox.question(
                self,
                "Simulation is running",
                "Stop the active segment and close? The last completed checkpoint remains safe.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.controller.stop_now()
        event.accept()


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    install_application_font(app)
    app.setApplicationName("Moon Tectonics Laboratory")
    app.setOrganizationName("Habitable Moon Project")
    window = MoonWindow()
    window.show()
    return app.exec()
