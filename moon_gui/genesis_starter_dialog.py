"""Independent, cancellable launch of the coarse molten-start experiment."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout,
                               QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QSizePolicy,
                               QSpinBox, QVBoxLayout)

from tectonics.simulation import load_config

ROOT = Path(__file__).resolve().parent.parent


class GenesisStarterDialog(QDialog):
    continuation_ready = Signal(str)

    def __init__(self, parent=None, *, project_root: Path = ROOT):
        super().__init__(parent)
        self.root = project_root
        self.config_path = self.root / "configs" / "genesis_moon.yaml"
        self.output_path: Path | None = None
        self._pixmap = QPixmap()
        self._generation = 0
        self._stopping = False
        self._closing = False
        self.continuation_path: Path | None = None
        self._published_continuation: Path | None = None
        self._output_tail = ""
        self._continuation_started = False
        self.setWindowTitle("Стартер генезиса — кандидат первых плит")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.resize(1120, 850)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
        config = load_config(self.config_path)
        layout = QVBoxLayout(self)
        introduction = QLabel(
            "Генезис A: остывание после расплавленного старта. Образование в диске ещё не рассчитывается. "
            "Единая оболочка затвердевает и повреждается; океан конденсируется по мере охлаждения. "
            "После первого разделения можно автоматически продолжить расчёт зрелым движком."
        )
        introduction.setWordWrap(True)
        layout.addWidget(introduction)
        form = QFormLayout()
        self.duration = self._number(.0001, 1000, 20, 4)
        self.step = self._number(.0001, 20, 1, 4)
        self.step.setToolTip("Интервал сохранения. Нагрузка внутри рассчитывается более короткими шагами.")
        self.subdivisions = QComboBox()
        for level in (2, 3, 4):
            self.subdivisions.addItem(f"{20 * 4 ** level:,} ячеек".replace(",", " "), level)
        self.subdivisions.setCurrentIndex(2)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(20260927)
        self.seed.setToolTip("Воспроизводимые гладкие неоднородности нагрузки и прочности; не готовые границы плит.")
        traction = config.get("genesis_shell", {}).get("convective_traction_pa", 20000.) / 1e6
        self.convective_traction = self._number(0, 1, traction, 4)
        self.convective_traction.setToolTip(
            "0,02 МПа = 20 кПа — выбранное условие эксперимента. Нагрузка пока не вычисляется "
            "из скорости конвекции или орбиты спутника. Приливные напряжения рассчитываются отдельно."
        )
        self.traction_hint = QLabel("Нагрузка задаётся вручную. Исходные 0,02 МПа — экспериментальное допущение.")
        self.traction_hint.setWordWrap(True)
        self.tides = QCheckBox("Приливы и орбитальный нагрев")
        self.tides.setChecked(True)
        self.water_weakening = QCheckBox("Ослабление при доступе жидкой воды")
        self.water_weakening.setChecked(True)
        self.intact_control = QCheckBox("Контроль: отключить механические нагрузки")
        self.intact_control.setToolTip("Остывание и орбитальный нагрев сохраняются; механические воздействия мантии, приливов и неравномерного охлаждения отключены.")
        self.intact_control.toggled.connect(self._update_controls)
        self.continue_mature = QCheckBox("После разделения продолжить в зрелом движке")
        self.continue_mature.setToolTip(
            "Включите перед запуском: после первого разделения зрелый движок автоматически "
            "получит те же области, возраст, повреждение и тепловое состояние и рассчитает указанное время. "
            "Готовый результат передаётся в основное окно для дальнейшего продолжения. Без разделения этот этап не запускается."
        )
        self.continue_mature.toggled.connect(self._update_controls)
        self.continuation_duration = self._number(.0001, 4500, 10, 4)
        self.continuation_step = self._number(.0001, 20, 1, 4)
        self.continuation_duration.setEnabled(False)
        self.continuation_step.setEnabled(False)
        self.continuation_duration.setToolTip("Время после первого разделения, не возраст от расплавленного старта.")
        self.continuation_step.setToolTip("Шаг зрелой модели. Продолжительность должна делиться на шаг без остатка.")
        for title, widget in (("Конец расчёта после расплавленного старта, млн лет", self.duration),
                              ("Интервал результатов, млн лет", self.step),
                              ("Сферическая сетка", self.subdivisions),
                              ("Seed гладких неоднородностей", self.seed),
                              ("Заданная нагрузка мантии, МПа", self.convective_traction),
                              ("", self.traction_hint),
                              ("", self.tides), ("", self.water_weakening), ("", self.intact_control),
                              ("", self.continue_mature),
                              ("Продолжение после разделения, млн лет", self.continuation_duration),
                              ("Шаг зрелой модели, млн лет", self.continuation_step)):
            form.addRow(title, widget)
        layout.addLayout(form)
        self.transition_hint = QLabel()
        self.transition_hint.setWordWrap(True)
        layout.addWidget(self.transition_hint)
        self._controls = (self.duration, self.step, self.subdivisions, self.seed,
                          self.convective_traction, self.tides, self.water_weakening, self.intact_control,
                          self.continue_mature, self.continuation_duration, self.continuation_step)
        note = QLabel("Разрушение параметризовано на масштабе сетки. Отсутствие разделения — допустимый результат. Состав первичной коры принят мафическим; устойчивость движения плит ещё проверяется. Охлаждение и конденсация продолжаются. При остановке зрелого этапа сохраняется исходный checkpoint; новый записывается в конце этапа.")
        note.setWordWrap(True)
        layout.addWidget(note)
        actions = QHBoxLayout()
        self.start_button = QPushButton("Рассчитать стартер")
        self.start_button.clicked.connect(self.start)
        self.stop_button = QPushButton("Остановить")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        self.folder_button = QPushButton("Папка результатов")
        self.folder_button.setEnabled(False)
        self.folder_button.clicked.connect(self._open_folder)
        self.open_result_button = QPushButton("Открыть графики крупно")
        self.open_result_button.setEnabled(False)
        self.open_result_button.clicked.connect(self._open_result)
        self.continue_button = QPushButton("Перейти к продолжению")
        self.continue_button.setEnabled(False)
        self.continue_button.setToolTip("Закрыть стартер и продолжить этот же мир в основном окне.")
        self.continue_button.clicked.connect(self._go_to_continuation)
        for button in (self.start_button, self.stop_button, self.folder_button,
                       self.open_result_button, self.continue_button):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.status = QLabel("Готов к запуску. Каждый расчёт сохраняется в отдельной папке.")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.preview = QLabel("Карты и история появятся после завершения расчёта")
        self.preview.setMinimumSize(650, 260)
        self.preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.preview, 1)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMaximumHeight(95)
        layout.addWidget(self.log)
        self._update_controls()

    @staticmethod
    def _number(low, high, value, decimals):
        widget = QDoubleSpinBox()
        widget.setDecimals(decimals)
        widget.setRange(low, high)
        widget.setValue(value)
        return widget

    def is_running(self):
        return self.process.state() != QProcess.ProcessState.NotRunning

    def arguments(self, output):
        args = ["-u", str(self.root / "run_genesis_starter.py"), "--config", str(self.config_path),
                "--output", str(output), "--duration-myr", str(self.duration.value()),
                "--step-myr", str(self.step.value()), "--subdivisions", str(self.subdivisions.currentData()),
                "--seed", str(self.seed.value()), "--convective-traction-mpa", str(self.convective_traction.value())]
        if not self.tides.isChecked():
            args.append("--no-tides")
        if not self.water_weakening.isChecked():
            args.append("--no-water-weakening")
        if self.intact_control.isChecked():
            args.extend(["--control", "intact"])
        if self.continue_mature.isChecked():
            args.extend(["--continue-myr", str(self.continuation_duration.value()),
                         "--continuation-step-myr", str(self.continuation_step.value())])
        return args

    def start(self):
        if self.is_running():
            return
        self.output_path = self.root / "results" / "genesis_runs" / datetime.now().strftime("starter_%Y%m%d_%H%M%S_%f")
        self._generation += 1
        self._stopping = False
        self._closing = False
        self.continuation_path = None
        self._published_continuation = None
        self._output_tail = ""
        self._continuation_started = False
        self.log.clear()
        self._pixmap = QPixmap()
        self.preview.clear()
        self.preview.setText("Идёт расчёт…")
        self.folder_button.setEnabled(False)
        self.open_result_button.setEnabled(False)
        self.continue_button.setEnabled(False)
        self._set_busy(True)
        self.status.setText("Расчёт запущен. Последний готовый отсчёт сохраняется в starter_checkpoint.npz.")
        env = QProcessEnvironment.systemEnvironment()
        for key, value in {"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg", "PYTHONPATH": str(self.root)}.items():
            env.insert(key, value)
        self.process.setProcessEnvironment(env)
        self.process.setWorkingDirectory(str(self.root))
        self.process.start(sys.executable, self.arguments(self.output_path))

    def _set_busy(self, busy):
        self.start_button.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
        self.continue_button.setEnabled(not busy and self.continuation_path is not None)
        for widget in self._controls:
            widget.setEnabled(not busy)
        self._update_controls(busy=busy)

    def _update_controls(self, _checked=None, *, busy=None):
        if not hasattr(self, "_controls"):
            return
        busy = self.is_running() if busy is None else busy
        self.convective_traction.setEnabled(not busy and not self.intact_control.isChecked())
        self.continuation_duration.setEnabled(not busy and self.continue_mature.isChecked())
        self.continuation_step.setEnabled(not busy and self.continue_mature.isChecked())
        if hasattr(self, "transition_hint"):
            self.transition_hint.setText(
                "Переход автоматический: после разделения рассчитывается указанное время продолжения. "
                "Затем тот же мир передаётся в основное окно; там можно задать следующий срок расчёта."
                if self.continue_mature.isChecked() else
                "Сейчас выбран только стартер. Для общего прогона включите продолжение выше перед запуском."
            )

    def stop(self):
        if not self.is_running():
            return
        self._stopping = True
        self.stop_button.setEnabled(False)
        self.status.setText("Остановка… Последний готовый checkpoint и параметры сохранятся в папке результатов.")
        generation = self._generation
        self.process.terminate()
        QTimer.singleShot(1500, lambda: self._kill_if_current(generation))

    def _kill_if_current(self, generation):
        if generation == self._generation and self.is_running():
            self.process.kill()

    def _read_output(self):
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self.log.appendPlainText(text.rstrip())
            combined = self._output_tail + text
            self._output_tail = combined[-128:]
            if not self._continuation_started and "GENESIS_STARTER_CONTINUATION_START" in combined:
                self._continuation_started = True
                self.status.setText("Первое разделение получено. Рассчитывается заданное время продолжения; затем этот же мир будет доступен в основном окне.")

    def _finished(self, exit_code, exit_status):
        self._read_output()
        self._set_busy(False)
        output = self.output_path
        self.folder_button.setEnabled(output is not None and output.is_dir())
        if self._stopping:
            self.status.setText("Остановлено. Готовые checkpoints сохранены. Стартер продолжается через run_genesis_starter.py --resume; зрелый этап — через run_genesis_starter_continuation.py --resume из папки готового продолжения.")
        elif exit_code == 0 and exit_status == QProcess.ExitStatus.NormalExit and output is not None:
            try:
                summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
                final = summary["final"]
                if summary.get("candidate_partition"):
                    result = "Получен кандидат начального разделения. Подвижность и устойчивость плит ещё не проверены."
                elif summary.get("status") == "completed":
                    result = "Расчёт дошёл до заданного времени без разделения оболочки."
                else:
                    result = f"Расчёт остановлен: {summary.get('status', 'неизвестная причина')}."
                continuation = summary.get("continuation", {})
                if continuation.get("status") == "completed":
                    self._publish_continuation(output / "continuation")
                elif continuation.get("status") == "skipped_no_partition":
                    result += " Зрелый этап не запускался: разделения нет."
                if continuation.get("status") != "completed":
                    self.status.setText(f"{result} Возраст: {final['time_myr']:.4f} млн лет; областей: {final['domain_count']}; повреждено: {final['damaged_area_fraction']:.2%}.\n{output}")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.status.setText(f"Не удалось прочитать результат: {exc}")
        else:
            self.status.setText("Расчёт завершился с ошибкой. Подробности в журнале ниже.")
        image_path = self._result_image()
        if image_path is not None:
            self._pixmap = QPixmap(str(image_path))
            self.open_result_button.setEnabled(not self._pixmap.isNull())
            self._show_preview()
        if self._closing:
            QTimer.singleShot(0, self.close)

    def _publish_continuation(self, path):
        # The paired young/mature checkpoint is required: the mature archive alone
        # cannot resume the ongoing thermal and fracture calculation.
        from .backend import read_genesis_continuation

        continuation = read_genesis_continuation(path, verify_integrity=True)
        elapsed = continuation.time_myr - continuation.origin_time_myr
        self.status.setText(
            f"Продолжение завершено: рассчитано {elapsed:g} млн лет после разделения. "
            f"Текущий возраст: {continuation.time_myr:.4f} млн лет; областей: {continuation.plate_count}. "
            "Этот мир готов к дальнейшему расчёту в основном окне. Нажмите «Перейти к продолжению»."
        )
        self.continuation_path = continuation.root
        self.continue_button.setEnabled(True)
        if self._published_continuation != continuation.root:
            self._published_continuation = continuation.root
            self.continuation_ready.emit(str(continuation.root))

    def _go_to_continuation(self):
        if not self.is_running() and self.continuation_path is not None:
            self.accept()

    def _error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self._set_busy(False)
            self.status.setText("Не удалось запустить Python: " + self.process.errorString())
            if self._closing:
                QTimer.singleShot(0, self.close)

    def _show_preview(self):
        if not self._pixmap.isNull():
            self.preview.setPixmap(self._pixmap.scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                      Qt.TransformationMode.SmoothTransformation))

    def _open_folder(self):
        if self.output_path is not None and self.output_path.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_path.resolve())))

    def _open_result(self):
        path = self._result_image()
        if path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def _result_image(self):
        if self.output_path is None:
            return None
        for path in (self.output_path / "continuation" / "continuation.png", self.output_path / "genesis_starter.png"):
            if path.is_file():
                return path
        return None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "preview"):
            self._show_preview()

    def closeEvent(self, event):
        if self.is_running():
            self._closing = True
            self.stop()
            event.ignore()
        else:
            event.accept()
            super().closeEvent(event)

    def reject(self):
        if self.is_running():
            self._closing = True
            self.stop()
        else:
            super().reject()
