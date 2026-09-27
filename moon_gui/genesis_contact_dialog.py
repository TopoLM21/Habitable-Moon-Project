"""Cancellable Qt launcher for the separate frozen-state contact experiment."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QMovie, QPixmap
from PySide6.QtWidgets import (QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
                               QPushButton, QComboBox, QSizePolicy, QVBoxLayout)

ROOT = Path(__file__).resolve().parent.parent


class GenesisContactDialog(QDialog):
    def __init__(self, parent=None, *, project_root: Path = ROOT, checkpoint: Path | None = None):
        super().__init__(parent)
        self.root = Path(project_root)
        self.checkpoint_path: Path | None = None
        self.output_path: Path | None = None
        self._contact_output_checkpoint: Path | None = None
        self._run_kind = "contact"
        self._process_output = ""
        self._resume = False
        self._source_elapsed = 0.
        self._pixmap = QPixmap()
        self._movie: QMovie | None = None
        self._closing = self._stopping = False
        self._generation = 0
        self.setWindowTitle("Генезис — контакт разделённых берегов")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.resize(1110, 900)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
        layout = QVBoxLayout(self)
        self.introduction = QLabel(
            "Короткое механическое продолжение сохранённого состояния: раздельные берега, "
            "раскрытие, сжатие и трение. Температура, орбита, повреждение пород и доступ воды "
            "зафиксированы. Остывание здесь не продолжается."
        )
        self.introduction.setWordWrap(True)
        self.introduction.setStyleSheet("QLabel { background: #e8edf5; color: #172638; padding: 9px; border-radius: 5px; }")
        layout.addWidget(self.introduction)
        self.source_path = QLineEdit()
        self.source_path.setReadOnly(True)
        self.source_path.setPlaceholderText("Выберите fault_checkpoint.npz или contact_checkpoint.npz")
        self.select_button = QPushButton("Выбрать состояние…")
        self.select_button.clicked.connect(self.choose_checkpoint)
        row = QHBoxLayout()
        row.addWidget(self.source_path, 1)
        row.addWidget(self.select_button)
        layout.addLayout(row)
        self.source_kind = QLabel("Исходное состояние не выбрано")
        self.source_kind.setWordWrap(True)
        layout.addWidget(self.source_kind)
        form = QFormLayout()
        self.duration = QDoubleSpinBox()
        self.duration.setRange(.01, 1e7)
        self.duration.setDecimals(2)
        self.duration.setValue(1000.)
        self.duration.setToolTip("Время от исходного состояния разломов; при возобновлении это новый конец расчёта.")
        self.step = QDoubleSpinBox()
        self.step.setRange(.01, 1e5)
        self.step.setDecimals(2)
        self.step.setValue(10.)
        self.step.setToolTip("Шаг запроса результата. Решатель автоматически уменьшает внутренний шаг при необходимости.")
        form.addRow("Конец механического продолжения, лет", self.duration)
        form.addRow("Шаг результата, лет", self.step)
        layout.addLayout(form)
        hint = QLabel("Исходно соседние рёбра взаимодействуют при малых скольжениях. Общий поиск столкновений, "
                      "субдукция и образование новой коры пока отсутствуют. Каждый запуск создаёт новую папку.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QHBoxLayout()
        self.start_button = QPushButton("Рассчитать контакт")
        self.start_button.clicked.connect(self.start)
        self.start_button.setEnabled(False)
        self.handoff_button = QPushButton("Проверить переход…")
        self.handoff_button.setToolTip("Проверить готовность контактного состояния к зрелой модели. Проверка не запускает тектонику.")
        self.handoff_button.clicked.connect(self.start_handoff)
        self.handoff_button.setEnabled(False)
        self.coupled_button = QPushButton("Остывание и разломы…")
        self.coupled_button.setToolTip("Продолжить физическое остывание из исходного состояния разломов в отдельном окне.")
        self.coupled_button.clicked.connect(self._open_coupled)
        self.coupled_button.setEnabled((self.root / "run_genesis_coupled.py").is_file())
        self.stop_button = QPushButton("Остановить")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        self.folder_button = QPushButton("Папка результатов")
        self.folder_button.clicked.connect(self._open_folder)
        self.folder_button.setEnabled(False)
        for button in (self.start_button, self.coupled_button, self.handoff_button, self.stop_button, self.folder_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.status = QLabel("Выберите сохранённое состояние разломов.")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.result_view = QComboBox()
        self.result_view.setEnabled(False)
        self.result_view.currentIndexChanged.connect(self._select_preview)
        self.open_result_button = QPushButton("Открыть крупно")
        self.open_result_button.setEnabled(False)
        self.open_result_button.clicked.connect(self._open_result)
        views = QHBoxLayout()
        views.addWidget(self.result_view, 1)
        views.addWidget(self.open_result_button)
        layout.addLayout(views)
        self.preview = QLabel("Карта зазоров и сдвигов появится после завершения расчёта")
        self.preview.setMinimumSize(600, 300)
        self.preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.preview, 1)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMaximumHeight(95)
        layout.addWidget(self.log)
        if checkpoint is not None:
            self.set_checkpoint(checkpoint)

    def is_running(self):
        return self.process.state() != QProcess.ProcessState.NotRunning

    def _open_coupled(self):
        if self.is_running():
            return
        from .genesis_coupled_dialog import GenesisCoupledDialog
        # Frozen-contact states cannot resume physical cooling. Ask for the
        # original fault source when this dialog currently owns such a state.
        source = self.checkpoint_path if not self._resume else None
        dialog = GenesisCoupledDialog(self, project_root=self.root, checkpoint=source)
        if source is None:
            dialog.choose_checkpoint()
        dialog.exec()

    def choose_checkpoint(self):
        if self.is_running():
            return
        selected, _ = QFileDialog.getOpenFileName(
            self, "Состояние генезиса для контакта", str(self.checkpoint_path or self.root/"results"/"genesis_runs"),
            "Сохранённое состояние (*.npz)")
        if selected:
            self.set_checkpoint(Path(selected))

    def set_checkpoint(self, path):
        if self.is_running():
            return False
        try:
            path = Path(path).resolve()
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(str(data["metadata"]))
            version = meta["format"]
            if not isinstance(version, str) or not version.startswith(("genesis-faults-", "genesis-contact-")):
                raise ValueError("требуется состояние разломов или контакта")
            resume = version.startswith("genesis-contact-")
            elapsed = float(meta.get("state", {}).get("elapsed_years", 0.)) if resume else 0.
            if not np.isfinite(elapsed) or elapsed < 0:
                raise ValueError("неверное время сохранённого состояния")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            self.status.setText(f"Не удалось выбрать состояние: {exc}")
            return False
        self.checkpoint_path, self._resume, self._source_elapsed = path, resume, elapsed
        self._contact_output_checkpoint = None
        self.source_path.setText(str(path))
        self.source_path.setToolTip(str(path))
        self.source_kind.setText(f"Возобновление контакта: уже рассчитано {elapsed:g} лет." if resume
                                 else "Состояние разломов: будет создана разорванная сетка.")
        if self.duration.value() <= elapsed:
            self.duration.setValue(min(self.duration.maximum(), elapsed+1000.))
        self.start_button.setEnabled(True)
        self.handoff_button.setEnabled(resume)
        self.status.setText("Готов к запуску. Тепловое и орбитальное состояние будет зафиксировано.")
        return True

    def arguments(self, output):
        if self.checkpoint_path is None:
            raise ValueError("Сначала выберите сохранённое состояние")
        return ["-u", str(self.root/"run_genesis_contact.py"),
                "--resume" if self._resume else "--checkpoint", str(self.checkpoint_path),
                "--output", str(output), "--duration-years", str(self.duration.value()),
                "--step-years", str(self.step.value())]

    @staticmethod
    def _is_contact_checkpoint(path):
        if path is None or not Path(path).is_file():
            return False
        try:
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"]))
            return str(metadata["format"]).startswith("genesis-contact-")
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return False

    def handoff_checkpoint(self):
        for path in (self._contact_output_checkpoint, self.checkpoint_path if self._resume else None):
            if self._is_contact_checkpoint(path):
                return path
        return None

    def handoff_arguments(self, output):
        source = self.handoff_checkpoint()
        if source is None:
            raise ValueError("Для проверки перехода выберите contact_checkpoint.npz или завершите расчёт контакта")
        return ["-u", str(self.root/"run_genesis_handoff.py"), "--checkpoint", str(source),
                "--output", str(output), "--probe-years", "10", "--intervals", "3"]

    def _new_output(self, kind):
        base = self.root/"results"/"genesis_runs"/datetime.now().strftime(kind+"_%Y%m%d_%H%M%S_%f")
        output, suffix = base, 1
        while output.exists():
            output = base.with_name(base.name+f"_{suffix}")
            suffix += 1
        return output

    def start_handoff(self):
        if self.is_running():
            return
        output = self._new_output("handoff")
        try:
            arguments = self.handoff_arguments(output)
        except ValueError as exc:
            self.status.setText(str(exc))
            self.handoff_button.setEnabled(False)
            return
        self.output_path = output
        self._launch(arguments, "handoff")

    def start(self):
        if self.is_running():
            return
        if self.checkpoint_path is None or not self.checkpoint_path.is_file():
            self.status.setText("Сначала выберите существующее сохранённое состояние.")
            return
        if self.duration.value() <= self._source_elapsed:
            self.status.setText("Конец расчёта должен быть позже сохранённого времени контакта.")
            return
        self.output_path = self._new_output("contact")
        self._contact_output_checkpoint = None
        self._launch(self.arguments(self.output_path), "contact")

    def _launch(self, arguments, kind):
        self._run_kind = kind
        self._process_output = ""
        self._generation += 1
        self._closing = self._stopping = False
        self.log.clear()
        self._stop_movie()
        self.result_view.clear()
        self.result_view.setEnabled(False)
        self.open_result_button.setEnabled(False)
        self.folder_button.setEnabled(False)
        self._pixmap = QPixmap()
        self.preview.setText("Идёт проверка готовности перехода…" if kind == "handoff" else "Идёт расчёт контакта…")
        self._set_busy(True)
        self.status.setText("Проверка запущена: три коротких интервала по 10 лет. Зрелая тектоника не запускается."
                            if kind == "handoff" else "Расчёт запущен. Готовые отсчёты сохраняются в contact_checkpoint.npz.")
        env = QProcessEnvironment.systemEnvironment()
        for key, value in (("PYTHONUNBUFFERED", "1"), ("PYTHONIOENCODING", "utf-8"),
                           ("MPLBACKEND", "Agg"), ("PYTHONPATH", str(self.root))):
            env.insert(key, value)
        self.process.setProcessEnvironment(env)
        self.process.setWorkingDirectory(str(self.root))
        self.process.start(sys.executable, arguments)

    def _set_busy(self, busy):
        self.start_button.setEnabled(not busy and self.checkpoint_path is not None)
        self.coupled_button.setEnabled(not busy and (self.root / "run_genesis_coupled.py").is_file())
        self.handoff_button.setEnabled(not busy and self.handoff_checkpoint() is not None)
        self.stop_button.setEnabled(busy)
        for widget in (self.select_button, self.duration, self.step):
            widget.setEnabled(not busy)

    def stop(self):
        if not self.is_running():
            return
        self._stopping = True
        self.stop_button.setEnabled(False)
        self.status.setText("Остановка проверки перехода… Исходное контактное состояние сохранится."
                            if self._run_kind == "handoff" else "Остановка… Последний готовый contact_checkpoint.npz сохранится.")
        generation = self._generation
        self.process.terminate()
        QTimer.singleShot(1500, lambda: self._kill_if_current(generation))

    def _kill_if_current(self, generation):
        if generation == self._generation and self.is_running():
            self.process.kill()

    def _read_output(self):
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self._process_output = (self._process_output+text)[-8192:]
            self.log.appendPlainText(text.rstrip())

    def _finished(self, exit_code, exit_status):
        from visualization.genesis_connectivity import connectivity_text

        self._read_output()
        output = self.output_path
        if self._run_kind == "contact" and output is not None:
            candidate = output/"contact_checkpoint.npz"
            if self._is_contact_checkpoint(candidate):
                self._contact_output_checkpoint = candidate
        self._set_busy(False)
        self.folder_button.setEnabled(output is not None and output.is_dir())
        if self._stopping:
            self.status.setText("Проверка перехода остановлена. Исходное контактное состояние сохранено; проверку можно запустить снова."
                                if self._run_kind == "handoff" else "Расчёт остановлен. Для продолжения выберите сохранённый contact_checkpoint.npz.")
        elif exit_code == 0 and exit_status == QProcess.ExitStatus.NormalExit and output is not None:
            try:
                if self._run_kind == "handoff":
                    result = self._handoff_summary(output)
                else:
                    summary = json.loads((output/"summary.json").read_text(encoding="utf-8"))
                    final = summary["final"]
                    result = (f"Рассчитано {final['elapsed_years']:g} лет. "
                              f"Максимальное раскрытие: {final['max_opening_m']:.3g} м; "
                              f"относительный сдвиг: {final['max_abs_jump_m']:.3g} м.")
                    if note := connectivity_text(final):
                        result += "\n" + note
                    if summary.get("status", "completed") != "completed":
                        result += f" Расчёт остановлен: {summary['status']}."
                self.status.setText(result+"\n"+str(output))
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                self.status.setText(f"Не удалось прочитать результат: {exc}")
        else:
            self.status.setText("Проверка перехода завершилась с ошибкой. Подробности в журнале ниже."
                                if self._run_kind == "handoff" else "Расчёт контакта завершился с ошибкой. Подробности в журнале ниже.")
        self._load_previews(output)
        # The longer readiness summary changes the available image height.
        # Rescale after Qt has applied that layout update.
        QTimer.singleShot(0, self._show_preview)
        if self._closing:
            QTimer.singleShot(0, self.close)

    def _handoff_summary(self, output):
        summary = json.loads((output/"handoff_report.json").read_text(encoding="utf-8"))
        if "HANDOFF_SCREEN_COMPLETE" not in self._process_output or not isinstance(summary.get("handoff_ready"), bool):
            raise ValueError("нет подтверждения завершения проверки")
        result = ("Проверка завершена. Диагностические условия перехода выполнены."
                  if summary["handoff_ready"] else "Проверка завершена. Состояние пока не готово к переходу.")
        blockers = summary.get("blockers", [])
        messages = [str(item.get("message", item.get("reason", item.get("code", ""))))
                    if isinstance(item, dict) else str(item) for item in blockers]
        if messages:
            result += "\n"+" ".join(messages[:3])
        return result+"\nЗрелая тектоника не запускалась. Подробности: handoff_report.md."

    def _error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self._set_busy(False)
            self.status.setText("Не удалось запустить Python: "+self.process.errorString())
            if self._closing:
                QTimer.singleShot(0, self.close)

    def _load_previews(self, output):
        self._stop_movie()
        self.result_view.blockSignals(True)
        self.result_view.clear()
        if output is not None:
            previews = (("Готовность перехода", "handoff_assessment.png"),) if self._run_kind == "handoff" else (
                ("Зазоры, сдвиги и баланс", "genesis_contact.png"), ("Анимация контакта", "genesis_contact.gif"))
            for label, name in previews:
                path = output/name
                if path.is_file():
                    self.result_view.addItem(label, str(path))
        self.result_view.blockSignals(False)
        self.result_view.setEnabled(self.result_view.count() > 0)
        self._select_preview()

    def _select_preview(self, _index=None):
        self._stop_movie()
        self._pixmap = QPixmap()
        self.preview.clear()
        path = self.result_view.currentData()
        self.open_result_button.setEnabled(bool(path))
        if not path:
            self.preview.setText("Сохранённые изображения пока недоступны")
        elif Path(path).suffix.lower() == ".gif":
            self._movie = QMovie(path, parent=self)
            if self._movie.isValid():
                self._movie.frameChanged.connect(self._show_movie_frame)
                self._movie.start()
            else:
                self.preview.setText("Не удалось открыть анимацию")
        else:
            self._pixmap = QPixmap(path)
            if self._pixmap.isNull():
                self.preview.setText("Не удалось открыть изображение")
            else:
                self._show_preview()

    def _show_preview(self):
        if not self._pixmap.isNull():
            self.preview.setPixmap(self._pixmap.scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                       Qt.TransformationMode.SmoothTransformation))

    def _show_movie_frame(self, _frame=None):
        if self._movie is not None:
            self._pixmap = self._movie.currentPixmap()
            self._show_preview()

    def _stop_movie(self):
        if self._movie is not None:
            self._movie.stop()
            self._movie.deleteLater()
            self._movie = None

    def _open_result(self):
        path = self.result_view.currentData()
        if path and Path(path).is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).resolve())))

    def _open_folder(self):
        if self.output_path is not None and self.output_path.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_path.resolve())))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "preview"):
            self._show_preview()

    def reject(self):
        self.close()

    def closeEvent(self, event):
        if self.is_running():
            self._closing = True
            self.stop()
            event.ignore()
        else:
            self._stop_movie()
            event.accept()
