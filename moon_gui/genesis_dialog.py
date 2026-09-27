"""Thermal and spatial genesis experiments in a cancellable child process."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QMovie, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout,
                               QLabel, QPlainTextEdit, QPushButton, QSizePolicy, QVBoxLayout)

from tectonics.simulation import load_config

ROOT = Path(__file__).resolve().parent.parent


class GenesisDialog(QDialog):
    def __init__(self, parent=None, *, project_root: Path = ROOT):
        super().__init__(parent)
        self.root = project_root
        self.config_path = project_root / "configs" / "genesis_moon.yaml"
        self.output_path: Path | None = None
        self._pixmap = QPixmap()
        self._movie: QMovie | None = None
        self._closing = False
        self._stopping = False
        self._generation = 0
        self.setWindowTitle("Генезис — остывание, океан и движение оболочки")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.resize(1160, 920)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
        configuration = load_config(self.config_path)
        defaults = configuration["genesis"]
        layout = QVBoxLayout(self)
        introduction = QLabel(
            "Расплавленный старт → океан и твёрдая покрышка → повреждение и движение оболочки. "
            "Приливы и доступ воды меняют ослабление оболочки. "
            "В движущемся материале можно рассчитывать трение и необратимый сдвиг разломных зон; "
            "для сохранённого состояния доступен отдельный короткий расчёт контакта берегов."
        )
        introduction.setWordWrap(True)
        layout.addWidget(introduction)
        form = QFormLayout()
        self.mode = QComboBox()
        if (self.root / "run_genesis_faults.py").is_file():
            self.mode.addItem("Трение и сдвиг разломных зон", "faults")
        self.mode.addItem("Движущаяся оболочка", "mobile")
        self.mode.addItem("Приливы, вода и начало движения", "onset")
        self.mode.addItem("Покрышка и повреждения", "shell")
        self.mode.addItem("Только остывание и океан", "thermal")
        default_mode = ("faults" if (self.root / "run_genesis_faults.py").is_file() else
                        "mobile" if (self.root / "run_genesis_mobile.py").is_file() else
                        "onset" if (self.root / "run_genesis_onset.py").is_file() else
                        "shell" if (self.root / "run_genesis_shell.py").is_file() else "thermal")
        self.mode.setCurrentIndex(self.mode.findData(default_mode))
        self.mode.currentIndexChanged.connect(self._update_mode_controls)
        form.addRow("Эксперимент", self.mode)
        self.temperature = self._number(2000, 4000, defaults["initial_temperature_k"], 1)
        self.stellar_flux = self._number(0, 5000, defaults["stellar_flux_w_m2"], 3)
        self.stellar_flux.setToolTip("Поток на площадку поперёк лучей; модель сама применяет (1 − альбедо) / 4.")
        self.water_volume = self._number(0, 1.30, defaults["water_volume_km3"] / 1e9, 9)
        self.water_volume.setToolTip("Общий запас воды, включая пар. Максимум ограничен областью теплового прототипа.")
        self.duration = self._number(0.0001, 1000, 3, 4)
        self.sample_interval = self._number(0.0001, 10, 0.1, 4)
        self.max_step = self._number(0.000001, 1, 0.01, 6)
        self.max_step.setToolTip("Верхняя граница внутреннего шага. При быстрых изменениях он автоматически уменьшается.")
        self.subdivisions = QComboBox()
        for level in (2, 3, 4):
            self.subdivisions.addItem(f"{20 * 4 ** level:,} ячеек (уровень {level})".replace(",", " "), level)
        self.subdivisions.setCurrentIndex(1)
        self.shell_step = self._number(0.000001, 0.1, 0.002, 6)
        self.shell_step.setToolTip("Шаг пространственной модели: 0,002 млн лет = 2 тысячи лет. В режиме движущейся оболочки он дополнительно уменьшается при необходимости.")
        traction_mpa = configuration.get("genesis_shell", {}).get("convective_traction_pa", 20000) / 1e6
        self.convective_traction = self._number(0, 0.2, traction_mpa, 3)
        self.convective_traction.setSingleStep(0.01)
        self.convective_traction.setToolTip("Амплитуда касательной нагрузки на нижнюю границу оболочки; 0,02 МПа = 20 кПа.")
        self._initial_traction = self.convective_traction.value()
        self.intact_control = QCheckBox("Контроль: отключить все источники напряжений")
        self.intact_control.setToolTip("Остывание и образование покрышки продолжаются; механические нагрузки равны нулю.")
        self.intact_control.toggled.connect(self._update_mode_controls)
        self.tides = QCheckBox("Приливы")
        self.tides.setChecked(True)
        self.tides.setToolTip("Циклические напряжения и средний нагрев по параметрам орбиты спутника.")
        self.water_weakening = QCheckBox("Ослабление при доступе воды")
        self.water_weakening.setChecked(True)
        self.water_weakening.setToolTip("Ослабление зависит от жидкой воды и доступа к повреждённым породам.")
        onset_effects = QHBoxLayout()
        onset_effects.addWidget(self.tides)
        onset_effects.addWidget(self.water_weakening)
        onset_effects.addStretch()
        self.regularization = self._number(1, 5000, 800, 0)
        self.regularization.setSingleStep(100)
        self.regularization.setToolTip("Пространственная длина сглаживания повреждений. Это параметр приближённой модели, не ширина настоящего разлома.")
        for label, widget in (
            ("Начальная температура недр и поверхности, K", self.temperature),
            ("Поток звезды до усреднения по сфере, Вт/м²", self.stellar_flux),
            ("Общий запас воды, млрд км³", self.water_volume),
            ("Конец расчёта, млн лет", self.duration),
            ("Интервал результатов, млн лет", self.sample_interval),
            ("Максимальный внутренний шаг, млн лет", self.max_step),
            ("Сферическая сетка покрышки", self.subdivisions),
            ("Шаг покрышки, млн лет", self.shell_step),
            ("Касательное напряжение от мантии, МПа", self.convective_traction),
            ("", self.intact_control),
            ("Воздействия на спутник", onset_effects),
            ("Масштаб сглаживания повреждений, км", self.regularization),
        ):
            form.addRow(label, widget)
        self._controls = (self.temperature, self.stellar_flux, self.water_volume,
                          self.duration, self.sample_interval, self.max_step, self.mode)
        self._shell_controls = (self.subdivisions, self.shell_step, self.convective_traction, self.intact_control)
        self._onset_controls = (self.tides, self.water_weakening, self.regularization)
        self._initial_values = (self.temperature.value(), self.stellar_flux.value(), self.water_volume.value())
        layout.addLayout(form)
        hint = QLabel("0,1 млн лет = 100 тысяч лет. Альбедо, нагрев гиганта и остальные параметры — в configs/genesis_moon.yaml. Даты событий зависят от приближений атмосферы и теплообмена.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QHBoxLayout()
        self.start_button = QPushButton("Рассчитать генезис")
        self.start_button.clicked.connect(self.start)
        self.stop_button = QPushButton("Остановить")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        self.folder_button = QPushButton("Папка результатов")
        self.folder_button.clicked.connect(self._open_folder)
        self.folder_button.setEnabled(False)
        self.contact_button = QPushButton("Контакт берегов…")
        self.contact_button.clicked.connect(self._open_contact)
        self.contact_button.setEnabled((self.root / "run_genesis_contact.py").is_file())
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.folder_button)
        buttons.addWidget(self.contact_button)
        layout.addLayout(buttons)
        self.status = QLabel("Готов к запуску. Каждый расчёт получает отдельную папку.")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.result_view = QComboBox()
        self.result_view.setEnabled(False)
        self.result_view.currentIndexChanged.connect(self._select_preview)
        self.open_result_button = QPushButton("Открыть крупно")
        self.open_result_button.setEnabled(False)
        self.open_result_button.clicked.connect(self._open_result)
        preview_controls = QHBoxLayout()
        preview_controls.addWidget(self.result_view, 1)
        preview_controls.addWidget(self.open_result_button)
        layout.addLayout(preview_controls)
        self.preview = QLabel("Графики появятся после завершения расчёта")
        self.preview.setMinimumSize(650, 300)
        self.preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.preview, 1)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMaximumHeight(95)
        layout.addWidget(self.log)
        self._update_mode_controls()

    @staticmethod
    def _number(low, high, value, decimals):
        widget = QDoubleSpinBox()
        widget.setDecimals(decimals)
        widget.setRange(low, high)
        widget.setValue(value)
        return widget

    def is_running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def arguments(self, output: Path) -> list[str]:
        mode = self.mode.currentData()
        script = self._runner(mode)
        args = ["-u", str(self.root / script), "--config", str(self.config_path),
                "--output", str(output), "--duration-myr", str(self.duration.value()),
                "--sample-interval-myr", str(self.sample_interval.value()),
                "--max-step-myr", str(self.max_step.value())]
        if mode in {"shell", "onset", "mobile", "faults"}:
            args.extend(["--subdivisions", str(self.subdivisions.currentData()),
                         "--shell-step-myr", str(self.shell_step.value())])
            if self.intact_control.isChecked():
                args.extend(["--control", "intact"])
            elif self.convective_traction.value() != self._initial_traction:
                args.extend(["--convective-traction-mpa", str(self.convective_traction.value())])
        if mode in {"onset", "mobile", "faults"}:
            args.extend(["--regularization-km", str(self.regularization.value())])
            if not self.tides.isChecked() or self.intact_control.isChecked():
                args.append("--no-tides")
            if not self.water_weakening.isChecked():
                args.append("--no-water-weakening")
        for flag, control, initial, scale in (
            ("--initial-temperature-k", self.temperature, self._initial_values[0], 1),
            ("--stellar-flux-w-m2", self.stellar_flux, self._initial_values[1], 1),
            ("--water-volume-km3", self.water_volume, self._initial_values[2], 1e9),
        ):
            if control.value() != initial:
                args.extend([flag, str(control.value() * scale)])
        return args

    @staticmethod
    def _runner(mode: str) -> str:
        return {"faults": "run_genesis_faults.py", "mobile": "run_genesis_mobile.py", "onset": "run_genesis_onset.py", "shell": "run_genesis_shell.py"}.get(mode, "run_genesis.py")

    @staticmethod
    def _checkpoint(mode: str) -> str:
        return {"faults": "fault_checkpoint.npz", "mobile": "mobile_checkpoint.npz", "onset": "onset_checkpoint.npz", "shell": "shell_checkpoint.npz"}.get(mode, "checkpoint.json")

    def start(self) -> None:
        if self.is_running():
            return
        self.output_path = self.root / "results" / "genesis_runs" / datetime.now().strftime("genesis_%Y%m%d_%H%M%S_%f")
        self._generation += 1
        self._stopping = False
        self._closing = False
        self._run_mode = self.mode.currentData()
        self.log.clear()
        self._stop_movie()
        self.result_view.clear()
        self.result_view.setEnabled(False)
        self.open_result_button.setEnabled(False)
        self._pixmap = QPixmap()
        self.preview.clear()
        self.preview.setText("Идёт расчёт…")
        self._set_busy(True)
        self.folder_button.setEnabled(False)
        checkpoint = self._checkpoint(self._run_mode)
        self.status.setText(f"Расчёт запущен. Последний готовый отсчёт сохраняется в {checkpoint}.")
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("MPLBACKEND", "Agg")
        env.insert("PYTHONPATH", str(self.root))
        self.process.setProcessEnvironment(env)
        self.process.setWorkingDirectory(str(self.root))
        self.process.start(sys.executable, self.arguments(self.output_path))

    def _set_busy(self, busy: bool) -> None:
        self.start_button.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
        self.contact_button.setEnabled(not busy and (self.root / "run_genesis_contact.py").is_file())
        for widget in self._controls:
            widget.setEnabled(not busy)
        self._update_mode_controls(busy=busy)

    def _open_contact(self) -> None:
        if self.is_running():
            return
        from .genesis_contact_dialog import GenesisContactDialog
        checkpoint = self.output_path / "fault_checkpoint.npz" if self.output_path is not None else None
        if checkpoint is not None and not checkpoint.is_file():
            checkpoint = None
        dialog = GenesisContactDialog(self, project_root=self.root, checkpoint=checkpoint)
        if checkpoint is None:
            dialog.choose_checkpoint()
        dialog.exec()

    def _update_mode_controls(self, _index=None, *, busy: bool | None = None) -> None:
        if not hasattr(self, "_shell_controls"):
            return
        if busy is None:
            busy = self.is_running()
        for widget in self._shell_controls:
            widget.setEnabled(not busy and self.mode.currentData() in {"shell", "onset", "mobile", "faults"})
        for widget in self._onset_controls:
            widget.setEnabled(not busy and self.mode.currentData() in {"onset", "mobile", "faults"})
        if self.intact_control.isChecked():
            self.convective_traction.setEnabled(False)
            self.tides.setEnabled(False)

    def stop(self) -> None:
        if not self.is_running():
            return
        self._stopping = True
        self.stop_button.setEnabled(False)
        self.status.setText("Остановка… Последний завершённый checkpoint останется в папке результатов.")
        generation = self._generation
        self.process.terminate()
        QTimer.singleShot(1500, lambda: self._kill_if_current(generation))

    def _kill_if_current(self, generation: int) -> None:
        if generation == self._generation and self.is_running():
            self.process.kill()

    def _read_output(self) -> None:
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self.log.appendPlainText(text.rstrip())

    def _finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self._read_output()
        self._set_busy(False)
        output = self.output_path
        self.folder_button.setEnabled(output is not None and output.is_dir())
        if self._stopping:
            mode = getattr(self, "_run_mode", self.mode.currentData())
            script, checkpoint = self._runner(mode), self._checkpoint(mode)
            self.status.setText(f"Расчёт остановлен. Продолжение: {script} --resume {checkpoint}, если checkpoint успел сохраниться.")
            self._load_previews(output)
        elif exit_code == 0 and exit_status == QProcess.ExitStatus.NormalExit and output is not None:
            try:
                summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
                ocean = summary.get("events_myr", {}).get("ocean_start")
                result = f"Конденсация началась около {ocean:.3f} млн лет." if ocean is not None else "За этот интервал конденсация не началась."
                shell = summary.get("shell")
                if shell:
                    first = shell.get("first_fracture_time_myr")
                    result += (f" Порог повреждения достигнут: {first:.3f} млн лет." if first is not None
                               else " Порог повреждения покрышки не достигнут.")
                    if "damaged_area_fraction" in shell:
                        result += f" Повреждено: {shell['damaged_area_fraction']:.1%} площади."
                    if "intact_region_count" in shell:
                        result += f" Связных областей: {shell['intact_region_count']}."
                    if "fault_active_area_fraction" in shell:
                        active_fraction = shell["fault_active_area_fraction"]
                        result += (f" Активные разломные зоны: {100*active_fraction:.4g}% площади." if active_fraction > 0
                                   else " Активные разломные зоны не образовались (0% площади).")
                    if "max_equivalent_slip_km" in shell:
                        result += f" Максимальный эквивалентный сдвиг: {shell['max_equivalent_slip_km']:.4g} км."
                onset = summary.get("onset", {})
                if "mean_speed_cm_yr" in onset:
                    result += f" Средняя скорость смещений: {onset['mean_speed_cm_yr']:.3f} см/год."
                status = summary.get("status", "completed")
                if status == "completed":
                    result = "Расчёт завершён до заданного времени. " + result
                elif status == "shell_small_strain_limit":
                    result += " Достигнут предел малых деформаций оболочки; дальнейший расчёт требует изменения геометрии."
                elif status in {"surface_freezing_limit", "surface_reached_freezing_limit_ice_not_modelled"}:
                    result += " Достигнут предел модели: замерзание воды пока не рассчитывается."
                elif status != "completed":
                    result += f" Расчёт остановлен: {status}."
                self.status.setText(result + "\n" + str(output))
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                self.status.setText(f"Не удалось прочитать результат: {exc}")
            self._load_previews(output)
        else:
            self.status.setText("Расчёт завершился с ошибкой. Подробности в журнале ниже.")
        if self._closing:
            QTimer.singleShot(0, self.close)

    def _error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._set_busy(False)
            self.status.setText("Не удалось запустить Python: " + self.process.errorString())
            if self._closing:
                QTimer.singleShot(0, self.close)

    def _show_preview(self) -> None:
        if not self._pixmap.isNull():
            self.preview.setPixmap(self._pixmap.scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                        Qt.TransformationMode.SmoothTransformation))

    def _stop_movie(self) -> None:
        if self._movie is not None:
            self._movie.stop()
            self._movie.deleteLater()
            self._movie = None

    def _load_previews(self, output: Path | None) -> None:
        self._stop_movie()
        self.result_view.blockSignals(True)
        self.result_view.clear()
        if output is not None:
            for label, filename in (("Трение и сдвиг разломных зон", "genesis_faults.png"),
                                    ("Анимация сдвига разломных зон", "genesis_faults.gif"),
                                    ("Движущаяся оболочка", "genesis_mobile.png"),
                                    ("Анимация движения оболочки", "genesis_mobile.gif"),
                                    ("Приливы, вода и первые смещения", "genesis_onset.png"),
                                    ("Анимация первых смещений", "genesis_onset.gif"),
                                    ("Карта покрышки и повреждений", "genesis_shell.png"),
                                    ("Анимация формирования покрышки", "genesis_shell.gif"),
                                    ("История остывания и океана", "genesis_history.png")):
                path = output / filename
                if path.is_file():
                    self.result_view.addItem(label, str(path))
        self.result_view.blockSignals(False)
        self.result_view.setEnabled(self.result_view.count() > 0)
        self._select_preview()

    def _select_preview(self, _index=None) -> None:
        self._stop_movie()
        self._pixmap = QPixmap()
        self.preview.clear()
        path = self.result_view.currentData()
        self.open_result_button.setEnabled(bool(path))
        if not path:
            self.preview.setText("Сохранённые изображения пока недоступны")
            return
        if Path(path).suffix.lower() == ".gif":
            self._movie = QMovie(path, parent=self)
            if not self._movie.isValid():
                self.preview.setText("Не удалось открыть анимацию")
                return
            self._movie.frameChanged.connect(self._show_movie_frame)
            self._movie.start()
        else:
            self._pixmap = QPixmap(path)
            if self._pixmap.isNull():
                self.preview.setText("Не удалось открыть изображение")
            else:
                self._show_preview()

    def _open_result(self) -> None:
        path = self.result_view.currentData()
        if path and Path(path).is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).resolve())))

    def _show_movie_frame(self, _frame=None) -> None:
        if self._movie is not None:
            self._pixmap = self._movie.currentPixmap()
            self._show_preview()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "preview"):
            self._show_preview()

    def _open_folder(self) -> None:
        if self.output_path is not None and self.output_path.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_path.resolve())))

    def reject(self) -> None:
        self.close()

    def closeEvent(self, event) -> None:
        if self.is_running():
            self._closing = True
            self.stop()
            event.ignore()
        else:
            self._stop_movie()
            event.accept()
