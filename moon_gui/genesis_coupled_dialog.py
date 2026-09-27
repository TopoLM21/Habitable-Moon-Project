"""Cancellable launcher for physical thermal/orbital/Maxwell/contact evolution."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import QProcess, QTimer
from PySide6.QtWidgets import QFileDialog, QLabel

from .genesis_contact_dialog import GenesisContactDialog, ROOT


class GenesisCoupledDialog(GenesisContactDialog):
    """Reuse process and image-view plumbing, with a separate physical workflow."""

    def __init__(self, parent=None, *, project_root: Path = ROOT, checkpoint: Path | None = None):
        super().__init__(parent, project_root=project_root)
        self.setWindowTitle("Генезис — остывание и развитие разломов")
        self.introduction.setText(
            "Остывание продолжается вместе с изменением орбиты, доступом воды, ростом оболочки, "
            "релаксацией напряжений и контактом разделённых берегов. Все процессы используют одно физическое время. "
            "Новые затвердевшие участки контакта получают собственную историю; раскрытые щели сами собой не срастаются."
        )
        self.source_path.setPlaceholderText("Выберите исходный fault_checkpoint.npz или coupled_checkpoint.npz")
        self.start_button.setText("Продолжить остывание")
        self.handoff_button.hide()
        self.coupled_button.hide()
        self.step.setValue(100.)
        self.duration.setToolTip("Целевое время от исходного состояния разломов; при возобновлении задайте новый конец.")
        self.source_path.setToolTip("Для возобновления нужен совместный архив версии 0.2. Для старого расчёта 0.1 выберите исходный файл разломов.")
        for label in self.findChildren(QLabel):
            if label.text() == "Конец механического продолжения, лет":
                label.setText("Конец физического продолжения, лет")
            elif label.text().startswith("Исходно соседние рёбра взаимодействуют"):
                label.setText("Эксперимент ограничен малыми деформациями и скольжениями. "
                              "Тепловой и механический балансы учитываются отдельно. "
                              "Переход к зрелой тектонике ещё не выполняется.")
        self.status.setText("Выберите исходное состояние разломов. Замороженный контакт не подходит для возобновления остывания.")
        self.preview.setText("Карты берегов и история температуры появятся после расчёта")
        if checkpoint is not None:
            self.set_checkpoint(checkpoint)

    def choose_checkpoint(self):
        if self.is_running():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Исходное состояние для остывания и разломов",
            str(self.checkpoint_path or self.root / "results" / "genesis_runs"), "Состояние генезиса (*.npz)")
        if path:
            self.set_checkpoint(Path(path))

    def set_checkpoint(self, path):
        if self.is_running():
            return False
        try:
            path = Path(path).resolve()
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"]))
            version = str(metadata["format"])
            if not version.startswith(("genesis-faults-", "genesis-coupled-")):
                raise ValueError("нужно состояние разломов или совместного расчёта; замороженный контакт несовместим")
            resume = version.startswith("genesis-coupled-")
            elapsed = ((float(metadata["state"]["time_myr"]) - float(metadata["source_time_myr"])) * 1e6
                       if resume else 0.)
            if not np.isfinite(elapsed) or elapsed < 0:
                raise ValueError("неверное физическое время состояния")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            self.status.setText(f"Не удалось выбрать состояние: {exc}")
            return False
        self.checkpoint_path = path
        self._resume, self._source_elapsed = resume, elapsed
        self.source_path.setText(str(path))
        self.source_path.setToolTip(str(path))
        self.source_kind.setText(f"Возобновление совместного расчёта: прошло {elapsed:g} физических лет." if resume
                                 else "Продолжение исходного охлаждающегося состояния разломов.")
        if self.duration.value() <= elapsed:
            self.duration.setValue(min(self.duration.maximum(), elapsed + 1000.))
        self.start_button.setEnabled(True)
        self.status.setText("Готово: температура, орбита, оболочка и контакт будут развиваться совместно.")
        return True

    def arguments(self, output):
        if self.checkpoint_path is None:
            raise ValueError("Сначала выберите исходное состояние")
        return ["-u", str(self.root / "run_genesis_coupled.py"),
                "--resume" if self._resume else "--checkpoint", str(self.checkpoint_path),
                "--output", str(output), "--duration-years", str(self.duration.value()),
                "--step-years", str(self.step.value())]

    def start(self):
        if self.is_running():
            return
        if self.checkpoint_path is None or not self.checkpoint_path.is_file():
            self.status.setText("Сначала выберите существующее исходное состояние.")
            return
        if self.duration.value() <= self._source_elapsed:
            self.status.setText("Конец расчёта должен быть позже сохранённого физического времени.")
            return
        self.output_path = self._new_output("coupled")
        self._launch(self.arguments(self.output_path), "coupled")
        self.preview.setText("Идёт совместное остывание и развитие разломов…")
        self.status.setText("Совместный расчёт запущен. Готовые отсчёты сохраняются в coupled_checkpoint.npz.")

    def stop(self):
        running = self.is_running()
        super().stop()
        if running:
            self.status.setText("Остановка… Последнее завершённое физическое состояние сохранится в coupled_checkpoint.npz.")

    def _finished(self, exit_code, exit_status):
        from visualization.genesis_connectivity import connectivity_text

        self._read_output()
        self._set_busy(False)
        output = self.output_path
        self.folder_button.setEnabled(output is not None and output.is_dir())
        if self._stopping:
            self.status.setText("Расчёт остановлен. Для продолжения выберите последний coupled_checkpoint.npz.")
        elif exit_code == 0 and exit_status == QProcess.ExitStatus.NormalExit and output is not None:
            try:
                result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
                if result.get("model") != "genesis-coupled" or "GENESIS_COUPLED_COMPLETE" not in self._process_output:
                    raise ValueError("нет подтверждения завершения совместного расчёта")
                final = result["final"]
                message = (f"Возраст: {final['time_myr']:.6f} млн лет; продолжение: {final['elapsed_years']:.6g} лет. "
                           f"Поверхность: {final['surface_temperature_k']:.4g} K. "
                           f"Прочная оболочка: {final['mean_lid_thickness_km']:.4g} км.")
                if note := connectivity_text(final):
                    message += "\n" + note
                if result.get("status", "completed") != "completed":
                    message += f" Остановка: {result['status']}."
                self.status.setText(message + "\n" + str(output))
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                self.status.setText(f"Не удалось прочитать результат: {exc}")
        else:
            self.status.setText("Совместный расчёт завершился с ошибкой. Подробности в журнале ниже.")
        self._load_previews(output)
        QTimer.singleShot(0, self._show_preview)
        if self._closing:
            QTimer.singleShot(0, self.close)

    def _load_previews(self, output):
        self._stop_movie()
        self.result_view.blockSignals(True)
        self.result_view.clear()
        if output is not None:
            for label, filename in (("Остывание, океан и разломы", "genesis_coupled.png"),
                                    ("Анимация физической эволюции", "genesis_coupled.gif")):
                path = output / filename
                if path.is_file():
                    self.result_view.addItem(label, str(path))
        self.result_view.blockSignals(False)
        self.result_view.setEnabled(self.result_view.count() > 0)
        self._select_preview()
