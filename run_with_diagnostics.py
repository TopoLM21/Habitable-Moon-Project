#!/usr/bin/env python3
"""Run an existing model entry point with observable stages and hang dumps."""
from __future__ import annotations

import argparse
import functools
import importlib.abc
import importlib.machinery
import importlib.util
from pathlib import Path
import sys


_STAGES = {
    "parse_args": "Чтение параметров запуска",
    "load_config": "Чтение конфигурации",
    "build_prototype": "Создание начальной модели и сетки",
    "load_checkpoint": "Загрузка контрольной точки",
    "initialize_lithosphere": "Инициализация литосферы и континентов",
    "initialize_oceanic_crust_ages": "Расчёт начального возраста океанической коры",
    "initialize_mantle_flow": "Инициализация потоков мантии",
    "initialize_topography": "Инициализация рельефа",
    "initialize_hydrosphere": "Инициализация океана и уровня моря",
    "advance_thermal_state": "Расчёт температуры",
    "advance_mantle_flow": "Расчёт потоков мантии",
    "boundary_records_for_state": "Расчёт границ плит",
    "advance_subduction_memory": "Расчёт памяти субдукции",
    "advance_slab_breakoff": "Расчёт отрыва погружающихся плит",
    "advance_rollback": "Расчёт отступания зон субдукции",
    "update_plate_dynamics": "Расчёт движения плит",
    "advance_lithosphere": "Перенос литосферы и континентального материала",
    "advance_continental_cycle": "Расчёт континентального цикла и плюмов",
    "refresh_mechanical_lithosphere": "Обновление механической литосферы",
    "advance_late_tectonics": "Расчёт поздней тектоники и рифтов",
    "compute_volcanic_arc_forcing": "Расчёт вулканических дуг",
    "advance_topography": "Расчёт рельефа",
    "advance_sediments": "Эрозия, перенос и осаждение осадков",
    "advance_hydrosphere": "Расчёт океана и уровня моря",
    "equilibrium_elevation": "Расчёт равновесного рельефа",
    "topography_components": "Расчёт составляющих рельефа",
    "build_checkpoint": "Подготовка контрольной точки",
    "save_checkpoint": "Запись контрольной точки",
    "advance_mantle_plumes": "Расчёт мантийных плюмов",
    "advance_plume_rifting": "Расчёт рифтинга под действием плюмов",
    "advance_hotspot_tracks": "Расчёт следов горячих точек",
}

_DETAIL_STAGES = {
    "tectonics.simulation": {
        "build_icosphere": "Построение сферической сетки и площадей ячеек",
        "random_plate_system": "Создание начальных плит",
        "classify_boundaries": "Классификация начальных границ плит",
        "rigid_motion_residual": "Расчёт начальных скоростей плит",
    },
    "tectonics.mesh": {
        "_subdivide": "Деление треугольников сетки",
        "_build_topology": "Построение соседей и рёбер сетки",
    },
    "tectonics.transport": {
        "build_transport_map": "Построение карты переноса материала",
        "_optimal_assignment": "Согласование исходных и целевых ячеек переноса",
    },
    "tectonics.lithosphere": {
        "_redistribute_collision_overflow": "Перераспределение материала при столкновении",
        "_redistribute_continental_footprint_overflow": "Перераспределение площади континентов",
    },
    "tectonics.topography": {"solve_flexural_response": "Расчёт упругого изгиба литосферы"},
    "tectonics.flexure": {"_geometry_operators": "Построение геометрических весов рельефа"},
}


def _details(args, kwargs):
    """Read small scalar metadata only; never stringify model arrays."""
    result = {}
    for value in (*args, *kwargs.values()):
        time = getattr(value, "time_myr", None)
        if isinstance(time, (int, float)):
            result.setdefault("time_myr", float(time))
        if isinstance(value, (str, Path)):
            result.setdefault("path", str(value))
    return result


class _Hooks:
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        self.patches = []
        self.installed = False
        self.failure_dumped = False

    def failure(self):
        if not self.failure_dumped:
            self.failure_dumped = True
            self.diagnostics.dump("Ошибка или прерывание выполнения")

    def replace(self, owner, name, replacement):
        original = getattr(owner, name)
        setattr(owner, name, replacement)
        self.patches.append((owner, name, original, replacement))

    def wrap(self, owner, name, label):
        original = getattr(owner, name, None)
        if not callable(original):
            return

        @functools.wraps(original)
        def observed(*args, **kwargs):
            with self.diagnostics.stage(label, function=name, **_details(args, kwargs)):
                try:
                    return original(*args, **kwargs)
                except BaseException:
                    self.failure()
                    raise

        self.replace(owner, name, observed)

    def defer(self, base):
        original = base.main

        @functools.wraps(original)
        def observed_main(*args, **kwargs):
            # Version, CPU, GPU and rendering entry points rebind these symbols.
            # Install only now, after their configuration has taken effect.
            self.install(base)
            return original(*args, **kwargs)

        self.replace(base, "main", observed_main)

    def install(self, base):
        if self.installed:
            return
        self.installed = True
        for module_name, module in tuple(sys.modules.items()):
            if module is None or not module_name.startswith("run_long_evolution_v"):
                continue
            for name in tuple(vars(module)):
                label = _STAGES.get(name)
                if label is None and name.startswith("_write_v") and name.endswith("_outputs"):
                    label = "Запись итоговых материалов " + name.split("_")[2]
                if label is None and (name.startswith("save_") or name.startswith("_save_frame")):
                    label = "Построение и запись изображения / графика"
                if label is None and name.startswith("build_") and "gif" in name:
                    label = "Сборка GIF-анимации"
                if label:
                    self.wrap(module, name, label)
        for module_name, stages in _DETAIL_STAGES.items():
            module = sys.modules.get(module_name)
            if module is not None:
                for name, label in stages.items():
                    self.wrap(module, name, label)
        manager = getattr(base, "PlateTopologyManager", None)
        if manager is not None:
            self.wrap(manager, "update", "Изменение топологии: разделение и объединение плит")
            self.wrap(manager, "extension_suppression_field", "Расчёт влияния столкновений на рифтинг")
        rendering = sys.modules.get("visualization.render_runtime")
        if rendering is not None and hasattr(rendering, "RenderExecution"):
            self.wrap(rendering.RenderExecution, "flush", "Ожидание завершения записи изображений")
        original_steps = base.step_sizes

        @functools.wraps(original_steps)
        def observed_steps(start, end, dt):
            steps = original_steps(start, end, dt)
            current = start
            count = len(steps) if hasattr(steps, "__len__") else None
            for number, step in enumerate(steps, 1):
                with self.diagnostics.stage(
                    "Шаг моделирования", step=number, step_count=count,
                    time_myr=current, target_time_myr=end, dt_myr=step,
                ):
                    yield step
                current += step  # Reporting only; the original step is yielded unchanged.

        self.replace(base, "step_sizes", observed_steps)

    def restore(self):
        for owner, name, original, replacement in reversed(self.patches):
            if getattr(owner, name, None) is replacement:
                setattr(owner, name, original)


class _BaseImportHook(importlib.abc.MetaPathFinder):
    """Observe one lazy import without moving it ahead of execution setup."""
    def __init__(self, hooks):
        self.hooks = hooks

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "run_long_evolution_v123":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        original_loader = spec.loader
        hooks = self.hooks

        class Loader:
            def create_module(self, module_spec):
                create = getattr(original_loader, "create_module", None)
                return create(module_spec) if create else None

            def exec_module(self, module):
                original_loader.exec_module(module)
                hooks.defer(module)

        spec.loader = Loader()
        return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--optimize-assignment", action="store_true",
                        help="Use the experimental sparse heap assignment solver")
    parser.add_argument("runner", type=Path)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    runner_path = options.runner.resolve()
    from moon_gui.diagnostics import WorkerDiagnostics

    diagnostics = WorkerDiagnostics(options.diagnostics_dir)
    hooks = _Hooks(diagnostics)
    finder = _BaseImportHook(hooks)
    old_argv, old_path = sys.argv, sys.path[:]
    sys.argv = [str(runner_path), *options.runner_args]
    sys.path.insert(0, str(runner_path.parent))
    sys.meta_path.insert(0, finder)
    diagnostics.start()
    try:
        with diagnostics.stage("Выполнение расчёта", runner=runner_path.name):
            try:
                with diagnostics.stage("Загрузка программы и вычислительных библиотек"):
                    base = sys.modules.get("run_long_evolution_v123")
                    if base is not None:
                        hooks.defer(base)
                    spec = importlib.util.spec_from_file_location(runner_path.stem, runner_path)
                    if spec is None or spec.loader is None:
                        raise ImportError(f"Cannot load runner: {runner_path}")
                    runner = importlib.util.module_from_spec(spec)
                    sys.modules[spec.name] = runner
                    spec.loader.exec_module(runner)
                    if runner_path.stem == "run_long_evolution_v123":
                        hooks.defer(runner)
                from tectonics.assignment_runtime import AssignmentExecution
                with AssignmentExecution(diagnostics, optimized=options.optimize_assignment):
                    runner.main()
            except BaseException:
                hooks.failure()
                raise
    finally:
        hooks.restore()
        sys.meta_path.remove(finder)
        sys.argv, sys.path[:] = old_argv, old_path
        diagnostics.close()


if __name__ == "__main__":
    main()
