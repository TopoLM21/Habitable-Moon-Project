"""Check measured genesis fragment motion before transfer to mature tectonics.

An optional short contact continuation supplies consecutive observations. The
source checkpoint is never modified. This command writes a screening report,
not a mature checkpoint and not a continuation of thermal/orbital evolution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from tectonics.genesis_contact import load_contact_checkpoint, save_contact_checkpoint
from tectonics.genesis_handoff import HandoffScreenParameters, assess_contact_handoff


def write_report(path, report):
    lines = ["# Переход от генезиса к зрелой тектонике", "",
             "**Текущее состояние пока не готово к передаче.**", "",
             f"Тепловой возраст: {report['source_time_myr']:.6g} млн лет. "
             f"Наблюдение механики: {report['start_elapsed_years']:.6g}–{report['end_elapsed_years']:.6g} лет после исходного снимка.",
             "Эти механические годы не прибавлены к рассчитанной тепловой истории.", "",
             "| Фрагмент | Ячеек | Доля поверхности | Относительная скорость, см/год | Остаток после подбора жёсткого вращения |",
             "|---|---:|---:|---:|---:|"]
    for region in report["regions"]:
        lines.append(f"| {region['region_id']+1} | {region['cell_count']} | {region['area_fraction']:.3%} | "
                     f"{region['relative_rms_speed_km_myr']/10:.4f} | {region['rigid_residual_fraction']:.2%} |")
    lines.extend(["", "Из скоростей вычтено общее вращение оболочки. Малый остаток означает хорошее приближение жёсткой плитой; "
                  "сам по себе он не доказывает возникновение плит.", "", "## Что препятствует переходу", ""])
    lines.extend(f"- {blocker['message']}" for blocker in report["blockers"])
    lines.extend(["", "## Что уже можно использовать", "",
                  "Зрелый движок уже рассчитывает перенос материала, образование океанической коры, субдукцию, "
                  "континентальный цикл, мантийное течение, плюмы, рельеф и океан. Этот отчёт проверяет начальные "
                  "условия их подключения; повторная реализация этих процессов в генезисе не требуется.", "",
                  "Пороги проверки — явные численные настройки, пока не калиброванные физические критерии. "
                  "Суммарная знаковая площадь не измеряет отдельно отверстия и перекрытия.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def save_figure(path, report, arrays):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 5.2), constrained_layout=True)
    ax = fig.add_subplot(121, projection="mollweide")
    centers = arrays["centroids"]
    longitude = np.arctan2(centers[:, 1], centers[:, 0])
    latitude = np.arcsin(np.clip(centers[:, 2], -1, 1))
    ax.scatter(longitude, latitude, c=arrays["region_labels"], cmap="tab10", s=22)
    ax.grid(alpha=.3)
    ax.set_title("Связные фрагменты оболочки")
    ax = fig.add_subplot(122)
    regions = report["regions"]
    x = np.arange(len(regions))
    ax.bar(x-.18, [r["relative_rms_speed_km_myr"]/10 for r in regions], .36, label="Наблюдаемое относительное движение")
    ax.bar(x+.18, [r["rigid_residual_km_myr"]/10 for r in regions], .36, label="Не объясняется жёстким вращением")
    ax.set_xticks(x, [f"Фрагмент {r['region_id']+1}\n{r['cell_count']} ячеек; {r['area_fraction']:.2%}" for r in regions])
    ax.set_ylabel("Среднеквадратичная скорость, см/год")
    ax.set_title("Движение без общего вращения оболочки")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.25)
    fig.suptitle(f"Переход пока не готов | тепловой возраст {report['source_time_myr']:.4g} млн лет | "
                 f"наблюдение механики {report['observation_years']:.4g} лет")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-years", type=float, default=10., help="Duration of each additional mechanical observation")
    parser.add_argument("--intervals", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        if not np.isfinite(args.probe_years) or args.probe_years <= 0 or not 2 <= args.intervals <= 100:
            raise ValueError("Use a positive finite probe interval and 2..100 intervals")
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            raise ValueError("Output must be new or empty; existing results are protected")
        model, state = load_contact_checkpoint(args.checkpoint)
        if state.stopped_reason:
            raise ValueError("Cannot probe a stopped contact state")
        if state.elapsed_years+args.intervals*args.probe_years > model.frozen_window_years:
            raise ValueError("Requested observations exceed the source-dependent frozen contact window")
        states = [state]
        for index in range(1, args.intervals+1):
            state = model.step(state, states[0].elapsed_years+index*args.probe_years)
            states.append(state)
            print(f"handoff observation {index}/{args.intervals}: {state.elapsed_years:.6g} mechanical years", flush=True)
            if state.stopped_reason:
                if state.elapsed_years == states[-2].elapsed_years:
                    states.pop()
                break
        if len(states) < 2:
            raise ValueError("Contact solver could not supply a new observation")
        screen = HandoffScreenParameters(min_observation_years=model.source_model.onset_p.persistence_time_myr*1e6)
        report, arrays = assess_contact_handoff(model, states, screen)
        if state.stopped_reason:
            report["probe_stopped_reason"] = state.stopped_reason
            if not any(b["code"] == "accepted_solver_states" for b in report["blockers"]):
                report["blockers"].append({"code": "probe_solver_limit", "message": "Пробное продолжение достигло ограничения контактного решателя."})
            report["kinematic_screen_passed"] = False
        report["provenance"] = {"source_path": str(args.checkpoint.resolve()),
                                "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                                "probe_years": args.probe_years, "requested_intervals": args.intervals}
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/"handoff_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        np.savez_compressed(args.output/"handoff_fields.npz", **arrays)
        save_contact_checkpoint(args.output/"probe_contact_checkpoint.npz", model, states[-1])
        write_report(args.output/"handoff_report.md", report)
        save_figure(args.output/"handoff_assessment.png", report, arrays)
        print(f"HANDOFF_SCREEN_COMPLETE ready=false {args.output.resolve()}", flush=True)
        return 0  # Successful diagnosis can legitimately find an unready shell.
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Genesis handoff assessment error: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
