"""Standalone thermal genesis diagnostics, with no invented plate geometry."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_genesis_history(rows: list[dict], path: str | Path, events: dict[str, float]) -> None:
    t = np.array([row["time_myr"] for row in rows])
    def column(name):
        return np.array([row[name] for row in rows])
    with plt.rc_context({"font.size": 10, "axes.titlesize": 12, "figure.facecolor": "#f7f9fc",
                         "axes.facecolor": "white", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": "#29374d",
                         "text.color": "#172638"}):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
        ax = axes[0, 0]
        ax.plot(t, column("mantle_temperature_k"), color="#dc593d", label="Недра")
        ax.plot(t, column("surface_temperature_k"), color="#e49a24", label="Поверхность")
        ax.axhline(647.096, color="#7d8c9f", ls=":", lw=1, label="Критическая T воды")
        ax.set(title="Раздельное остывание недр и поверхности", ylabel="Температура, K")
        ax.legend(loc="best", fontsize=9)
        ax = axes[0, 1]
        ax.plot(t, column("mantle_melt_fraction"), color="#dc593d", label="Расплав в недрах")
        ax.plot(t, column("surface_melt_fraction"), color="#e49a24", label="Расплав у поверхности")
        ax.plot(t, column("ocean_fraction"), color="#237fab", label="Вода в океане")
        ax.set(title="От океана магмы к водному океану", ylabel="Доля", ylim=(-0.03, 1.06))
        ax.legend(loc="best", fontsize=9)
        ax = axes[1, 0]
        ax.plot(t, column("steam_pressure_bar"), color="#795bac", label="Пар / горячий флюид")
        ax.set(title="Запас воды сохраняется при конденсации", ylabel="Эквивалентное давление воды, бар")
        ax2 = ax.twinx()
        ax2.spines["right"].set_visible(True)
        ax2.plot(t, column("ocean_volume_km3") / 1e9, color="#237fab", label="Океан")
        ax2.set_ylabel("Объём океана, млрд км³", color="#237fab")
        ax.legend(loc="upper left", fontsize=9)
        ax2.legend(loc="upper right", fontsize=9)
        ax = axes[1, 1]
        ax.plot(t, column("outgoing_longwave_w_m2"), color="#237fab", label="Излучение в космос")
        ax.plot(t, column("absorbed_stellar_flux_w_m2") + column("giant_absorbed_flux_w_m2"), color="#e49a24", label="Поглощённый внешний поток")
        ax.plot(t, column("net_cooling_flux_w_m2"), color="#475365", label="Чистое охлаждение")
        ax.axhline(0, color="#a5afbc", lw=0.8)
        ax.set(title="Баланс тепла на единицу площади", ylabel="Средний поток, Вт/м²")
        ax.set_yscale("symlog", linthresh=10)
        ax.set_ylim(min(0.0, float(column("net_cooling_flux_w_m2").min()) * 1.2),
                    float(max(column("outgoing_longwave_w_m2").max(),
                              (column("absorbed_stellar_flux_w_m2") + column("giant_absorbed_flux_w_m2")).max())) * 1.2 + 1)
        ax.legend(loc="best", fontsize=9)
        for ax in axes.flat:
            ax.set_xlabel("Время после расплавленного старта, млн лет")
            ax.grid(alpha=0.16)
            ax.set_xlim(float(t[0]), float(t[-1]) if t[-1] > t[0] else float(t[0]) + 0.1)
            if "ocean_start" in events:
                ax.axvline(events["ocean_start"], color="#237fab", alpha=0.35, ls="--", lw=1)
            # Resolve the formation interval while retaining the late history.
            if t[-1] > 3:
                ax.set_xscale("symlog", linthresh=1.0)
                ticks = [v for v in (0, 0.5, 1, 2, 5, 10, 20, 50, 100) if t[0] <= v <= t[-1]]
                ax.set_xticks(ticks, [f"{v:g}" for v in ticks])
        ocean = f"Начало конденсации: {events['ocean_start']:.3f} млн лет" if "ocean_start" in events else "Конденсация в этом интервале не началась"
        fig.suptitle("Генезис · тепловой эксперимент\n" + ocean, fontsize=16, fontweight="bold")
        note = "Упрощённая атмосфера и реология; география и подвижные плиты ещё не рассчитываются."
        if t[-1] > 3:
            note += "\nШкала времени после 1 млн лет — логарифмическая."
        fig.supxlabel(note, fontsize=10)
        fig.savefig(path, dpi=155)
        plt.close(fig)
