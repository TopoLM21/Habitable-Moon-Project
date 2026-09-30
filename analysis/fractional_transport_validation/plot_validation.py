"""Russian figure for the independent frozen-motion transport experiment."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    data=json.loads(args.source.read_text(encoding="utf-8"))
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
        "axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8),layout="constrained")
    colors=("#0077a5","#c16b14","#7561a5")
    names=("saved50_dt1","saved50_dt0p5","saved50_dt0p25")
    labels=("Шаг 1 млн лет","Шаг 0,5 млн лет","Шаг 0,25 млн лет")
    for name,label,color in zip(names,labels,colors):
        case=data["cases"][name]
        times=[0.]+[row["elapsed_myr"] for row in case["history"]]
        volume=[0.]+[row["cumulative_losses"]["oceanic_volume_km3"]/1e3 for row in case["history"]]
        axes[0].plot(times,volume,"o-",ms=4,color=color,label=label)
    raster=next(row for row in data["raster_reference"]["saved50"] if row["kind"]=="zero_residual_control")
    if raster["committed_plates"]==0:
        axes[0].scatter([1.],[0.],marker="s",color="#555555",s=45,
            label="Растр: смен ячеек ещё нет",zorder=5)
    axes[0].set(title="Принятый объём на одной сетке",
        xlabel="Время проверки (млн лет)",ylabel="Объём океанической коры (тыс. км³)",xlim=(-.025,1.025))
    axes[0].legend(fontsize=8,frameon=False)
    first=data["cases"][names[0]]["cumulative_losses"]["oceanic_volume_km3"]
    last=data["cases"][names[-1]]["cumulative_losses"]["oceanic_volume_km3"]
    difference=f"{100*abs(last/first-1):.5f}".replace(".",",")
    axes[0].text(.48,.20,"Шаг 1 → 0,25 млн лет:\nразличие объёма "+difference+"%",
        transform=axes[0].transAxes,fontsize=9,color="#444444")
    fractions=[100*data["cases"][name]["multi_owner_cell_fraction"] for name in names]
    axes[1].bar(["1","0,5","0,25"],fractions,color=colors,width=.55)
    axes[1].set(title="Разные владельцы внутри ячейки",
        xlabel="Шаг переноса (млн лет)",ylabel="Доля ячеек с несколькими плитами (%)")
    for index,value in enumerate(fractions):
        axes[1].text(index,value+max(fractions)*.02,f"{value:.2f}%".replace(".",","),ha="center",fontsize=9)
    axes[1].set_ylim(0,max(fractions)*1.15)
    for ax in axes:
        ax.grid(axis="y",alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("Дробный перенос при фиксированном движении плит",fontsize=14,fontweight="bold")
    fig.supxlabel("Один исходный снимок 0.5 на 50 млн лет. Скорости, нагрев и прочность не пересчитываются.\nРастровый контроль начинается с нулевого остаточного поворота. Смешение включает численную диффузию.",
        fontsize=9,color="#555555")
    fig.savefig(args.output,dpi=180)
    print(args.output.resolve())


if __name__=="__main__":
    main()
