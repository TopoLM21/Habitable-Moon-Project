"""Write a compact Russian report from completed transport-only validation."""
from pathlib import Path
import argparse
import json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    data=json.loads(args.source.read_text(encoding="utf-8"))
    cases=data["cases"]
    reference=cases["saved50_dt1"]["cumulative_losses"]["oceanic_volume_km3"]
    fine=cases["saved50_refined_dt1"]["cumulative_losses"]["oceanic_volume_km3"]
    quarter=cases["saved50_dt0p25"]["cumulative_losses"]["oceanic_volume_km3"]
    donor_max=max(value for case in cases.values() for row in case["history"]
        for value in row["material_id_budget"]["maximum_relative_residual"].values())
    lines=[]
    for name,label in (("saved50_dt1","sub4; шаг1"),("saved50_dt0p5","sub4; шаг0,5"),
            ("saved50_dt0p25","sub4; шаг0,25"),("saved50_refined_dt1","sub5; шаг1")):
        case=cases[name]
        lines.append(f"| {label} | {case['cumulative_losses']['oceanic_volume_km3']:.6f} | "
            f"{100*case['multi_owner_cell_fraction']:.3f}% | {100*case['minority_owner_area_fraction']:.6f}% |")
    text=f"""# Проверка консервативного дробного переноса

Это независимый эксперимент транспорта. Он не меняет механику0.5 по умолчанию и не пересчитывает скорости плит, нагрев, прочность, разрушение или силу погружённой плиты. Использованы исходный Starter и сохранённое состояние0.5 на50 млн лет; старые файлы не изменены.

На исходном Starter за1 млн лет принято {cases['starter_dt1']['cumulative_losses']['oceanic_volume_km3']:.6f} км³ океанической коры. В контрольном растровом отображении ещё нет смены ячеек. Для состояния на50 млн лет новый алгоритм также принимает дробные порции до raster commit в отдельно обозначенном контроле с нулевым остаточным поворотом; фактический сохранённый остаток исследован отдельно в JSON. Из raster map не выводилась фиктивная величина принятой массы.

Ниже перенос одного и того же состояния на одинаковый интервал1 млн лет при неизменных угловых скоростях:

| Сетка и шаг, млн лет | Принято коры, км³ | Ячейки с несколькими владельцами | Площадь меньших владельцев |
|---|---:|---:|---:|
{chr(10).join(lines)}

Уменьшение шага1→0,25 меняет принятый объём на {100*(quarter/reference-1):+.6f}%; уточнение сетки sub4→sub5 при шаге1 — на {100*(fine/reference-1):+.6f}%. Это чувствительность отдельного транспорта на коротком интервале, а не доказательство сходимости всей тектонической модели. Число смешанных ячеек включает очень малые численные хвосты схемы первого порядка; оно не является показателем физической точности.

Проверены четыре независимых бюджета **каждого material_id**: площадь, объём океанической коры, объём холодной мантии и избыточная масса. Максимальная относительная невязка равна {donor_max:.3e}. Принятые и созданные порции участвуют в одном уравнении `исходное + созданное = оставшееся + принятое`; контролируется повторное использование донора. При общем вращении всех плит рождения и потери точно нулевые. Продолжение после сериализации даёт побитовое совпадение состояния на том же расписании шагов.

Ранняя попытка `actual_frozen_v1` остановилась при уточнении сетки: несовпадение сохранённой ёмкости дочерней ячейки с её распределённой площадью породило ложную потерю порядка6,35e−10 км². Исправлено согласование дочерних ёмкостей с площадью родителя; проверки баланса и запрет самопогружения не ослаблялись. Состояние и минимальная постановка сохранены в `failure_refined_dt1`. Основные результаты находятся в `actual_frozen_v2`.

## Файлы

- `baseline.json`, `TEST_MAP.md`, `test_catalogue.json`: контроль прежнего транспорта0.5 и карта тестов.
- `actual_frozen_v2/validation.json`: результаты, остатки бюджетов, хеши исходников и исходных файлов.
- `comparison.png`: график коротких проб на русском.
- `tests/test_fractional_surface_io.py`: импорт, сохранение меньшинств и истории, контроль целостности, отказ от неподдерживаемых резервуаров, консервативное уточнение.
- `run_fractional_transport_probe.py` в корне репозитория: пользовательский CLI независимой пробы; его отдельные снимки можно продолжать через `--resume`.

Внутри смешанной ячейки пока не восстановлена геометрия желоба. Поэтому эти события нельзя безопасно передавать в существующий расчёт сил погружения или в потребителей, допускающих только одного владельца ячейки. Тепловая эволюция отдельных порций и поддержка континентов/осадков остаются следующими интеграционными задачами; их материал не отбрасывается молча.
"""
    for before,after in (("механику0.5","механику 0.5"),("состояние0.5","состояние 0.5"),
            ("на50","на 50"),("за1 ","за 1 "),("интервал1 ","интервал 1 "),
            ("шага1","шага 1"),("шаге1","шаге 1"),("порядка6","порядка 6"),
            ("транспорта0.5","транспорта 0.5")):
        text=text.replace(before,after)
    args.output.write_text(text,encoding="utf-8")
    print(json.dumps(dict(dt1_to_quarter_relative_change=quarter/reference-1,
        coarse_to_fine_relative_change=fine/reference-1,maximum_material_id_relative_residual=donor_max,
        snapshot_restart_bitwise_equal=data["snapshot_restart_bitwise_equal"]),indent=2))


if __name__=="__main__":
    main()
