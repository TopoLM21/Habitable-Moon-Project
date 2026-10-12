# Явный рабочий runtime: midpoint / BE50 / BE25

`VolumeSlipModel.configured_runtime(state)` сохраняет midpoint по умолчанию;
существующие constructor, `step`, физические параметры и legacy checkpoint не
меняются. Новый headless CLI — `run_genesis_volume.py`. BE50 — инженерный
кандидат (50 лет), BE25 — контроль (25 лет), не сертификаты физической точности.
GUI и основной зрелый цикл этим не объявляются завершёнными.

## Запуск в восстановленной облачной среде

Из корня репозитория, без нового механического шага:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
/workspace/habitable-moon-venv/bin/python run_genesis_volume.py inspect \
  --input /workspace/genesis-volume-runtime-validation-01/be50.json \
  --integrator be50
```

Сохранённая цепочка упаковывается явно (новый путь output обязателен):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
/workspace/habitable-moon-venv/bin/python run_genesis_volume.py pack \
  --input analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json \
  --integrator be50 \
  --chain /workspace/genesis-be-time-control-01/50/step_1.json /workspace/genesis-be-time-control-01/50/step_2.json \
  --output /workspace/be50-runtime.json
```

Будущее явно разрешённое продолжение (в этой проверке **не запускалось**):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
/workspace/habitable-moon-venv/bin/python run_genesis_volume.py advance \
  --input /workspace/genesis-volume-runtime-validation-01/be50.json \
  --config configs/genesis_volume_be50.json --output /workspace/be50-next.json
```

Для BE25 выбираются `be25` и соответствующий config/bundle. Без выбора режима
CLI ожидает midpoint и отвергает BE bundle. Переключение BE50↔BE25 внутри
журнала запрещено. Физические параметры через config не переопределяются.
`--steps` ограничен 1..4, wall budget 30..600 с, память адресного пространства
4 GiB; по умолчанию один шаг / 300 с. Истечение бюджета требует разбирательства,
а не автоматического повторного расчёта. Существующие файлы не перезаписываются.

## Границы допуска

BE принимает только точный корень 0,8684 Myr с SHA256
`62579a0bad128830a66e2f44579cf2b1f5d3835b691225b8d431102f3e4486d7`
или его собственный bundle. Bundle содержит исходные байты корня, явные
integrator/dt/role, каждый журнал и SHA256. Legacy midpoint остаётся в старом
формате с прежней кинематикой; описание runtime явно называет midpoint.
Нет неявного rebasing, сброса накопленной энергии или смешения журналов.

Каждый BE шаг проходит независимые аудит и восстановление всей цепочки до
присваивания нового состояния. Gate останавливает расчёт перед активацией
разлома; remelt не поддерживается. Runtime использует проверенный BE kernel
из `analysis`, не является отдельным новым физическим решателем. Полный
повторный аудит растёт с длиной истории: это ограниченный рабочий драйвер,
а не оптимизированная длинная планетная эволюция.

Численные потери памяти и кинетики — отдельные диагностические sinks, они
никогда не добавляются к теплу. Их отношение к внешней работе больше единицы
само по себе не дефект: баланс также включает высвобождение накопленной
упругой/кинетической энергии, термомеханическую работу, birthflux и damage.
Полный тепловой резервуар этой проверкой не замкнут.

## Проверка без новой эволюции

`python -m analysis.genesis_volume_runtime_validation.run --output NEW_DIRECTORY`
загружает принятый midpoint и уже существующие 2×50/4×25 цепочки, проверяет
независимые аудиты, типизированные рестарты, побайтовое повторное сохранение,
ошибочные integrator и неизменность тепловых payload. Новых Genesis шагов нет.
Локальные крупные bundle остаются в `/workspace`; краткий результат и хеши —
`summary.json`. Тесты CLI включают mocked routing: полноценный новый advance
в этом этапе не выполнялся, использованы ранее проверенные численные цепочки.

Результат: **PASS**, 156 тестов; реальный CLI `inspect be50` PASS.
Проверка сохранённых состояний заняла 444,87 с, пик RSS 1 375 172 KiB
(≈1,31 GiB); midpoint и оба BE restart побайтово совпали. BE50: 960 когорт /
2 интервала; BE25: 1120 / 4, оба до 0,8685 Myr. Gates закрыты, тепловые
payload неизменны, принятый корень побайтово сохранён.

| Источник/сток, Дж | BE50 | BE25 |
|---|---:|---:|
| Высвобождение упругой энергии | 1,535819e18 | 1,542625e18 |
| Термомеханическая работа | 4,292947e17 | 4,182905e17 |
| Внешняя работа | 2,470609e15 | 2,460730e15 |
| Изменение энергии damage | −2,479532e15 | −2,928320e15 |
| Физические потери | 1,931926e18 | 1,943740e18 |
| Численные потери | 3,317833e16 | 1,670877e16 |
| Численные / высвобождение упругой энергии | 2,1603% | 1,0831% |

Баланс дополнительно включает ≈5014,83 Дж высвобождения кинетики и birthflux
55,61/27,81 Дж. Эти числа — разложение сохранённого суммарного журнала,
прошедшего независимые аудиты; термомеханическая работа не является
доказательством замкнутого глобального теплового резервуара.
