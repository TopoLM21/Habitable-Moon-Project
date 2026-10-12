# Sidecar неприменённых обменов BE50

Создан `be50_unapplied_sidecar.json` для двух сохранённых интервалов
0,8684→0,8685 Myr. Есть SHA256 всех трёх входов, времена, integrator/scheme,
единицы Дж/Myr, отдельные записи каждого интервала и всего окна.
Получатели/доноры не назначены; `applied_to_thermal_state=false`,
`historical_opening_balance=null`. Это не запас тепла и не рассчитанный нагрев.

За окно: физическая диссипация 1,931926e18 Дж, numerical sink 3,317833e16 Дж.
Они находятся в разных полях. Встречная термомеханическая запись −4,292947e17
Дж только pending. Damage, Kbirth и работа traction имеют самостоятельные
pending-записи; общей суммы «нагрева» нет (`combined_heating_j=null`).
Неизвестное прошлое не заменяется нулём и не становится начальной энергией.

Валидатор заново выводит sidecar из закреплённых SHA входов. Проверяет
схему/шаг BE50, цепочку, времена, знаки, конечность, неотрицательность sinks,
суммы интервалов, seal и полное совпадение с источником. Изменение unknown,
единиц, смешивание physical/numerical или включение тепла отвергается даже
после пересчёта seal. Существующие файлы не перезаписываются. 28 тестов PASS.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
/workspace/habitable-moon-venv/bin/python -m analysis.genesis_energy_inventory.sidecar \
  --verify analysis/genesis_energy_inventory/be50_unapplied_sidecar.json
```

Создание другой копии: вместо `--verify` передать `--output NEW_PATH`.
Никакой интеграции или изменения checkpoint не производится.

Схема общего теплового coupling остаётся открытой. [Компактное предложение
архитектуры](ARCHITECTURE_PROPOSAL.md) подготовлено, но его реализация зависит
от ответа пользователя. После этого результата дополнительные расчёты
в ожидании ответа не запускаются.
