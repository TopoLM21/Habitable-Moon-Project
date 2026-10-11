# Трёхмерные carriers исходного материала и трёх архивных когорт

Завершённый компонентный выпуск887, 07.10.2026 (UTC), ветка
`fix/genesis-mantle-convection`. Scope:
`DECLARED_GLOBAL_PAIRED_THREE_DIMENSIONAL_ARCHIVE_CARRIER_ONLY`.

Пункт1 исходного плана остаётся открыт: один существенный блок с прежними
тремя условиями. Эта реализация закрывает внутри первого условия явную
геометрию и закон движения целых исходных live/archive объёмов, включая
трёх непустых архивных родителей octant:5,6,7. Перенос сил на новую геометрию
и новое смешанное равновесие не получены.

Модель задаёт новое совместное размещение: исходный донор находится ниже
радиуса R, принимающий исходный материал — в верхнем слое с основанием R.
Это объявление новой reference geometry, не восстановление старых roof,
полной контактной грани, prestress или работы подготовки.

Подлинная глубина z является areal-volume координатой: r0^3=R^3-3R^2 z,
а dV_ref=R^2 dOmega dz. Поэтому сохраняется объём R^2 H на стерадиан,
а не ошибочно присваивается геометрическая толщина R-z. Для верхнего слоя
r_upper^3=R^3+3R^2(H-z); его объёмный якобиан ровно1.

Для всех исходных донорских меток используется одна карта. В receiving
frame u=Q_receiver(t)^T Q_donor(t)q, g=-u_x, а
beta(q,t)= integral_0^t max(g(s,q),0)^3 ds / (6 Gamma^2).
Радиус донорского столбца r=r0 exp(-tan(delta) beta), направление смещается
вниз по dip на угол beta. Literal dip и полная единственная сохранённая
история каждого owner аутентифицированы; скрытое последующее движение
не предполагается. Нормированный t относится к подлинному journal.

Вход включает дополнительную скорость непрерывно; beta не убывает при
последующем выходе или повторном входе. Формула не дифференцирует singular
first-entry root и не выбирает его midpoint. Точные границы окон событий
используются лишь для вычислительного разбиения интеграла.

Получены положение, скорость, полный Cartesian F относительно исходного
X=r0q, объёмный J и multiplier плотности1/J. Source amounts и коррелированные
live+archive резервуары, IDs, material fields, ages и first receiver ancestry
сохранены. Донорская карта сжимаема; изменение физического объёма учитывается
через плотность, не объявляется потерей материала.

Глобальная оценка использует исходный donor hemisphere и полный bound Gamma:
beta<=Gamma/6, 1+beta_mu>=1/2,
J>=exp(-tan(delta)*Gamma/2)*cos(2Gamma)/2>0.
Монотонность latitude и radius доказывает injectivity всего donor volume.
Единая карта склеивает соседние исходные участки каждого owner. Исходная
полная cross-bank fault face этим размещением не восстанавливается.
Архив и live donor разделены подлинными implicit material domains;
принимающий слой имеет radial interior>R, донорский radial support<=R.
Это компонентная геометрическая оценка, не допуск всех физических пар.

HOT point membership/birth chronology пока не подключены: public HOT query
возвращает отказ. Формула верхнего слоя сама по себе не аутентифицирует HOT.
56 зарегистрированных cold coordinates, прежние virtual source attachments,
HOT law, current follower contact law, нагрузки/drag и mixed equilibrium
ещё требуют физической сборки на новой reference geometry.

Старые force solve, тесты и tensor/contact quadrature не повторяются. Public
DTO restoration повторно проверяет подлинную прежнюю геометрию и хроники;
это фактическая причина заметной стоимости initial loading. Старые files
и receipts остаются неизменными.

Дальше внутри того же первого условия: связать32 исходные материальные
метки и единый потенциал16 sources+8 beams+8 remainders с новой физической
геометрией, переносом prestress и работы; затем HOT/mean, нагрузки/drag и
новое смешанное равновесие. После этого остаются прежние допуск всех пар
на всём интервале включая C5->C1, затем принятое продолжение с работой,
балансами и joint restart. Принят1 прежний и0 новых интервалов; пункты2-6
плана открыты. Остаток существенный; план не расширен.

Фактическая frozen попытка02: **18/18 тестов PASS за354,90с**; wrapper
измерил355.174099768с. Native контроль **80/80 PASS**
за147.526633346с:16 материальных лучей по5 глубинных запросов,
включая полный `[0,H]`, плюс ordered observer и полный model/point restart.
Same-process gate **PASS за519.633902382с**, exit0; все5 outputs
проверены. Источник887 заморожен до запуска, четыре фактически исполненных
источника сохранены как TXT snapshots. Независимый аудит **PASS
за10.833255191с**:81 запрос,1935 численных включений,420 decimaldigits,
8 родителей/3 когорты/4 literal dip/32 once-only резервуара и7 source-root
brackets. Он не импортирует project modules, тесты или старый solver.

Первая frozen попытка остановилась ДО тестов: системный Python не имел
pytest. Её FAIL/actual stdout/source snapshots и BEFORE/AFTER883 PASS
сохранены. Затем использована имеющаяся `.venv` без установки зависимостей;
исправлен только новый wrapper runtime metadata. Физические core/test/control
между этими frozen попытками не менялись. Developmental interruption после
9 PASS сохранён отдельно и не считается итоговым тестовым результатом.

Свежий initial883 reader PASS15,117036754с; восстановление TMP не требовалось.
Before/after frozen02 guard подтвердил все883 прежних Python,341 raw pins,
manifest56 и старый финальный архив6,15 TMP associations и11 ancestry closures.
Прежний879 manifest83 и его архив6 тоже сохранены. Wrapper raw output-map
содержит3 строки из-за basename-фильтра report.json: nested control report
связан отдельным selfseal, а manifest и final reader проверяют все5 outputs.
Frozen wrapper/receipt после запуска не переписывались.

Новый manifest содержит **41 фактически обнаруженных файлов**.
Raw `a1c8f99b579db19e8d8fea03d5b7a5a6ffa389bcb0cecca19e8c597fdcde56c3`,
selfseal `227814a215161ff2f615338e8eb273f10f40c46bbd61d31424a6398d5b8cfb57`.
Frozen887 canonical `f5220449a37364621723b4f97adb5c01efdaeb62afff11bd774c12caf9c289ad`,
raw `c8d66ce04fac77c486bf6ee3971205508435063e40032d5310e0a419f6d534a9`.
Wrapper selfseal `8dea3477beb94dfd0eab1b22913ee8b0b20092b04b2d4fade58166fb452eeb2e`;
independent audit selfseal `684a8e58775cec9ba2ca1fad16e2104982032093aabd9196eae1b6227d228d25`.

Выпуск: [контрольный gate](analysis/geometric_native_archive_material_carrier_validation/release_validation_02/report.json),
[полный packet](analysis/geometric_native_archive_material_carrier_validation/release_validation_02/native_control/source_packet.json),
[независимый аудит](analysis/geometric_native_archive_material_carrier_validation/independent_audit/report.json),
[manifest](analysis/geometric_native_archive_material_carrier_validation/artifact_manifest.json).
Финальные result/time/docraw определяются отдельным
[readback](analysis/geometric_native_archive_material_carrier_final_readback/readback.json)
и его [четырёхфайловым архивом](analysis/geometric_native_archive_material_carrier_final_readback/archive.json).
Документы записываются перед этим чтением; после него они не меняются.

Для повторного evidence-only чтения не запускать закрытый gate: он повторит
новую математику. Использовать final reader с НОВЫМ output ВНЕ репозитория:

```bash
python analysis/geometric_native_archive_material_carrier_final_readback/readback_source.txt --output /workspace/archive_carrier_evidence_recheck_20261007.json
```

Reader не исполняет tests/solver/quad; старые физические metadata читает как
сохранённые receipts. Он проверяет actual887, все raw pins/closures и документы
до/после отдельно от41-file component manifest. Его новый prior receipt
создаётся вне repo с уникальным именем от полного output path.

После перезапуска среды вернуть четыре initial baseline файла из
`analysis/geometric_native_archive_material_carrier_validation/baseline_883_*`
в `/workspace/native_archive_material_carrier_baseline_20261007/`, сохраняя
точные bytes. Для15 отсутствующих TMP aliases использовать только их
`permanent_copy` и SHA из `baseline_883_readback.json` поля
`temporary_dependency_exact_closed_copy_associations`; неверные существующие
копии не подменять. Исторические receipts/stdout не воссоздавать.

Часы прежнего закрытого COMMON: source geometry0/contact2e6/memory1e6/
material geometry1e6, сохранены раздельно. Normalized journal clock новой
карты не переименовывает эти часы; accepted interval, contact/memory/work
ledger или topology не продвигаются.
