# Полные исходные механические носители сил и работы

Дата: 06.10.2026 (МСК). Ветка: `fix/genesis-mantle-convection`.
Продолжение первого условия последнего блока исходного пункта 1 после
[единого учёта источника 863](GEOMETRIC_NATIVE_COMMON_SOURCE_ALLOCATION_IMPLEMENTATION.md).
Исходный план из шести пунктов не расширен.

## Статус

Компонент исходных механических носителей завершён: все 16 элементов,
включая восемь вне активных receiver supports, сохраняют площадь,
локальное состояние, полную историю и единственную линию работы.
Силы и производные следуют из одного последовательного контактно-колонного
потенциала с моментом вращающегося receiver frame.

Авторитетная попытка02 в `release_validation_02/`:
**18/18 свежих тестов PASS за 752,69 с**, pytest wrapper 753,864 с;
**68/68 native queries PASS за 481,670 с**, callback 481,712 с.
Полный typed source replay в том же процессе завершён;
единый gate **PASS за 2195,901 с**.

Независимый stdlib-аудит **PASS за 148,938 с**, с первого запуска:
обе настоящие силовые базы всех 16 элементов, 136 scalar B/H,
68 potential gradients/Hessians, force/couple/virtual-work identities,
четыре finite radial work identities, 48 состояний и 288 work-store scalars.
Все 867 исходников неизменны до/после; прежние 863, цепочка 1633 артефактов
и закрытые выпуски сохранены. Bootstrap863 PASS за 63,281 с.
Старые тесты, физические решения и сертифицированные квадратуры не повторялись.

Первая попытка сохранена как FAIL с двумя ошибками новых тестовых оракулов.
Исправлены только эти сравнения; core/control/wrapper между попытками
не менялись. Буквальные архивные значения и точная механика проверяются
раздельно с явными остатками округления; общий физический допуск не увеличен.
История попыток и точные архивы приведены ниже.

Scope отчёта: `COMMON_WHOLE_SOURCE_MECHANICAL_CARRIER_COMPONENT_ONLY`.
Scope DTO: `SOURCE_COMPLETE_MECHANICAL_OPERATORS_NO_COMMON_MATERIAL_HANDOVER`.
Это реконструкция и вариационный механический оператор исходной системы.
Материальные привязки к COMMON, полный перенос старых сил в 92 координаты,
3D концы стержней, HOT-закон, общее равновесие и новый интервал не приняты.

Frozen867 canonical:
`bfabfa58dbe49ce1102ef9ab72680a301f8cea6fb30093442150d8d4bda45ef8`;
raw `frozen_source_map_867.json`:
`4b2b98b406cf1f955260dcc24770b9a860cf330e49adc72b4d47e018a140ffc6`.
Прежние 863 Python-файла сохранены; ровно четыре добавленных пути:

- `tectonics/geometric_native_common_source_carrier.py`, SHA
  `b188348e7b927c1a1d797cf19b99c33990ba402dd3b942f0ac23cc8d6fe3b0eb`;
- `tests/test_geometric_native_common_source_carrier.py`, SHA
  `cb27a44ad54ce55d5301e5bd97d7131bca74556b4d950ced41e2275bc0ecf1a8`;
- `analysis/geometric_native_common_source_carrier_validation/native_control.py`, SHA
  `a764110e4db612001303f103a520890ec5116fbd7db3416e933c6ca3d8d34fed`;
- `analysis/geometric_native_common_source_carrier_validation/validate.py`, SHA
  `19ae91eb2617d93f62132825b4d4e79b4ba94a0c87f990cbda040ddbd4e34fce`.

## Две настоящие механические базы

Архивные `actual_candidate.physical_points` рассчитаны на исходной
замороженной геометрии `original_physical_history.accepted_before.snapshot`:
clock 0, identity poses и нулевые radial displacements. Они воспроизводятся
через прежнюю literal binary64 операцию на настоящих исходных point/frame
данных. Отдельно хранятся математические interval реакции и их точные
отличия от архивных binary64 значений. Остатки округления не скрываются.

У старого `original_reference.operators` есть отдельный путь построения:
sin/cos умножаются на горизонтальный row; новые механические производные
используют буквальные сохранённые normal/down. Точное различие включает
округление направлений и операций, а radial values ещё переводятся из
`w/R` в метры. Сравнение с допуском относительно конечной компоненты
неправильно при отмене больших моментов или почти нулевых cross products.
Тесты должны подтверждать старую binary64 запись отдельно от точного
literal-векторного оператора и сохранять их знаковую разницу.

Текущая объективная база использует `before.snapshot.owner_histories` и
настоящие candidate radial values, native clock 1 000 000 с. Контактное
состояние остаётся `unaccepted_candidate`, clock 2 000 000 с/step2.
Memory clock — 1 000 000 с/stage1. Все clocks сохраняются раздельно;
новая текущая мировая сила не выдаётся за старую архивную силу.

Сохраняются буквальные angular velocity, omega, оба duration и фактические
rotation increments. Угловая вариация измеряется в радианах, не в скорости;
точные binary64 остатки duration conversion записаны отдельно.
Исходные `q/n/down` не нормируются и не заменяются midpoint значениями.

## Объективный оператор с вращающимся frame

Оператор охватывает все 14 координат: шесть left-SO(3) углов владельцев
и восемь физических радиальных смещений балок в метрах. Старые scaled
radial columns относятся к `w/R`; при сравнении их нужно разделить на R.
Полные first/second owner derivatives определены в свежем локальном
left exponential chart. Конечные owner increments здесь отвергаются.
Конечные radial-only queries разрешены на сжатой normal ветке.

Для каждого исходного элемента:

`x_sub=(R+w_sub) Q_sub q`, `x_over=(R+w_over) Q_over q`,
`e_a=Q_over e_a_original`, `h_a=e_a dot (x_over-x_sub)`.

Это настоящие исходные bank tips. Длины колонн L не дают право объявить
их дальними 3D концами стержней или построить такие концы как `X +/- L*n`.

First variation содержит spin принимающего frame:

`B_sub_theta=-x_sub cross e_a`, `B_over_theta=+x_sub cross e_a`,
`B_sub_radial=-q_sub dot e_a`, `B_over_radial=q_over dot e_a`.

Для world traction `F=A*(t_n*n+t_t*down)` сохраняются:
`F_sub=F`, `F_over=-F`, `M_sub=x_sub cross F`, `M_over=-M_sub`.
Последний момент складывается из `x_over cross (-F)` и
carrier couple `(x_over-x_sub) cross F`. Без couple теряется объективная
виртуальная работа. Полный геометрический Hessian включает spin,
owner/owner и owner/radial second variations, а не только `K*B^T*B`.

## Одна энергия и сохранённые якоря

Исходные свободные якоря являются результатом интегрированного драйвера.
Они не заменяются конечным геометрическим `h(0)`. Сохраняется выражение:

`free(q)=stored_free+[h(q)-h(0)]`.

При самой передаче correlated difference равна нулю точно, а не разности
двух независимо округлённых интервальных значений. Для каждого носителя
объявлена одна локальная механическая модель:

`E(q)=E_source + A*(t_source dot eta + eta^T K eta/2)`,
`eta=h(q)-h(0)`.

`E_source` — точная сумма двух буквальных сохранённых контактной/колонной
энергий. Линейный term создаёт настоящую преднагрузку; постоянный архив
энергии сам по себе не создаёт restoring force. Gradient и Hessian
полностью следуют из этого потенциала, включая `t_source*H_geometric`.

Exact harmonic `K=1/(1/k_contact+L_sub/E_sub+L_over/E_over)` сохраняется
вместе с прежним binary64 source K. Остатки traction/energy округления
записаны точно. Saved source data не подгоняются под новую арифметику.
Для настоящей нулевой strength shear ветки effective tangential K равен
нулю; proposed plastic increment хранится как query receipt. Отдельный
frozen elastic predictor имеет ненулевой shear K и не является relaxed
решением. Ни один query не продвигает source memory/state/work.

Все 16 механизмов входят один раз. Непрерывная мера `C dtheta` остаётся
отдельной моделью учёта; её potential не добавляется вторым источником.
Последовательная энергия уже содержит contact и обе локальные колонны;
bare contact853 и повторный local rod store не добавляются к ней параллельно.
Сумма сохранённых энергий всех 16 механизмов — **42,857793429 млрд Дж**:
contact **14,285931143**, локальные колонны **28,571862286**.
Глобальная изгибная энергия восьми исходных балок — отдельный механизм;
её калибровка **251,841262157 млрд Дж** не входит в `E_source` этого
компонента и здесь не энергизируется. Согласование глобального изгиба
и контактно-колонных энергий/работ остаётся частью того же физического handover.
Это прямо подтверждено сохранённым `contact_energy_decomposition_audit.json`
в `analysis/geometric_native_common_contact_validation/`: raw
`b164936ac0217e4c8c4feaebc8d7495731edd218e4da93db36e1d2660a3815b0`,
selfseal `e04e97ba0345977f6e5f97c0fee167fa4dabf6cba69c51ebac81faed95a69685`.

## Восстановление и контроль

Системный и response DTO зарегистрированы; полное from-dict replay
восстанавливает весь ответ из неизменного источника. Подмены площади,
force/couple, B/H, clocks, anchors или scope не принимаются после reseal.
Системный DTO сохраняет четыре исходных DTO, saved128, all16 histories,
CURRENT memory, work ledgers и полный rate/duration conversion receipt.

Новые 18 тестов используют независимые аналитические B/H и Decimal
Rodrigues variations. Проверяются архив/current epochs, receiver couple,
objectivity, точная энергия, stiffness residuals, inactive restoring
force, rebasing, обе shear ветки и отказы при подмене данных.

Native control формирует 64 baseline responses: все 16 узлов в двух
настоящих базах и двух явно объявленных ветках. Четыре дополнительные
current radial-only queries относятся к одному active и одному inactive
исходному узлу в обеих ветках; их delta_sub равна точному `1e-6 m`.
Всего 68 responses, затем полное typed source replay. Geometry/reference
проверяются в том же процессе после свежих imports. Старые тесты,
силы/Hessian и сертифицированные квадратуры повторно не запускаются.

В родительском validation-каталоге, вне четырёх выходов:

- `baseline_863_bootstrap_source.txt`, raw
  `eafbd34a81e9a6ee1f6d06c368797e4cec6291f445c8efc5e024ac16ffda4572`;
- `baseline_863_bootstrap.json`, raw
  `bfd7d18e5f407083ade3aea48e7ed183952c00b01d7ad66ad2cbbe2a09811732`,
  selfseal `a109a6dfce8999aa2fd6ec6bc19f0cb3ab9d5fd4adce29703e2889d16182c0da`;
- reusable baseline producer, raw
  `2f9168e7691a6ad4cd56633323f60895abd71f502beb553b8a239f0e66c5a46c`.

Reusable допускает ровно четыре заявленных новых Python-пути, проверяя
весь source map 867 до/после и все 863 прежних SHA непосредственно.
Для readback старого859 producer используется явно раскрытая frozen863
проекция; полный867 map проверяется независимо, новые paths не скрываются.

Неудачная попытка01 сохранена без перезаписи: 16 PASS / 2 FAIL за
750,39 с, pytest wrapper 751,081 с, gate 781,029 с. Native control
в этой попытке не запускался. В момент отказа source867 и вся старая
цепочка были неизменны. `attempt_01_preservation.json`
raw `274d44b52d8b9a03a38dd089bb4cb04d241da15a886b06f5d2cd8a56a80aa52a`,
selfseal `aa2108c86b6f3381c779ab0a874f12f1cbdd8cb46bcf3d9cee2e77d2ed734207`.
Она содержит SHA девяти сохранённых файлов, включая все четыре исходника.
Её исходная frozen867 canonical —
`02bcf5c83b5157432a40157870fbed98bcf14a078ed97dd317847abc6d1c6c68`;
исходный test SHA —
`f1a5eaaacc7606486336b7bd4f30c0c0821c4ea67ad4b1521ee31c3c4eaae1c4`.
Core/control/wrapper между попытками не менялись.

Предварительный guard `attempt_02_test_oracle_guard.json` raw
`69aae00e178d1370a48c6805a5010ddc60f584400f9222b658a23c47b6d1b6ef`,
selfseal `8712947c9353a81646a42addbbb533e2b2fac85412a49e6423635ed48a4b421a`.
Exact script TXT raw
`6733f1e9b5195422921f7d17f87cafa2d04ef6102796501cfe0aea86b8d8429d`;
console TXT raw
`45cebc522794994aa6734b68c5552f19af3975391bb88267ccec063299c08607`.
Это чистая проверка source literals и извлечённых функций, с настоящим
interval helper; не полный binder, свежий pytest или независимый stdlib-аудит.

## Закрытый выпуск и независимый аудит

В `analysis/geometric_native_common_source_carrier_validation/release_validation_02/`
строго четыре выхода плюс sealed wrapper. Независимые receipts лежат
в родительском каталоге.

| Файл | Raw SHA256 |
| --- | --- |
| `report.json` | `47b5c882c041f5a2c6d460c71eae11091a6217ff78a0df14af9b57f214ba633e` |
| `native_control/report.json` | `be3f45ed3c5b12d3216c3428402f3ed0b4ebae4bc3518ef40dc9cfcbb9aa57ac` |
| `native_control/source_packet.json` | `265206430b4c864db32bce1be0a9a75b71fa180158f3d81e4f52131bebdf320c` |
| `pytest.log` | `f31e0bc936a532a6fb10cce40b932712fae1134f94b91046644384b379230576` |
| `control_console.log` | `6d50275a24b0ab59921454b449f895ca1c08364128c93e1de3680851156ff3ae` |

Wrapper selfseal:
`d3bf099047fe0b995ba06360789fb29a998a0ba71e7d4042941963aeca7ce9e0`;
native selfseal:
`57bbe7b66b0d21b585b768064580a21ebcefdca7170a86282b1b86f6e20fc952`;
packet selfseal:
`a46fe79aeabef70ec01672ad4c33c737bfc06d5323a112edcfeb2616f7b148a3`.
Packet: 86 487 519 байт. Complete carrier binding:
`bd82b3388005598f8f4981d97f56d4751793ac9a5069ce9c13a5479db39a8fdf`.

`independent_audit.json` raw
`a73bc310e25e368d0dc99ebe968602002609885b413224a4e4675888eb02b3d5`,
selfseal `9313431f64093ccda637e48a0e51da1adbb930dd036724be9799f02d1876062e`.
Exact `independent_audit_source.txt` raw
`3e3226705f7a1931e01d9147046686492e73d728d5a7b33a01d1e4555eaa43f4`;
`independent_audit_console.log` raw
`f586d1fca4de980a814e98f43c7dc182b68854481a5b8b788aac46320c1b3b0a`.
Перед заморозкой02 в auditor изменены только два map pins;
предыдущий exact producer сохранён в `independent_audit_prefreeze01_source.txt`,
raw `13306b11a95c275e6783bbf9825fb9573ac0c09821331e832cd50af16ab319bc`.

`baseline_after_gate.json` raw
`997dd75c49334e3a0c57e3a378818c1093edb07a1e461f6fc9270aafddd35aa2`,
selfseal `293bbbc03c53650ac32fe9fbd2b57702f15dddc8c0477d4eec2eaf7d4e2e3755`,
32,795 с. Независимый `independent_audit_baseline_863_readback.json` raw
`82f7b4205bef515af701835604b3d178cf432feabab7796cf1164ac9326328f0`,
selfseal `ad127a4f978478b46e0a70400d6a605f3ecdd4bbe4bf5eb127cc5a9b98416011`,
95,289 с. Последний выполнен в начале независимого аудита, после gate PASS;
после новых формул auditor ещё раз проверил source867, зависимости
и strict4 outputs. Второго readback или solver не запускал.

[Манифест 24 файлов](analysis/geometric_native_common_source_carrier_validation/artifact_manifest.json)
raw `e0ebc4fbd1086285035fc868181f3ba7cb34833df897d8c576537b71716a91bb`,
selfseal `503525b0c461b4e905844aff4d0178536b763446b56bc6c2b4a4a44d3f7f9160`.
При неизменных источниках проверять raw/selfseals и stdlib readback;
старые физические расчёты, тесты и квадратуры не повторять.
Воспроизводимая проверка текущего компонента из корня репозитория:

```bash
.venv/bin/python analysis/geometric_native_common_source_carrier_validation/independent_audit_source.txt \
  --release analysis/geometric_native_common_source_carrier_validation/release_validation_02 \
  --baseline analysis/geometric_native_common_source_carrier_validation/baseline_863_reusable_source.txt \
  --out /tmp/native_source_carrier_readback.json
```

## Остаток исходного пункта 1

**Два из трёх прежних блоков завершены; остаётся один блок с тремя условиями.**
Реконструкция механического источника закрывает часть первого условия.
Следующая конкретная реализация — аутентифицированные материальные
привязки carrier к COMMON и work-preserving embedding `B_new*T=B_old`
в общие 92 координаты, включая consistent second variation/potential.
Узлы active/inactive являются source angular buckets, не target labels.
OLD incoming-donor archived/retained labels не заменяют эту привязку.
Неактивный носитель с живой преднагрузкой требует реального механического
драйвера либо явного перехода материала/силы/работы в другой subsystem.

Read-only контракт следующего шага сохранён в
`analysis/geometric_native_common_source_carrier_validation/physical_common_attachment_design_source.txt`,
SHA `a34d597bbafb6664848d6ab663bfcf043ab6820552de395e843f1a1ed2ae866f`.
Точный join всех 16 IDs подтверждает препятствие прямому переносу:
активные восемь старых donor labels уже архивированы; остальные восемь
сохранены, но их receiver labels находятся вне настоящих текущих cuts.
Inverse chart сам по себе не подтверждает принадлежность материалу.

Архивированный материал не уничтожен: source сохраняет три original
archive cohorts. В существующей COMMON-системе восемь original_live
bodies и пять hot_birth bodies. Последние — новый материал с другими IDs,
а 16 causal first-event branches не дают старому archived q новой HOT-карты.
Даже совпадение original material ID с оставшейся живой частью не
подтверждает принадлежность архивированной точки этой части.
Существующие 13 bodies/92 coordinates сами по себе не охватывают
динамику архивов.

Поэтому следующая реализация должна явно объявлять correspondence,
проверять обе настоящие material masks и parameter core. Для архивов
нужна подлинная карта того же материала/внутренней подсистемы либо
объявленный переход к новым labels с материальным бюджетом и работой.
Остальные носители также требуют настоящих bulk attachments внутреннего
механизма, если их исходная пара отсутствует на текущем face cut.
Reduced rigid lever driver может сохранять исходные virtual tips и
owner moments без выдуманных дальних концов колонн. В настоящем
material-local chart с transfer pose `Q_side0`:
`X_ref=Q_side0^T*X_attachment_world0`,
`a_side=Q_side0^T*(p_source_basis0-X_attachment_world0)`,
`y_side=Q_side*(X_ref+u_material+a_side)`,
`e_ref=Q_over0^T*e_source_basis0`,
`h_new=(Q_over*e_ref) dot (y_over-y_sub)`.
При переносе сокращения `Q0*Q0^T` должны быть символически связанными,
а не произведениями независимых интервалов или midpoint rotations.
Источник `p/e/B/H/world reactions` относится к одной явно выбранной базе.
Для сохранения фактических архивных сил нужна `archival_old_reference`
clock0 с сохранёнными binary64 остатками. Выбор `current_objective`
требует отдельного доказательства объективного переноса сил и работы,
которое здесь ещё не получено.
Силы и дополнительные моменты должны следовать из того же потенциала.
Это объявленная closure, требующая глобального доказательства `T`,
вторых производных полного embedding и связи с body/modal coordinates;
она сама по себе не доказывает original-material принадлежность,
физический перенос или равновесие. Это уточнение того же первого
условия, а не дополнительный этап исходного плана.

В том же первом условии остаются HOT law/mean chart, нагрузки/drag и
fresh mixed equilibrium. Затем прежние два условия: whole-clock all-pair
admission/reverse C5→C1 и принятое продолжение с независимой работой,
балансами и joint restart. **1 прежний принятый интервал / 0 новых**.
Пункт 1 остаётся открытым.
