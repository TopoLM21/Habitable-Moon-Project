# Явная опорная мера и последовательная контактно-стержневая цепочка

Дата: 05.10.2026 UTC. Ветка: `fix/genesis-mantle-convection`.

Продолжение первого условия последнего блока исходного пункта 1 после
[локального контактного закона 853](GEOMETRIC_NATIVE_COMMON_CONTACT_FORCE_IMPLEMENTATION.md).
План из шести пунктов не расширен.

## Статус

Реализованы две явно объявляемые части модели: нормированное распределение
исходной непрерывной опорной площади по настоящей глубине грани и полная
последовательная цепочка contact/sub-column/over-column. Пять чистых проверок
меры **PASS за 5,03 с**, две независимые проверки Schur complement **PASS за
4,27 с**. Свежие **21/21 тестов PASS за 438,92 с**. Единый gate **PASS за
2038,478 с**; native control **827,348 с**, callback **827,535 с**.
Полный source replay 20 area/state cases / 40 series responses / 16 000
Hessian entries завершён. Авторитетный выпуск:
`analysis/geometric_native_common_contact_transfer_validation/release_validation_01/`.
Независимый stdlib-аудит **PASS за 101,498 с**: площадь, три части энергии,
800 residual entries и все 16 000 Hessian entries проверены отдельным кодом.

Scope: `COMMON_REFERENCE_AREA_SERIAL_CARRIER_COMPONENT_ONLY`.
Полный перенос всех старых атомов, пространственных стержней и мировых сил,
совместное равновесие и новый физический интервал не приняты.

Ровно шесть добавленных Python-файлов:

- `tectonics/geometric_native_common_contact_measure.py`;
- `tectonics/geometric_native_common_contact_series.py`;
- `tests/test_geometric_native_common_contact_measure.py`;
- `tests/test_geometric_native_common_contact_series.py`;
- `analysis/geometric_native_common_contact_transfer_validation/native_control.py`;
- `analysis/geometric_native_common_contact_transfer_validation/validate.py`.

Frozen859 canonical:
`88f0d2e78a81712458beb61f21bbdfe0b7222467a0e3241e23757e3fac6bda7d`;
raw `frozen_source_map_859.json`:
`7351d27e3fe0bb198978badec9f173766f8522940073b057c0b9584c75392ce5`.
Прежние 853 Python-файла неизменны. Их source readback и цепочки
1633 артефактов/выпусков 826/833/841/847/853 проверены за **49,838 с**
без повторения тестов, арифметики сил или квадратур.

## Контрольный выпуск

Выпуск выполнен в одном свежем Python process: только два новых test modules,
затем native callback и полный authentic source replay. Source map совпадает
до/после; changed paths отсутствуют. Прежние 853 источника, цепочка 1633
артефактов и закрытые выпуски 826/833/841/847/853 сохранены. Исторические
тесты и новая объёмная квадратура не запускались. Pytest wrapper — 440,168 с.

Закрытый каталог содержит ровно четыре выхода плюс обёртку:

| Файл | Байты | Raw SHA-256 |
|---|---:|---|
| `report.json` | 5548 | `1b5e1695f2ddfa8ea879619e91cde6b4cfb2c7c7c89e34db04148d8765f7f9ae` |
| `pytest.log` | 111 | `0ec04bec76b949191ec6dce7091db04e361a332a2db98962d6712910b608ccf7` |
| `control_console.log` | 506 | `2baef486b5daaada1d843a5024f0f189101412d6e8efb3bd5fcc111bfca7b9ae` |
| `native_control/report.json` | 2445 | `aa5e2482cac4108c984b679a076fff0121125b962070cc91cc4530e105fe9169` |
| `native_control/source_packet.json` | 104157487 | `351482ce78e9d28634f1115a849f9d9b2a962f84b725075b898ad89824d42086` |

Wrapper selfseal:
`91bb5b263179df4e64ec24048bc56bd24f82ac63926024a47e012a28c4086ff3`;
native selfseal:
`bca96d744ce0fcfc6716e758b7705a1ce19af7f9d2dd96b64847308bab86b527`;
source packet selfseal:
`2ba9e314cbb1b4519d6a8143e6d99bec32f9d61462fe8605380568d710d26ba4`.

Первый запуск независимого аудитора остановился на его лишней предпосылке
`depth_span==10000`. Истинные пересечения дают около 9999,9 м из-за
смещённых source roof. Модель уже использует точный cube-derived span.
Исправлена только проверка аудитора: проверяются точный аутентифицированный
диапазон и его положительность. Источники 859 и закрытый выпуск не менялись.
Сохранены `independent_audit_attempt01_source.txt`,
`independent_formula_helper_attempt01_source.txt` и
`independent_audit_attempt01_failure.log` (raw
`f0e7ea988a2c864cb3b1f6cb19d87bb3d0d73737418e9b7bfd5f06bd5ae86439`).

Итоговый аудит независимо проверил 20 current source observations,
200 literal p/c coefficients, 80 harmonic enclosures, 20 projected-angle
производных/плотностей, 80 exact source rectangle CDF budgets, 40 serial
responses, 240 current/reference energy components и 40 weighted integrands.
Ни project imports, ни повторных pytest/квадратур не выполнялось.
Старый stdlib observer применяется к заявленной query precision 96 через
явный scratch adapter; прежние project sources не изменены.

В родительском validation-каталоге, вне четырёх выходов, сохранены:

- `independent_audit.json`, 5878 байт, raw
  `74195affc025d268eb2190ba496c01cfc411b8f49ab7eecc20bdc05f0ad790bc`,
  selfseal `6ed5ec34be105d8b383581060573fc96ee32636510c016498ef7ab7e607093d3`;
- точный `independent_audit_source.txt`, raw
  `f7eb5f730d6fca0b1fdbab075df71d91ebed7f1225278ec9a1bbf63d319f51d3`;
- обязательный `independent_formula_helper_source.txt`, raw
  `0ecbb318a302b5ae3acd387aca079ec42602a9a747d8232d4067c0d96b07ab6f`.

Обязательные зависимости аудита также зафиксированы: текущие baseline853
JSON/source имеют raw `c6d9702e0684e48d88d30b3361d942116ad484f1211aed4f0bf9f32cfcdf9750` /
`c92d970886f7bccdb0b570e605424c1eab5435638c8737525c360ac35b0bd748`;
source-theta design JSON/source —
`0eb62b8a9bd6b24a154b1764f61f78492364b682e9b1f1ba5b3dc247f1645396` /
`374f620725cd2b396c2c9fd981e5477b14bdc4470633820760c06481561616f9`.
Прежние producer/helper853 остаются в contact-force validation-каталоге
с raw `8d755b719156876a87cb6f51b60221159d8d96e3dd92688e0b696f4bc12ea9e8` /
`061505c158ae8026549788c242fbda541c0718526d61c452f07e9f32d24d50cd`.

Воспроизведение независимого readback из корня repository:

```sh
.venv/bin/python analysis/geometric_native_common_contact_transfer_validation/independent_audit_source.txt \
  --release analysis/geometric_native_common_contact_transfer_validation/release_validation_01 \
  --helper analysis/geometric_native_common_contact_transfer_validation/independent_formula_helper_source.txt \
  --out /tmp/native_common_contact_transfer_859_replay_audit.json
```

## Явная модель опорной площади

Источник задаёт непрерывную меру `dA0=C dtheta_comp`, где
`C=R*source_height/source_literal_frame_sine`. Все числа берутся из
аутентифицированного источника. Синус — сохранённое число original frame,
а не новый `sin(dip)` или нормаль другой грани.

Новая декларация задаёт условный профиль глубины `f(zeta)` на `[0,1]`:
неотрицательные точные Bernstein coefficients, интеграл строго равен 1.
В контроле используются uniform `(1,)` и linear `(0,2)`. Это объявление
новой модели: источник не содержит старого трёхмерного профиля памяти.

`tau=(R^3-r^3)/(3R^2)`; полный диапазон берётся из настоящего пересечения
радиусов обоих тел, без восстановления высоты 1000 м из старой пружины.
`zeta=(tau-tau_low)/(tau_high-tau_low)`.

Опорная плотность на координатах истинной грани:

`dA0/(ds dtau)=C*abs(dtheta_comp/ds)*f(zeta)/(tau_high-tau_low)`.

Угловая производная вычисляется из genuine receiving projection `(x,y)`
и её производной `(x',y')`: `(x*y'-y*x')/(x^2+y^2)`, затем применяется
буквальное computational extent / native half-pi. Сингулярная проекция
и неразрешённый знак производной отвергаются. Никакой midpoint не вводится.

Настоящая физическая coarea `J ds dtau` остаётся отдельной величиной.
Хранится отношение `dA0/dAphysical`; энергия умножается на опорную плотность
один раз. `J` не применяется вторым множителем. Плотность зафиксирована
в материальном reference, поэтому не получает новых модовых/owner производных.

`source_reference_rectangle_area` точно интегрирует Bernstein polynomial
по объявленному source rectangle. Полный бюджет, две части глубины и
внутренний прямоугольник сохраняются раздельно. Они не назначают весь
source family каждому cut и не являются доказательством target support.

Все current p/c coefficients, их provenance и разные source clocks
сохранены. Память наблюдается из actual `step.memory`; модовые пробы её
не обновляют. HOT-кривая не имеет старого receiving law и отвергается.

## Последовательная цепочка и силы

Источник задаёт две 1000-метровые нормальные колонны с буквальными E/G.
Комбинированная column stiffness уже включает их последовательное соединение.
Новый API требует явного anchor interpretation:
`common_body_trace_increment_drives_eliminated_serial_carrier`.
COMMON body trace теперь объявляется приводом исключённых внутренних
пружинных степеней свободы. Он не выдаётся за найденную 3D точку стержня.

Точное нормальное `K=1/(1/kn_contact+Ls/Es+Lo/Eo)`. Для shear аналогично
используются kt/G. Округлённые binary64 source coefficients и разности с
новым точным гармоническим K сохраняются. Reference anchor `-P_source/K`
также хранится отдельно от source free gap с его остатком округления.

Приращение genuine objective scalar trace обозначено `xi`:

`t_normal=-P_source+K*xi_normal`.

При нулевом xi давление сохраняется точно. Энергия меняется вместе с
деформацией и разделяется на:

`U_contact=t_normal^2/(2kn)`,
`U_sub=t_normal^2*Ls/(2Es)`,
`U_over=t_normal^2*Lo/(2Eo)`.

Это нормальные части энергии. В frozen-memory predictor к каждой
добавляется последовательная shear-часть с `kt_contact`, `Ls/Gs` и `Lo/Go`
соответственно; в релаксированной ветви эти части равны нулю.

Frozen-memory predictor содержит последовательный shear K; релаксированная
ветвь исходного zero-strength law имеет t_shear=0 и нулевой shear tangent.
Пластическая память только предлагается, не применяется. Раскрытие отвергается.

Полные 20 координат: cold7 + cold7 + six owner coordinates. Total residual
`t_a*xi_a,i`; Hessian `K_a*xi_a,i*xi_a,j+t_a*xi_a,ij` содержит все
receiver-frame, symmetric owner/owner, owner/body и preload terms.
Это замена bare contact853 при выбранной новой модели, а не параллельное
добавление rod energy: tangent равен K, не kn_contact+k_column.

Контроль формирует 20 area/state cases, 40 serial responses, все 16 000
Hessian entries и 40 weighted local integrands. Они являются плотностями
на `ds dtau`, не конечными силами одного узла и не новой общей 92-coordinate
силовой сборкой. Полные DTO/reference/cuts/history сохраняются один раз;
все serialized responses проходят authentic replay.

## Исходный учёт 16 узлов для следующего переноса

Отдельная pure stdlib проверка **PASS за 2,872 с** классифицирует буквальные
computational angles по настоящим receiver supports без midpoint и без
назначения новой площади/материальной точки. C9→C8 long содержит четыре
исходных узла, C10→C11 long — четыре; Oct1/C7 и Oct0 оставляют восемь
вне этих supports. Неопределённых endpoint-узлов нет. На трёх коротких
границах и положительном C8 tail старых узлов нет, но их непрерывная
площадь остаётся ненулевой. Старые Gauss weights не являются квадратурой
новых усечённых участков.

Угловые попадания восьми узлов содержат исходный rod budget
**16 089 739 995,575987 Дж**, остальные восемь —
**12 482 122 290,490795 Дж**. Все суммы представлены точными Fractions
буквальных binary64 значений, отдельно для prepared/previous/current
энергий и работы. Эти восемь угловых попаданий совпадают со старой
группой «archived» по входящему donor; это разные критерии. Старые donor
labels и состояния не переназначаются, physical transfer не принят.

Сохранены `original_atom_receiver_accounting.json`, точный
`original_atom_receiver_accounting_source.txt` и пояснение в родительском
validation-каталоге, вне четырёх выходов выпуска. Raw JSON SHA:
`cd3a28d4d38e43af4bdd4aec194f382409a1f035a5e9b8c5832ce5faab93e8ba`;
selfseal `556184de54d9f0dad09d3411a2c5c4d46be38d8bf2e1985412f166ac53a7723c`;
producer `8b9d1b641f85345ee450a0e7213ea827104075f316efbf922b2a79c13a8a7999`.

## Что остаётся в исходном пункте 1

**Закрыты два из трёх прежних блоков; остаётся один блок с тремя условиями:**

1. Распределить весь исходный материал/меру один раз, согласовать перенос
   old prestress/сил/работы, HOT law/mean chart, нагрузки/drag и новое общее равновесие.
2. Проверить все пары на всём времени движения, включая reverse C5→C1.
3. Принять продолжение, проверить независимую работу/балансы и joint restart.

Непрерывная мера `C dtheta` и 16 буквальных atomic weights — разные
представления источника. Новые local integrands сохраняют объявленную
непрерывную модель, но не переносят автоматически точный старый atomic
budget **28,571862286 млрд Дж**. Оба представления нельзя одновременно
считать живыми источниками энергии. Исходные 16 состояний/весов остаются
неизменными в истории.

У C11 два cuts образуют непрерывную непересекающуюся receiver partition.
У C8 остаётся положительный endpoint остаток порядка `6.021e-20 rad`,
у C7 — часть до trim, у Oct0 — весь диапазон. Они остаются явно вне
живого распределения. Design source allocation не создаёт для них bulk fields.

Исходные L/E/G не задают 3D endpoints, rest vectors и material masks
каждого стержня. При нулевом приращении новая цепочка сохраняет наблюдаемое
scalar pressure и задаёт variable reduced energy по буквальным L/E/G.
Равенство старому atomic energy budget, его перенос в 3D bulk и равенство
старых мировых обобщённых сил не сертифицированы. Модель глубины сама
по себе не исправляет эту недостающую связь и не завершает physical handover.

Принятых интервалов: **1 прежний / 0 новых**. Пункт 1 остаётся открытым.
