# Текущее контактное состояние и объективные локальные силы

Дата: 05.10.2026 UTC. Ветка: `fix/genesis-mantle-convection`.

Продолжение первого условия оставшегося блока исходного пункта 1 после
[проверенных контактных границ 847](GEOMETRIC_NATIVE_COMMON_CONTACT_IMPLEMENTATION.md).
План из шести пунктов не расширен.

## Статус проверки

Наблюдение состояния и локальный контактный закон реализованы. Свежие
**Единый gate 853 PASS за 1269,308 с.** Свежие **19/19 тестов PASS за
343,11 с**. Native control **PASS за 526,124 с**: 20 current-state cases,
40 local-force responses и 16 000 Hessian entries. Полный typed source replay
прошёл; карты 853 до/после одинаковы, прежняя цепочка повторно проверена.
Независимый stdlib-аудит **PASS за 54,757 с**: настоящая текущая память,
80 независимых harmonic enclosures, 800 residual entries и все 16 000 Hessian
entries проверены отдельной точной рациональной арифметикой.

Один Python-процесс: pytest-обёртка **343,427 с**, callback **526,291 с**,
полный gate с replay/финальной сверкой — **1269,308 с**. Закрытая обёртка
содержит `changed_python_paths=[]`, source count **853** и 19 passed.

Development: 18 из 19 тестов прошли за 330,63 с. Оставшийся тест выявил
повторное расширение интервала давления при нулевом приращении. Production
исправлен сохранением исходного pressure/gap без повторного округления;
затронутый тест **PASS за 192,91 с** с precision160 источника и precision96
сил. Три отдельные проверки производных и объективности **PASS за 9,66 с**.

Область выпуска — `COMMON_CONTACT_REST_LAW_COMPONENT_ONLY`. Это локальные
энергия, силы и производные на единицу исходной опорной площади. Полная
память по глубине новой грани, её мера, перенос энергии/преднапряжения
старых стержней, новый HOT-закон и совместное равновесие не установлены.

Шесть добавленных Python-файлов:

- `tectonics/geometric_native_common_contact_state.py`;
- `tectonics/geometric_native_common_contact_constitutive.py`;
- `tests/test_geometric_native_common_contact_state.py`;
- `tests/test_geometric_native_common_contact_constitutive.py`;
- `analysis/geometric_native_common_contact_force_validation/native_control.py`;
- `analysis/geometric_native_common_contact_force_validation/validate.py`.

Каноническая карта всех 853 Python-источников:
`9d4a7a71f81b8093b09d26c74f742909839496fe18dd1429875e3a6d54c05c21`.
Raw SHA `frozen_source_map_853.json`:
`f9fafa4aedce0265147d44f9338cfcbc8b49518f66dfa5a4ce0a4a0ee968420e`.
Текущий проверочный прогон — `release_validation_02/`. Первый bootstrap
остановлен до выполнения тестов: снимок карты совпал по времени с последней
правкой validator. Его прежняя карта сохранена только как диагностика
`frozen_source_map_853_prefreeze_race.json`; она не является источником выпуска.

Прежние 847 источников, 1633 артефакта и закрытые выходы 826/833/841/847
проверены независимым stdlib-аудитом за 23,472 с. Тесты прежних выпусков
и объёмные интеграторы не выполняются заново.

## Настоящее текущее состояние

`observe_native_common_contact_state` принимает аутентифицированный
COMMON-reference и порт конкретного настоящего материального запроса.
Он берёт receiving family из его реальной холодной стороны, включая
две перекрёстные пары. Пластическое и накопленное скольжение вычисляются
из всех закрытых пересечений **actual `step.memory`** при исходном u=1.
Ни ближайший Gauss-узел, ни `previous_memory`, ни пересчитанный dense
endpoint не заменяют сохранённые коэффициенты.

Для реального источника первый коэффициент текущей пластической памяти
`−0.046385781250361705` отличается от конечного free jump
`−0.04638578125036171`. Это сохранённая разница округления. Dense query
`query_native_contact_field_history(...,u=1)` строится из предыдущей памяти
и конечных якорей; он является отдельным выражением.

Давление вычисляется из конечного free gap и буквального прежнего series
коэффициента. Отдельно хранятся contact kn/kt и старые column/series
коэффициенты, полные куски текущей памяти и source law. Сохранены:

- memory elapsed time **1 000 000 с**, stage count **1**;
- native elapsed time **1 000 000 с**;
- local contact elapsed time **2 000 000 с**, step count **2**;
- исходная метка текущих узлов **`unaccepted_candidate`**, все 16 состояний,
  площади, пластическая память и work ledgers.

Сжатый contact-only reference явно задан скалярными выражениями
`g_ref=−P_source/kn_contact`, `j_ref=p_current+T_source/kt_contact`.
Текущий source law имеет нулевые friction/cohesion/viscosity, поэтому
возвращённый shear точно нулевой. Он не вычисляется независимым
интервальным вычитанием j−p. Нулевая разность геометрических X не заменяет
этот контактный eigen-offset и не доказывает отрицательный physical gap.

Кривая C6/HOT C1 не имеет старого receiving family. API отказывает при
попытке присвоить ей давление, память или law соседней холодной стороны.

## Силы и полные производные

`evaluate_native_common_contact_constitutive` использует настоящие cold7
поля обеих сторон и шесть глобальных owner-координат — всего **20**.
Геометрические опорные normal/down/angular directors объявлены новыми
материальными направлениями получателя и поворачиваются вместе с ним.
Сохраняется скалярный preload; прежнее мировое направление traction
не объявляется сохранённым. Это material-director law, не follower-normal law.

Энергия зависит от относительного перемещения перемещённых сторон:
`xi_n=(R_receiver N_ref)·(y_receiver−y_donor)` и аналогично для down.
Для обеих проекций учтены все first/second derivatives: моды, owner/body,
симметричные owner/owner производные свежего левого exponential chart,
а также все производные receiver frame. При общем owner скалярные
owner/mixed derivatives точно нулевые до интервальных операций.
Это отдельно проверяется на настоящей HOT-кривой без назначения ей закона.

На compressed relaxed source branch:

`W=kn*g²/2`, `f_i=kn*g*xi_n,i`,
`H_ij=kn*(xi_n,i*xi_n,j+g*xi_n,ij)`.

Член с preload и второй производной сохранён. Обобщённая механическая
сила равна −f. Касательная shear-жёсткость физической relaxed ветви — **0**;
пластическое обновление только предлагается, не применяется к источнику.
Отдельная `frozen_memory_predictor` сохраняет kt и пробную shear-энергию;
она явно не объявлена фактической релаксированной ветвью закона.
Trial elastic shear extension отделён от actual post-return extension,
который на relaxed branch точно нулевой. Раскрытие отвергается: compressed
квадратичная энергия не продолжается через opening/damage без его закона.

Единицы производных зависят от координат: моды — метры, owner — радианы.
Per-area residual: body Pa, owner N/m. Hessian: body/body Pa/m,
body/owner Pa, owner/owner N/m. Новая площадь к ним автоматически не применяется.

## Контроль и источники

Native control заново связывает все пять холодных cuts. По два строгих
материальных запроса на каждом включают zero/nonzero модовые приращения:
**20 state cases**, **40** локальных откликов двух ветвей и **16 000**
элементов полных симметричных Hessian. Отдельно сохраняется HOT scalar
kinematics и отказ отсутствующего receiving law.

Полные четыре исходных DTO, saved128, COMMON-reference, cuts и 16 историй
хранятся один раз в source packet. Точные local ingredients связаны с
source/geometry/port SHA; они воспроизводятся по source query arguments.
State и force `from_dict` заново вычисляют содержимое и сравнивают весь
экспорт, включая память, часы, поля и производные. Подделанный selfseal
не заменяет authentic replay. Source snapshots и все предыдущие выходы
проверяются до и после единого процесса.

Авторитетный выпуск:
`analysis/geometric_native_common_contact_force_validation/release_validation_02/`.
Ровно четыре выхода плюс обёртка, raw SHA-256:

| Файл | Bytes | SHA-256 |
| --- | ---: | --- |
| `pytest.log` | 111 | `842e37ad449ff730ac9b11dca0a921d094ff770957ba2d82225d01cd4c7b74b3` |
| `control_console.log` | 553 | `2c5119cfb007551e09d38f986adbb8bc15a3954c78fa0790fca1cb82ea95f6f6` |
| `native_control/report.json` | 2108 | `2b4842d523e66b40d11406bb7458e33ecc44dc669d36e8bc1c5e6eaccbda5846` |
| `native_control/source_packet.json` | 104374116 | `fe06614b05591969e99364ffc81299ac4115e3a9e6586ced18647bbe85baa0c9` |
| `report.json` | 6604 | `1d7a6ac8113a408aab6764f1c4a9fccb09f8d48184f5118434fc685574504313` |

Canonical selfseal обёртки:
`fda985396e2e81e043019ead3cf6a7de4865c63f30a8ae1bacad8eb19b81befc`;
native report:
`f00d58ea4571a2529af98fe8e4ff9c746275d6a4143b1284e19629d1d3883497`;
source packet:
`36c3a2b61b6bf814a2d6ed305dbbcb26c5375ab52373e8d2e57422c33661527d`.

Независимый аудит сохранён **вне четырёх выходов**, в родительском каталоге:

- `independent_audit.json`, raw
  `e4c2cb68ed6a9f102b5ec2628f7c38966e90a067df8c80d2fd055dccec87fc92`,
  selfseal `20ece06aa865a0c4fcfe2bc7cabb9ee62808777ba5876081bd8171066b601f63`;
- `independent_audit_source.txt`, точный исполняемый producer, raw
  `8d755b719156876a87cb6f51b60221159d8d96e3dd92688e0b696f4bc12ea9e8`;
- `independent_force_formula_helper_source.txt`, обязательный helper, raw
  `061505c158ae8026549788c242fbda541c0718526d61c452f07e9f32d24d50cd`.

Аудит не импортирует project modules и не повторяет pytest/квадратуры.
Он проверяет все четыре исходных DTO по запечатанному архиву, saved128,
20 closed current-memory pieces, 200 буквальных p/c коэффициентов,
исходные часы и 21 кинематический запрос, включая HOT. Отдельные формулы
проверили 1257 первых/25083 вторых scalar derivatives, все 40 local laws,
800 residual и 16 000 Hessian entries. При нулевых приращениях pressure,
contact gap и reference energy сохраняются, а memory proposals не применяются.

Для повторного readback без нового физического расчёта:

```sh
.venv/bin/python analysis/geometric_native_common_contact_force_validation/independent_audit_source.txt \
  --out analysis/geometric_native_common_contact_force_validation/release_validation_02 \
  --expected-source-map 9d4a7a71f81b8093b09d26c74f742909839496fe18dd1429875e3a6d54c05c21 \
  --expected-tests 19 --result /tmp/native_common_contact_force_repeat_audit.json
```

Дополнительный source-only bottleneck review:
`source_bottleneck_review.json`, raw
`463fc5b944cbba681bdf00d225f98913bd4427ac072ba49ad7dad9b2e0f5574a`,
selfseal `779f88c8e72d9b1f6f3ff306a7bcb6980d2ac9559c6ab2c26e516bfb3e30f3ba`.
Его `source_bottleneck_review.txt` — поясняющая записка, **не producer**.
Review не создаёт новой HOT-активации и не устанавливает 3D rod prestress.

## Оставшаяся работа исходного пункта 1

**Закрыты два из трёх прежних блоков. В последнем остаются три условия:**

1. Полная мера/память новой грани и перенос rod prestress/энергии, отдельный
   HOT-закон, HOT mean chart, нагрузки/drag и новое смешанное равновесие.
2. Допуск всех пар на всём времени движения, включая reverse C5→C1.
3. Принятое продолжение, независимая работа пути, балансы и joint restart.

Старая source-мера `R*1000/sin(source_dip) dtheta_comp` не задаёт память
по настоящей полной глубине. Swept solid-angle coarea, source line area и
16 node weights остаются разными объектами. Existing handover сохраняет
complete rods/contact на исходной мере; он не переносит rods в новый bulk.
**28,572 млрд Дж** старой rod energy не входят в beam-калибровку новых тел;
постоянный ledger не создаёт их преднапряжение и обобщённые силы.

Source born preload −0.01 м относится к двум другим born arcs, не к C6/HOT C1.
Это free anchor полного series закона. Возможная NEW virgin curve law требует
отдельного явного объявления и не наследует её активацию или память.

Принятых интервалов: **1 прежний / 0 новых**. Пункт 1 остаётся открытым.
