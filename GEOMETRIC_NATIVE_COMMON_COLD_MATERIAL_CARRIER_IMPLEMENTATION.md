# Полные холодные материальные поля на совместном carrier

Дата: **07.10.2026, московское время**. Ветка `fix/genesis-mantle-convection`.
Компонент895: `DECLARED_GLOBAL_CONTINUOUS_COLD_MATERIAL_CARRIER_ENDPOINT_ONLY`.
Новый cold endpoint остаётся кандидатом; физическое время и damage не приняты.

Выполнен следующий шаг, записанный в передаче891: ненулевые cold fields
собраны на полном исходном материале, а геометрический допуск проверен
для целых объёмов восьми родителей и трёх непустых архивных когорт.
Вывод опирается на равномерную оценку по всему объёму и всей объявленной
коробке56 координат, а не на число положительных точечных Jacobian.

## Полное поле и исходная материальная система координат

Подлинные полные authorities879/891 и carrier887 аутентифицированы
по raw SHA256 и selfseal. Полные исходные source states сравниваются
буквально. Сохранены material IDs, owners, parents, архивные области,
first-entry ancestry, старые memory и work ledgers. Старые force solves,
их тесты и tensor-quadrature controls не повторяются. Public saved
geometry/material restoration и свежие point chronologies выполняются.

Используются все56 подлинных ненулевых COMMON cold amplitudes: семь
мод на каждом из восьми исходных банков. Используется actual COMMON
depth flow из embedding bank, а не прежний standalone depth rate.
Для полного исходного Cartesian chart нужны:

```
r0³ = R³ - 3 R² z
r_old³ = (R + w_roof)³ - 3 R² z
tau = (R³ - r_old³)/(3 R²)
C_roof = (r_old/r0)(I - q⊗q) + (r0²/r_old²) q⊗q
D_original X_old = B_old Q_owner C_roof
```

`tau` отличается от `z` при ненулевой крыше. `det C_roof=1`, но матрица
не равна единичной. Старый891 закрыт и не переписывается: его локальное
`BQ` не объявляется полным производным оператором нового исходного
материального chart. Прежние force receipts нельзя автоматически
перенести на новое поле без согласования chart и новой силовой сборки.

Независимые прежние quarter banks дают настоящий отрицательный скачок
на общем шве octant:1/3, ray `(-1,0,100)`, depth5000m: около
`-3.413560530085602e-6m` по физической нормали шва. Положительный
Jacobian внутри каждого банка такой разрыв не исключает.

Внутри одного owner вводится объявленная quintic partition шириной
`delta=1/100rad` в исходных углах около y=0 и z=0. По каждой активной
координате `t=(theta/delta+1)/2`, `S(t)=6t^5-15t^4+10t^3`, со
строгим насыщением0/1 за переходной полосой. Полное поле и градиент:

```
u = sum_parent w_parent u_parent
D_original u = sum_parent (w_parent D_original u_parent
                           + u_parent⊗D_original w_parent)
```

Все32 исходные source labels находятся вне полос стыковки: их прежние
значения семи мод сохранены. Полные производные теперь включают
`C_roof`. Новое поле вне этих меток намеренно отличается от независимых
прежних банков; это новая объявленная геометрия, не восстановление
глобальной гладкости, которой прежние данные не имели.

Физическое поле C1 на всех швах. Отдельные selector weights имеют
произвольные значения1/4 на исходных полюсах; там все моды точно нулевые
в открытой inactive окрестности, поэтому полное поле и его градиент
регулярны без сингулярного beta или seam chart.

## Равномерное доказательство на целых поддержках

Reference и support guard фиксированы для всех координат
`|a_i| <= 2|a_i_registered|`; нулевые, обратные и изменённые значения
внутри этой коробки допустимы, выход за неё отклоняется. По полной
глубине, всем исходным полусферам и всем этим координатам получены
`|u_owner|<=U_owner`, `||D_original u_owner||<=L_owner`.
Оценка производной стыковки использует `max dS/dtheta=15/(16delta)` и
`||Dtheta||<=1/(r_min cos b)`. Переходные полосы y=0 и z=0 не
пересекаются на активной поддержке мод; полярные окрестности обработаны точно.

Исходный887 carrier имеет положительный Jacobian и глобальные
границы `r_donor<=R`, `r_receiver>=R`. Фиксированные homotheties:

```
s_donor = 1 - 2 U_donor/R
s_receiver = 1 + 2 U_receiver/R
x = s_owner chi_887 + u
F_original = s_owner A_887 + D_original u
```

дают `|x_donor|<=R-U_donor`, `|x_receiver|>=R+U_receiver` во всей
коробке. Это объявленный геометрический запас, не решение finite mean
и не подобранная по query поправка. Он меняет и buffered source
positions; побайтное совпадение нового размещения с891 не заявляется.
Работа изменения reference всей системы ещё не удостоверена.

Для каждого owner равномерно `||F_original-Q_owner||<=D_owner<1/2`.
Полусферический материальный shell с одинаковой подлинной толщиной
всех четырёх quarter parents имеет квазиевклидовость меньше2: радиальный
отрезок до меньшего радиуса плюс кратчайшая дуга остаются внутри shell.
Следовательно, глобальная нижняя Lipschitz-оценка `1-2D_owner>0`
доказывает инъективность целого owner и всех его live/archive subsets.
Ориентационная гомотопия даёт `J>=(1-D_owner)^3>0`.

Подлинная материальная мера `R² dOmega dz` переносится посредством
нового Cartesian Jacobian; новая пространственная мера равна
`J R² dOmega dz`, а multiplier плотности `1/J` сохраняет исходную массу.
Атомарные площади и calibrated lengths не подставляются как объёмы.
Архивные debits не повторяются. Это геометрия всего холодного материала,
но не допуск HOT, всех физических пар или всей временной траектории.

## Проверки

Итоговая frozen попытка02: **17/17 тестов PASS128.64s**;
pytest wrapper130.080299349s. **Native80/80 queries PASS492.354027330s**,
wrapper492.390561137s; **same-process gate PASS646.960706186s**, exit0.
Native samples включают все32 source labels, все8 родителей по четырём
глубинным запросам, обе семьи швов, полосы стыковки, исходные полюса и
нулевые/обратные56 координат. В samples фактически встречаются2 архивные
когорты; равномерное доказательство охватывает все3 полные когорты.

**Независимый аудит PASS64.477085502s**, exit0,220 decimal digits:
80 текущих queries/96 literal-depth point evaluations,93120 scalar
interval comparisons, все64 прежних879field queries,32 прежних891
unbuffered positions, full56 modal gradients, uniform8parents/3archives,
ненулевой старый seam jump, новое single-valued field, fixed envelope,
глобальная инъективность, строгий radial gap и density pullback.
Подтверждены margins donor0.615089085443052/receiver0.576738490717730
и gap≥0.004036229310403m. Production project modules не импортировались.

Frozen895 canonical
`5715b5dedb30f6bae816737a01a5b19e29005224e2f39071f33aff4c1fa42f36`,
raw `231de3fb6cec25ed65b7ad1265fa55f6983b6bc723f177ba73c5deb904aaef4a`.
Gate selfseal
`9b85e572088a9c2f3d81b42cab794cea493b06b5f02636585dc6fbfaff061ac3`.
Native packet raw
`fbafd092a4eb98472e60f79665e1e0fe8e9e1c2c2b9d3679344a32e59593cc89`,
selfseal `27cbc05b687509fa22c00aa4339dd66d461c4def73ed37d4eae3acd62847da70`.
Audit selfseal
`cda4994cd14cc75e4d30d3b48393dc0492896ec65f879330be8f58135f556463`,
executing TXT raw
`97b0809a74e6f99aee22b61b541a03a9cd10a930d785ea0ba481d48e410c52f6`.
Audit report и настоящий stdout побайтно одинаковы.

Audit attempt01 остановился до математики: в project .venv нет mpmath.
Подлинный установленный Python3.12.14 из
`/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python`
с mpmath1.3.0 выполнил audit. Attempt02 прошёл uniform8/3,64oldfields
и32oldpositions, но ошибочно сравнил численный `mp.diff` с точным нулём
на концах quintic. Проверка концов заменена точными Fraction polynomials
`p'=30t²(t−1)²`, `p''=60t(t−1)(2t−1)`; interior derivatives по-прежнему
считаются независимым прямым Cartesian differentiation. Attempt03PASS.
Все executing sources/actual stdout/stderr и observation сохранены;
frozen895 Python ни в одной audit попытке не менялись.

Свежий891 evidence baseline выполнен ДО четырёх новых источников:
PASS16.632888618s, exit0. Сохранены891P44/F4,887P41/F4,341 raw dependencies,
15 временных associations и11 ancestry closures. Новые895 источники
ровно четыре; прежние891 Python не изменены. Полные frozen maps,
все executing snapshots и actual consoles сохранены в новом компоненте.

Попытка01:16PASS/1FAIL110.77s. Тест неверно читал верхний уровень
старого887 point DTO вместо вложенного `response`; сравнения56 мод,
производных и весов до этого места прошли. Также root запустил gate
до окончания статической правки wrapper: AFTER guard обнаружил изменение
нового validate.py. Первый запуск сохранён какFAIL, не выдан за PASS;
повторный запуск использует окончательно замороженные четыре источника.

## Что осталось в пункте1

Пункт1 открыт. Два прежних блока выполнены; остаётся ОДИН существенный
блок с ТЕМИ ЖЕ тремя условиями. План не расширен:

1. Физическая сборка всего исходного live/archive материала, перенос
   сил/prestress/work, HOT law/finite mean chart, нагрузки/drag и NEW
   mixed equilibrium. Геометрия полных холодных поддержек теперь
   удостоверена для явно объявленного endpoint и56-coordinate box.
   Остаются полные объёмные законы/силы и работа нового reference,
   история normal damage, follower law, HOT/mean, нагрузки и равновесие.
2. Whole-clock all-pair admission, включая reverse C5→C1.
3. Принятое продолжение, работа всей системы, балансы и joint restart.

Следующий конкретный шаг внутри условия1: перенести ранее явно
объявленную layered cold constitutive model на полное гладкое поле895
и собрать настоящие интегральные энергию,56 сил и Hessian с
межбанковыми членами, сохранив восемь beams ровно один раз. Данные
upper E/nu, lower E/G и границы слоёв уже есть в
`geometric_native_cold_strain_energy.py` и подлинных879body contexts.
Прежние isotropic/rank-one closure остаются объявленными моделями,
не восстановленной архивной prestress history. Восемь зарегистрированных
bulk remainder receipts не заменяют интеграл по новому объёму.
Далее HOT/finite mean, нагрузки/drag и mixed equilibrium остаются
в том же условии. Геометрический запас
не заменяет эти механические условия. Это существенный остаток,
не последние одна-две проверки. Пункты2–6 исходного плана открыты.
Принят1 прежний и0 новых интервалов; source/contact/memory/material clocks
остаются0/2000000/1000000/1000000 секунд. Предложенное damage не принято.

## Артефакты и восстановление

[Gate](analysis/geometric_native_common_cold_material_carrier_validation/release_validation_02/report.json),
[native packet](analysis/geometric_native_common_cold_material_carrier_validation/release_validation_02/native_control/source_packet.json),
[independent audit](analysis/geometric_native_common_cold_material_carrier_validation/independent_audit/report.json),
[manifest](analysis/geometric_native_common_cold_material_carrier_validation/artifact_manifest.json),
[final readback](analysis/geometric_native_common_cold_material_carrier_final_readback/readback.json),
[закрытый архив4](analysis/geometric_native_common_cold_material_carrier_final_readback/archive.json).

Повторное чтение после закрытия — только новый отсутствующий внешний файл:

```bash
.venv/bin/python analysis/geometric_native_common_cold_material_carrier_final_readback/readback_source.txt --output /workspace/cold_material_carrier_baseline_20261007/fresh895_readback_next.json
```

Если staging не сохранился, восстановить семь baseline файлов из новогоP
побайтно с проверкой raw по manifest; предварительно восстановить
аутентичные staging зависимости предыдущего891 по его отчёту.
15 TMP associations восстанавливать только из authenticated постоянных
копий, без прежних main/solver/quad. Final reader stdlib-only, проверяет
байты/форматы/полные namespaces/seals/источники/docs до и после, caller
module identity и старые замкнутые результаты; физику не повторяет.
Документы закончены ДО final readback. Закрытые895P/F и891P/F не
дополнять/не менять; следующий компонент делать additive источниками.
