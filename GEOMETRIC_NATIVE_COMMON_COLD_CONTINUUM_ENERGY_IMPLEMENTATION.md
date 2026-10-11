# Полная холодная энергия на совместном материальном carrier

Ветка `fix/genesis-mantle-convection`, компонент899: четыре новых Python
источника поверх неизменённых895. Дата07.10.2026, московское время.
Выпускной gate и независимый аудит PASS. Документы завершены до
финального evidence-only чтения; его фактический результат хранится в
ссылках на readback и закрытый архив ниже.

## Выполненная модель

Холодный закон перенесён на всё исходное материальное множество восьми
родителей, включая все три непустые архивные когорты. Собираются энергия,
все56 сил и полная симметричная матрица56×56 вторых производных. Учитываются
исходные roofs, производные quintic partition на общих швах и настоящие
границы верхнего и нижнего слоёв каждого родителя. Areal depth границы
равна `((R+roof)^3-(R+roof-h)^3)/(3*R^2)` и не заменяется nominal h. Это новый явно объявленный
холодный закон на подготовленном endpoint895; историческая трёхмерная
archive prestress и работа изменения reference не восстановлены.

Reference фиксирован: `X0=s*chi887`, где scale взят из неизменяемого895.
Ни scale, ни материальная геометрия не подстраиваются под запрос энергии.
Для полных исходных модальных градиентов определены

```
Gi = D_original(psi_i) * inverse(D_original X0)
L  = sum_i a_i Gi
E  = sym(L) + L^T L / 2
Hi = sym(G_tip_i)
Mij = integral Hi:C:Hj dV_original
R = sum_non_tip a_i sym(Gi) + L^T L / 2
b_i = integral Hi:C:R dV_original
p = a_tip + inverse(M) b
U = sum_8 [E_literal + F_literal*p + K_old*p^2/2]
    + [integral R:C:R dV_original - b^T inverse(M)b]/2
```

Восемь подлинных beam ledgers входят ровно один раз. Литералы прежних
`reference_energy_j` и `reference_force_n`, включая остатки двоичного
округления, сохранены; они не заменены произведениями `K*w` и `K*w*w/2`.
Общая8×8 Gram matrix включает связи соседних банков. Прежние нормы восьми
registered bulk remainders на оставшихся масках не используются как нормы
полного нового continuum. Все производные Green strain, включая смешанные
вторые производные и геометрическую prestress curvature, сохранены.

Интегрирование ведётся в подлинной исходной мере
`dV_original=R^2*cos(beta)*dtheta*dbeta*dz`; Jacobian нового prepared
reference не добавляется. Upper E/nu, lower E/G, полные14-полевые material
DTO, состояния, clocks и восемь beam DTO восстановлены из полных
879/891/895/833 packets. Lower isotropic extension и новый начальный span
`sum(w_ref*H_tip)` остаются явно объявленными моделями.

## Сертифицированное интегрирование

Выпуск использует полные интервальные boxes, а не разность двух Gauss rules
как оценку погрешности. Все основные IEEE binary64 операции округляются
наружу через `nextafter`; преобразования точных рациональных данных также
ограничены снаружи. Sin/cos/cbrt получают границы от прежних проверяемых
рациональных функций. Результаты сохраняются как точные Fraction intervals.

Для каждого запроса используются3072 основных boxes:
8 родителей ×2 слоя ×3 угловых участка ×4 угловых ×8 широтных ×2 глубинных
подразбиений. Ещё24 полных receipts покрывают узкий donor activation collar.
На остальном donor support исходный carrier точно является owner pose;
неактивная часть вне authentic clamp имеет точно нулевой вклад.
Ни архивный материал, ни узкая зона входа не заменены выборочными точками.

Независимая интервальная оболочка элементов Gram может включать
неположительные матрицы, поскольку она теряет корреляции элементов.
Для настоящего интеграла сохранены строгие нижние оценки норм из внутренних
областей с одной активной tip mode: `M>=diag(delta)>0`. Остальные области
вносят положительно полуопределённую Gram matrix. Проверяемая обратная
матрица ограничивается через правый residual `I-M*C` и эти реальные
оценки; рациональная midpoint inverse `C` служит только preconditioner.
Придуманный floor и замена физического интеграла midpoint не используются.

Сохраняются все глобальные интегральные каналы,56 сил и Hessian. Для каждого
box сохраняются полные параметры области, source volume, порядок каналов,
максимальная ширина и SHA256 полного вычисленного receipt. Полные локальные
векторы не дублируются тысячи раз в packet. Подлинные interior norm intervals
сохранены отдельно для воспроизведения доказательства `delta`.
Интервалы могут быть широкими; высокая точность сил и новое равновесие
этим компонентом не заявляются.

## Проверки и неизменяемые артефакты

Fresh895 evidence-only baseline PASS16.090911721s выполнен до новых
исходников. Frozen899release01: **18/18 tests PASS581.78s**, native control
**PASS276.893182480s** с двумя полными операторами и28 label queries;
same-process **gatePASS912.645987146s**, actual process exit0. Защитные
проверки BEFORE/AFTER удостоверили899 источников, неизменность всех895
прежних и1384 raw dependency pins. Прежние895P50/F4,891P44/F4,887P41/F4,
341raw dependencies,15TMP associations и11ancestry closures сохранены.
Прежние force solves/tests/tensor quadrature controls не запускались.

Независимый audit **PASS105.056217583s**, exit0:64028 native scalar
containment checks,7056 независимых Cartesian shape checks,
14112 сравнений всех56 модальных gradients на28 native points,
112 полей material DTO. В каждом fine запросе559872 quadrature nodes;
расхождение coarse/fine не более3.2568868e-9 по нормированному критерию.
Проверены энергия, все56 сил, полный Hessian и нелинейная prestress
curvature. Gauss convergence служит независимой проверкой, а не native
доказательством интервальной погрешности. Auditor не импортировал project
modules и не запускал прежние reader/auditor mains. Все899 Python sources
и64 direct authority pins проверены BEFORE/AFTER. Его actual stdout
побайтно совпадает с полным report JSON; stderr пуст. Gate и audit прошли
с первой попытки. Неуспешная coarse development LDL оценка сохранена как
явно неавторитетное наблюдение, не как неуспешный выпуск.

Границы настоящего оператора остаются широкими. В endpoint сохранён
energy interval приблизительно `[2.4494947409156052e11,2.5884902535174258e11]J`;
в нулевом reference энергия точно равна сумме8 literal ledgers,
приблизительно `2.5184126215702756e11J`. Максимальная ширина endpoint force
interval около`1.825441384899538e13N`, Hessian interval около`6.642649244546584e15N/m`.
Все исходные интервалы сохранены как точные Fraction bounds; эти десятичные
значения иллюстративные. Подлинный minimum interior Gram bound≈`4.6696066783e14N/m`;
inverse enclosure error upper≈`4.0284014975e-15m/N`. Независимая fine Gram
matrix имеет минимальное eigenvalue≈`9.9700334e14N/m` и condition≈1.91776.
Для mixed solve потребуется достаточная точность границ сил; она этим
endpoint компонентом не заявляется.

Четыре executing source raw SHA256:

```
core      186bf177dd7c51b80cea337697626f57518ef9c85722b35c0d29627d333340b7
tests     6ad6af85ccb715d95ceac61fbb4f049aa5fb1b8ac5140dd4cb7aad5effeb58ab
native    3812db899a098fb14f9240873492b779a48d5256a6423ec8a048146cce4204fa
validate  ec3635121c61272ced7f1bf99b655ae5e30aec1728c7b0f90edc904c9adef7dc
```

Frozen899 canonical map seal `53604474f4ee4ed47a271a73779bbe3ab202c81ff7fefbd6a8d5a039abf5566d`,
raw `1cae2a853d15bd4303b0affc99a2207b9bc8117b7278c275b7f99f07f64412f6`.
Gate seal `0e351d558619a3305249b8ee6533e0d72f054769c9cc8771baeb49064ecec73b`.
Native packet raw `4a5c059176cef063bf12ff7bd98df830395504d6c12051f269a177fcbc375fec`,
selfseal `c863be05c9d641025666a1a4c3929701c7d472f9d574085926d64a2d08b4c439`.
Audit raw `d178c432c178af0dbd1397d6bbcb6a813600dcaa7216b9bc39c51d763993a6f8`,
selfseal `3e8344fa85666cf52db78637d46d201e46d64e87699c5670f0788f4f607b1997`.

[Gate](analysis/geometric_native_common_cold_continuum_energy_validation/release_validation_01/report.json),
[audit](analysis/geometric_native_common_cold_continuum_energy_validation/independent_audit/report.json),
[component manifest](analysis/geometric_native_common_cold_continuum_energy_validation/artifact_manifest.json),
[final readback](analysis/geometric_native_common_cold_continuum_energy_final_readback/readback.json),
[закрытый архив4](analysis/geometric_native_common_cold_continuum_energy_final_readback/archive.json).
Manifest и final selfseals находятся в самих закрытых файлах, чтобы не
создавать круговой зависимости с этими документами. Финальный reader
исполняет только stdlib evidence checks: complete source maps, namespaces,
raw/selfseals, saved gate/audit/aliases, старые closures и неизменность
caller module identities; tests/math/solvers/quadrature не повторяются.

Фактическая команда выполненного gate:

```bash
.venv/bin/python analysis/geometric_native_common_cold_continuum_energy_validation/validate.py --release release_validation_01
```

После закрытия для следующего чтения используйте новый внешний output,
сохраняя immutableP/F:

```bash
.venv/bin/python analysis/geometric_native_common_cold_continuum_energy_final_readback/readback_source.txt --output /workspace/cold_continuum_energy_baseline_20261007/fresh899_readback_next.json
```

Если staging не сохранился, восстановите семь baseline895 файлов из новогоP
в `/workspace/cold_continuum_energy_baseline_20261007/`, проверяя manifest
raw pins. Предшествующий staging и15 TMP associations восстанавливайте
только из аутентичных постоянных DTO по
[инструкции895](GEOMETRIC_NATIVE_COMMON_COLD_MATERIAL_CARRIER_IMPLEMENTATION.md).
Копирование исходников и raw-pin checks разрешены; закрытыеP/F и прежние
PY не переписывать. Не повторять прежние mathematical controls ради
восстановления evidence. Для нового компонента создаются additive sources.

## Остаток исходного пункта1

Пункт1 открыт. Два прежних блока выполнены; остаётся ОДИН существенный
блок с ТЕМИ ЖЕ тремя условиями:

1. Whole-source physical material and old-force/prestress/work handover,
   HOT law/mean chart, loads/drag and NEW mixed equilibrium.
2. Whole-clock all-pair admission including reverse C5 to C1.
3. Accepted continuation with independent path work, balances and joint restart.

Полный холодный endpoint оператор относится к первому условию. HOT,
finite mean, normal damage history, follower law, полный перенос старых
сил/работы и новый mixed solve остаются в нём. Условия2 и3 ещё не закрыты.
Это существенный остаток, а не последние одна-две проверки. Пункты2–6
исходного шестипунктного плана остаются открыты.

Принят1 прежний и0 новых физических интервалов. Часы source geometry0 /
contact2e6 /memory1e6 /material1e6 секунд не продвинуты; предложенные
damage/slip/work не приняты. Geometry и cold constitutive endpoint
не означают physical admission, full-system time work или mixed equilibrium.
