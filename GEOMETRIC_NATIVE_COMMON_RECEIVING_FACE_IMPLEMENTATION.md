# Полное сопряжение кривой C6–HOT C1 и односторонний контакт

08.10.2026. Ветка `fix/genesis-mantle-convection`, компонент913.

Собраны согласованные материальные поля обеих сторон всей подлинной
границы `retained-contact:octant:5:hot-curved-extra`, включая неявное начало,
конец, полный общий диапазон глубины и положительную кривую поправку.
Это новая объявленная контактная модель на подготовленном endpoint,
а не удостоверенный перенос прежних сил/работы или принятое равновесие.

Модули: `tectonics/geometric_native_common_cold_cut_fields.py` и
`tectonics/geometric_native_common_receiving_face.py`.

## Полная материальная граница

Сохранён источник аналитической кривой `(h,-k,1)/sqrt(1+h²+k²)`,
`k=B1/A1`, `h=b-a*k`, с подлинным original owner pose ровно один раз.
Нормированный параметр `u=u_m+(1-u_m)*lambda`, `0≤lambda≤1`, сохраняет
точный неявный корень `u_m` и его полную enclosure. На начале используются
коррелированные `h=0,k=q_max`, а не середина rootband. Прямоугольная
enclosure не становится множеством отдельных материальных точек.

Для донорского поля supporting-envelope identity даёт
`gap(t,u)≤0` на всём исходном времени. Поэтому кубическая positive-gap
activation archive carrier и её материальный gradient на самой живой
границе равны нулю. Это свойство точной кривой с полным source history,
не классификация произвольного прямоугольника интервальных лучей.
Положение archive boundary `r0*Qdonor*q`, gradient `Qdonor` сохраняются;
поверх них используются fixed895 scale, все56 мод, подлинная кровля,
depth twist и quintic quarter-seam blending, включая соседние банки.

HOT использует полный saved903 finite-mean chart и909 field algebra:
`X0=s*r*Qdepth(z)*U`, `A0=s*Qdepth*(I-r³/R²*(omega×U)⊗U)`.
У genuine HOT source roof равна0; `z=(R³-cubes)/(3R²)`.
Сохраняются first/second owner/body jets и implicit mean curvature.

Различие original input frames включает original cold roof. При
`Xhot_original=r*U`, `Xcold_original=r0*q`, `U=Qdonor*q`:

`B=Qdonor*((r/r0)I+(r0²/r²-r/r0)q⊗q)`, `det(B)=1`.

Cold material gradient в HOT source-world input равен `Acold*B^-1`.
Одна owner rotation не заменяет этот radial/tangential bridge.
Подготовленные материальные поверхности не принуждаются к совпадению,
исходные paired depths не меняются и фактический зазор не обнуляется.
В первом saved HOT query, на подлинном начале кривой при HOT-depth1000m,
получен gap≈−889.604m. Это различие подготовленных reference maps, а не
новый принятый поддвиг. Давление/энергия этого candidate сохраняются;
physical preparation/activation work не объявляется оплаченной.
[Наблюдение](analysis/geometric_native_common_receiving_face_validation/prepared_gap_observation.json).

## Новый физический контактный закон

Объявлен permanently-unbonded frictionless virgin contact-only law:
bare `kn=1e7 Pa/m` из существующего HOT parameter category, fixed restgap0
в совпадающем common source reference, нулевая сила сдвига. Это явная
новая декларация; она не наследует чужой born-shared free-gap−0.01,
старые16 atoms, память, activation ledger или condensed elastic columns.
Объёмные899/903 законы уже содержат собственную elastic energy.

`n=cof(Ahot)*m/|cof(Ahot)*m|`, где original-world material normal `m`
получена из подлинной common normal посредством `A0^T`.
`g=(yhot-ycold)·n`, `W=.5kn*min(g,0)²`.

`G_i=kn*min(g,0)*g_,i`,
`H_ij=k_active*g_,i*g_,j+kn*min(g,0)*g_,ij`.

Перенос сохраняет изменение нормали, HOT orientation hyperstress,
prestress curvature и все mixed owner/body entries. При `g=0`
возвращается semismooth tangent hull `[0,kn]`, а не единственный
классический Hessian. Opening имеет нулевую тягу.

Фиксированная подлинная source measure `J(u,z)*(1-u_m)d_lambda d_z`
учтена ровно один раз на всём `[0,1]` и всей общей глубине. Whole-face
outward range интеграл даёт энергию,100 сил и100×100 tangent enclosure.
Он не удостоверяет достаточную точность для mixed solve. Internal8 и
неактивные HOT/COLD координаты остаются в исходном каталоге с нулевым
вкладом этого интерфейса; новых DOF нет.

## Проверки и сохранение

Fresh909 evidence-only reader и исправленный baseline restorer PASS до новой
математики. Frozen gate **10/10tests PASS484.95s**, **12 full paired point
queries** и **8 whole-face100-coordinate operators**:4 HOT saved charts ×
zero/full cold. Native PASS911.273156731s; complete gate
PASS1075.680076721s, actualexit0/stderr0, stdout побайтово равен report.
Полный restart declaration/material/operator проверен. Whole-face диапазоны
остаются широкими; точность для равновесия не удостоверена.

Independent audit **PASS122.791921621s,220digits,146908comparisons**,
по всем12 points и8 operators; actualexit0/stdout==report/stderr0.
Независимо восстановлены curve representatives, COLD56 seam/mode gradients,
HOT chart jets, Nanson normal, все unweighted gap и weighted work derivatives,
roof bridge и100-coordinate range injections. Проверена совместная owner
objectivity до умножения на малую площадь. Геометрическая/root/mean theorem
остаётся inherited; interval representatives служат algebra diagnostics.
Сохранён первый failed audit adapter: Cartesian amplitudes ошибочно
читались как scalar h+five tracefree coordinates; исправлено на источник
scalar root и five_tracefree_amplitudes_m. Последний frozen audit PASS.



Закрытые909P39/F4,907P135/F4 и вся ancestry сохранены. Ровно4 additive
Python sources →913. Старые physical force solves, tensor controls,
старые pytest suites и reader/guard/auditor mains не исполнялись во время
новой математики. Fresh909 reader выполнен до неё; отдельно разрешённый
stdlib-byte-restorer907 выполнялся при восстановлении baseline.

Исправлена выявленная909 ошибка служебного восстановления: inherited907
restorer всегда получает обязательный `--output`. Старый sealed909 bundle
сохранён; исправленный913 restorer имеет отдельный receipt для inherited
восстановления. Его заключительная проверка выполняется после закрытия архива.

Сохранены development failures: ранний процесс импортировал interval-only
matrix helper для dense Jet (исправлено);8 из10 development tests прошли,
два неверных тестовых ожидания исправлены (fullcut rectangle support flag и
тот же interval-only helper в синтетическом примере). Они не скрыты final
PASS и не относятся к frozen validation. Raw captures/source facts сохранены.

## Остаток пункта1

**Пункт1 открыт: один существенный оставшийся блок, те же3 условия.**

1. Whole-source physical material/old-force/prestress/work handover,
   HOT law/mean chart, loads/drag и NEW mixed equilibrium. Готовы899/903,
   declared907, fields909 и полный объявленный C6–HOT C1 contact operator913.
   Остались физический handover и работа подготовленной reference/activation,
   modal loads/drag, достаточная force precision/general states и настоящий
   mixed solve. Ненулевой gap подготовленных сторон сохранён; его работа
   не считается оплаченной одним объявлением контактного потенциала.
2. Whole-clock all-pair admission, включая reverse C5→C1.
3. Accepted continuation с independent temporal path work, balances/joint restart.

Следующий конкретный шаг внутри прежнего условия1: согласовать подготовленную
reference с physical handover/work и обеспечить точный general-state mixed
operator для равновесия. Полный curved contact operator уже не требуется
собирать заново. Это ещё существенная физическая работа, не последние проверки.

Принят1 old и0 new intervals; source/contact/memory/material clocks,
предложенные damage/slip/work не приняты. Admission, новое равновесие и
оплата reference/activation work False. Другие HOT fronts этой отдельной
кривой не удостоверяются. Пункты2–6 исходного плана остаются открытыми.

[Gate](analysis/geometric_native_common_receiving_face_validation/report.json),
[audit](analysis/geometric_native_common_receiving_face_validation/audit_report.json),
[manifest](analysis/geometric_native_common_receiving_face_validation/artifact_manifest.json),
[final readback](analysis/geometric_native_common_receiving_face_final_readback/readback.json),
[archive4](analysis/geometric_native_common_receiving_face_final_readback/archive.json).

В следующей среде сначала:

```bash
python analysis/geometric_native_common_receiving_face_validation/baseline_restore_source.txt --output /workspace/receiving_face_restore_next.json
python analysis/geometric_native_common_receiving_face_final_readback/readback_source.txt --output /workspace/receiving_face_fresh_next.json
```

Это stdlib byte-restoration и evidence-only reader. После закрытия913/909/
907P/F их нельзя дополнять или переписывать; новые источники additive.
