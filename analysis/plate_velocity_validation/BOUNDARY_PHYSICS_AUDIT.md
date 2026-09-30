# Границы, силы и остаточная мантийная нагрузка

Аудит текущего Genesis → Starter → mature continuation на базе незакоммиченного
исправления тепла. Этот документ отделяет проверенное поведение от предложения
следующего изменения. Пороговые скорости, коэффициенты сил, время релаксации и
нулевой young ridge floor сохранены. Изменение проекции мантии описывается в
основном отчёте; здесь рассматриваются оставшиеся механизмы.

## Проверенная цепочка одного шага

`run_long_evolution_v123.py`, цикл `for dti in step_sizes(...)`:

1. `advance_thermal_state` обновляет тепло; `advance_mantle_flow` обновляет поле.
2. Рабочий `force_speed_scale_deg_per_myr` равен значению из конфигурации,
   умноженному на текущий `thermal.tectonic_activity_factor`.
3. `lithosphere.boundary_records_for_state` вызывает
   `kinematics.classify_boundaries` по **текущим** скоростям плит.
4. `subduction_memory.advance_subduction_memory`, затем
   `breakoff.advance_slab_breakoff` и `rollback.advance_rollback` обновляют память.
5. `dynamics.update_plate_dynamics` считает силы, target, релаксацию и удаление
   общего вращения. Затем следуют перенос материала и изменение топологии.

Поэтому расчёт target для сохранённого checkpoint при замороженных полях —
срез состояния, а не автоматически точное воспроизведение следующего полного
шага. Для последнего нужно воспроизвести также пункты 1–4. Трасса непосредственно
в `update_plate_dynamics(..., trace=...)` фиксирует фактические входы этой функции.

## Порог классификации действительно выключает физические ветви

В `tectonics/kinematics.py::classify_boundaries` нормаль направлена от A к B:

\[
v_n=(v_B-v_A)\cdot n_{AB},\qquad v=|v_B-v_A|.
\]

Сначала выбирается TRANSFORM, затем при `v_n > 4 km/Myr` — DIVERGENT,
при `v_n < -4 km/Myr` — CONVERGENT. Последнее условие `v < 1 km/Myr`
перезаписывает тип на INACTIVE. Это строгие неравенства. Оба числа берутся из
конфигурации; здесь приведены canonical значения.

Реальные потребители типа:

| Участок | Действующий барьер |
| --- | --- |
| `dynamics._boundary_force_terms_reference` | Ridge push только DIVERGENT; slab pull и continental collision только CONVERGENT; transform resistance только TRANSFORM. INACTIVE не добавляет этих членов. |
| `subduction_memory.advance_subduction_memory` | Любой контакт, не CONVERGENT, пропускается до накопления slab length. |
| `continental.advance_continental_cycle` | Boundary-local ветвь конвергентной переработки требует CONVERGENT. В современном runner внешний arc forcing поступает отдельно из slab memory. |
| `volcanic_arc.compute_volcanic_arc_forcing` | Обычная активная дуга требует активный slab, возраст и глубину; до рождения slab эта ветвь отсутствует. Сохраняется отдельная память post-breakoff импульсов. |
| `topology.PlateTopologyManager._update_collision_memory` | В ветви без `connected_collision_contacts` тип CONVERGENT участвует в доле конвергентного контакта и начале collision history; уже зрелый контакт может поддерживаться после смены типа. |

Следовательно, малое реальное сближение существует в `normal_rate_km_per_myr`,
но не создаёт начальный slab. Это проверяемая область нулевой обратной связи,
а не только ограничение цвета границы на карте.

При этом перенос материала не полностью заблокирован: `transport` накапливает
остаточные quaternion смещения; `lithosphere.advance_lithosphere`, участок
`True divergent gaps receive newborn oceanic crust`, создаёт материал из
**реальных геометрических пробелов** после raster commit без проверки DIVERGENT.
Число таких commits и появление пробелов нужно измерять отдельно. Новые разломы
могут возникать также через продолжающуюся young-shell damage модель.

## Ridge и slab: что есть сейчас

`dynamics.plate_ridge_push_factors` использует

\[
P=2\langle\max(\Delta\rho,0)\max(H,0)^2\rangle_{\rm ocean,plate},
\quad q=P/(64\times93.5^2).
\]

Далее действуют имеющиеся экспоненциальное насыщение, gain и ограничение.
`genesis_starter_continuation.build_starter_continuation` ставит
`ridge_gpe_min_factor=0`, так что при нулевых H или Δρ у океанической плиты
нет термического ridge члена. Ненулевой множитель вычисляется и без ridge
geometry, но его приложение требует DIVERGENT. Формула опирается на средний
первый момент плотности под плитой; она не измеряет непосредственно разность
GPE между реально открывшейся осью и флангом. Continental blend сохраняет
отдельный legacy вклад; это не young ocean ridge floor.

`subduction_memory.advance_subduction_memory` агрегирует **классифицированные**
конвергентные сегменты по ориентированной паре subducting → overriding.
Скорость сближения усредняется по длине траншеи. Сейчас:

\[
\Delta L=0.90\,\langle\max(-v_n,0)\rangle_{\rm trench}\,\Delta t,
\quad L\le1800\ {m km},\quad
D=\min(1100,L\sin\theta)\ {m km}.
\]

`genesis_starter_slab.young_slab_pull` умножает обычный slab proxy на
`min(L / slab_length_cap_km, 1)`, а также сохраняет прежний breakoff multiplier.
Нет зоны, зона broken-off или `L=0` — applied pull равен нулю. Residual pull
после потери контакта также масштабируется длиной один раз; активный slab не
добавляется повторно как residual. В обычном buoyancy proxy есть legacy lower
clip 0.02: raw proxy не следует интерпретировать как буквальную SI силу или
доказательство положительной отрицательной плавучести при любом H.

`rollback.zone_rollback_rate` требует активную зону, возраст не менее 20 Myr,
а также длину и глубину выше имеющихся минимумов 550 и 350 km. До появления
slab rollback не является источником первого движения.

## Предлагаемое устранение порогового барьера — ещё не реализовано

Сохранить текущие типы для диагностической уверенности/визуализации, а для
начала физических процессов использовать подписанную кинематику и накопленную
геометрию:

- Для реального океанического контакта интегрировать
  `Ldot = 0.90 * max(-v_n, 0)` с тем же выбором subducting side, геометрическими
  ограничениями и breakoff состоянием, **без условия типа CONVERGENT**.
  Применять существующее непрерывное развитие `min(L/Lcap,1)`. Это позволяет
  слабому вынужденному погружению накопить конечную геометрию без заданной
  стартовой силы. Необходим учёт поверхности/материала, чтобы память slab не
  записывала произвольное погружение, отсутствующее в транспортном ledger.
- Для spreading хранить реальное открытие `s_dot=max(v_n,0)` и соответствующую
  новую площадь/геометрию. Ridge push должен появляться из фактического
  термического/GPE контраста ось–фланг, который нулевой при отсутствии такого
  контраста. Не вводить ridge floor, новый launch threshold или постоянный
  ненулевой толчок при бесконечно малом положительном `v_n`.
- Если вводится непрерывный множитель физической активности, не делить torque
  на ту же activation-weighted длину: иначе множитель сокращается и конечная
  сила возникает при сколь угодно малой активности. Геометрическую нормировку
  и физический множитель нужно учитывать отдельно.

Это отдельное изменение causal chain с обновлением памяти, переноса и ledger,
а не скрытая замена `4` на меньшее число. Ни оно, ни полная физика slab initiation
в текущем исправлении проекции не заявляются выполненными.

## Остаточная мантийная нагрузка и fracture

`YoungWorldCoupling.advance_heat` продолжает `YoungShellFracture.advance`,
который вызывает общий `GenesisStarterModel._material_sample`. Сохраняются
мантийная prescribed stress tensor, охлаждение, приливы, накопление damage,
перенос памяти реальными material donors и поиск дополнительного разлома.
Нагрузка имеет масштаб `tau * mantle_stress_length_km / H * coupling(H)`.

Но `_material_sample` не получает текущие plate omega или `MantleFlowState`.
Поэтому это **не** вычисление обратной связи от фактического
`u_mantle - omega_plate × r`. В `late_tectonics.advance_late_tectonics` есть
эмпирическое накопление stress от размера плиты/старого океана, тоже без этого
локального residual. Само мантийное поле сохраняется; непредставимая часть
скорости не превращается в текущую локальную basal shear/stress в mature solve.

Разумное продолжение: вычислять `tau_res = basal_drag_pa_s_m * u_res` в SI,
передавать эту касательную нагрузку в согласованный мембранный stress solve и
существующий закон damage/fracture. Следует **заменить**, а не дополнительно
прибавить нынешнюю prescribed mantle часть, чтобы не считать одну нагрузку
дважды. Нужны моментный баланс, согласование сетки/материальной памяти и
проверки нулевой residual для rigid field. Это требует отдельной реализации;
добавочный произвольный множитель plate speed ничего не исправляет.

## Нормировка, релаксация и общее вращение

В `dynamics.update_plate_dynamics` ridge/slab векторы суммируются с длинами
границ и делятся на `boundary_weight`. Это безразмерные effective force proxies,
не N, Pa или N·m. Continental GPE аналогично нормируется собственными весами
`max(H-Href,0)*max(H-Hneighbor,0)`. Он действует при continental thickness
gradient внутри одной плиты и не зависит от типа межплитной границы.

\[
\omega_{\rm rel}=\operatorname{rad}(S)(d_{\rm boundary}+d_{\rm residual\ slab}
 +w_{\rm GPE}d_{\rm GPE})+\omega_{\rm rollback},
\]
\[
f_{\rm drag}=(1+C_{\rm collision}r_{\rm collision}
 +C_{\rm transform}r_{\rm transform})^{-1},\qquad
\omega_* =0.22\,\omega_{\rm mantle,plate}+f_{\rm drag}\omega_{\rm rel}.
\]

Drag не уменьшает mantle-advection член. Затем
`alpha=1-exp(-dt/velocity_relaxation_myr)` и
`omega_relaxed=omega_current+alpha*(omega_target-omega_current)`.
Релаксация не задаёт самостоятельную асимптотическую скорость.
Один общий area-weighted omega вычитается из всех плит: разности omega и
относительные скорости на границах сохраняются. Последующий индивидуальный
speed cap способен менять разности, если активен; это отдельная проверка
трассы. При явном mantle field legacy малые скорости не обнуляются.

Модули средних скоростей не аддитивны. Waterfall нужно сопровождать векторной
декомпозицией и одинаковыми area weights. Для физической SI замены в будущем
нужен plate torque balance с basal drag, ridge/slab и boundary resistance;
изменение одного `force_speed_scale` таким балансом не является.

## Проверки

`tests/test_dynamics_trace.py` проверяет точное совпадение tracing/non-tracing
для reference и optimized путей, сборку target из членов, релаксацию, удаление
gauge, отсутствие young ridge/slab при отсутствующей геометрии и реальную
область INACTIVE с ненулевыми normal rates. `tests/test_genesis_starter_slab.py`
проверяет рост длины и ненулевого pull при фактическом классифицированном
сближении, отсутствие роста без контакта, непрерывность `L/Lcap`, breakoff,
residual и отсутствие двойного счёта. Новая проверка в trace suite дополнительно
сверяет длину slab с интегралом его фактической скорости сближения.

Команда воспроизведения: `.venv/Scripts/python.exe -m pytest
tests/test_dynamics_trace.py tests/test_dynamics.py tests/test_ridge_push_v017.py
tests/test_subduction_memory_v018.py tests/test_genesis_starter_slab.py
tests/test_kinematics.py -q`.

Результат этой группы после добавления проверки интеграла длины: **39 passed**.
