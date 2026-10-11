# Конечный локальный перенос материальной поверхности, меры и памяти COMMON

Дата: 06.10.2026 (UTC). Ветка: `fix/genesis-mantle-convection`.

## Результат и граница выпуска

Статус: **PASS_LOCAL_TRANSPORT_COMPONENT_ONLY**. Во второй frozen попытке
все **14/14 новых тестов прошли за1378,12 с** по фактическому `pytest.log`
(22 мин58 с). Native control: **PASS10/10 surface и64/64 original queries
за350,96334030799335 с**. Единый gate: **PASS за2246,0146472550114 с**,
с отдельной root strict проверкой закрытых outputs. Независимый аудит:
**PASS за1073,233603071014 с** с одного разрешённого запуска.
Закрыт component evidence manifest83. Это завершённый локальный компонент;
пункт1 и физический handover остаются открытыми. Scope:
`FINITE_LOCAL_MATERIAL_SURFACE_MEASURE_AND_MEMORY_TRANSPORT_COMPONENT_ONLY`.

Новый компонент связывает конечные семь cold modes с локальной геометрией
настоящих retained COMMON cuts: положительной ориентацией, Nanson area-vector,
нынешней единичной нормалью и плотностью сохранённой reference меры. Current
plastic/cumulative slip остаются материальными скалярами того же receiving
label; их память и clocks не заменяются новым геометрическим состоянием.

Каталог содержит пять настоящих cold–cold cuts. Для каждого проверяются
нулевой и конечный ненулевой запрос: **10 локальных face queries**, каждая
с двумя отдельными сторонами. Дополнительно проверяются **64 конечных
original-material queries**: все 32 исходные attachments в глубинах `0`
и `H/2`. Это ограниченный набор локальных queries, а не сертификат всей
деформированной поверхности или всего трёхмерного архива.

Восемь archived donor labels сохраняют собственные original parent,
material ID, owner и cohort. Для вычисления их конечного поля явно объявлено
NEW продолжение семи мод того же original quarter за retained cold mask.
Наличие такого продолжения не объявляется принадлежностью нынешней face,
восстановлением исторического 3D archive motion или energetic handover
исходных сил, prestress и работы.

Существующий потенциал875 `U92=U871(Pi)+sum(Rperp)` не изменён. Новая continuous
face energy не активирована, quadrature не выполняется. Локальный перенос
геометрии и плотности не является новой сборкой сил, равновесием, whole-clock
admission или принятым продолжением. Сохраняется **один прежний / ноль новых
принятых интервалов**.

## Настоящая материальная геометрия и конечные поля

Пять cuts связывают original parents `5→1`, `5→3`, `6→2`, `7→2`, `7→3`.
Для `5→1` используется настоящий implicit terminal trim, а не весь polygon
proxy endpoint. Отдельная true curved граница `5→HOT` и born-shared0/3 не
получают receiving cold law через этот компонент.

Reference query использует одну общую физическую точку, но отдельные
материальные глубины сторон:

```text
r^3 = (R + roof_i)^3 - 3*R^2*z_i
tau = (R^3 - r^3)/(3*R^2)
X   = Q(A*tau)*(r*U)
```

Опорные masks, owner histories, trims, physical-radius intersection и
receiver computational coordinate остаются authenticated source expressions.
Неопределённый endpoint band не превращается в гарантированный material
patch через midpoint. После конечного относительного displacement стороны
могут иметь разные положения; reference gluing не объявляется нынешним
контактным ограничением.

При фиксированных material label и reference point `X`:

```text
chi(beta) = (1-beta/b)^2*(beta/b+2)/2
psi0      = chi*e
psiEi     = chi*Ei*X/R
u(a)      = sum_i a_i*psi_i
y(a)      = X + u(a)
F(a)      = I + sum_i a_i*G_i,     G_i = grad_X(psi_i)
```

`incremental_gradient` уже содержит identity; `displaced_point` уже содержит
`X+u`. Повторное добавление `I` или `X` изменило бы конечную карту.
`Ei` — фиксированные GLOBAL `xx,yy,zz,xy,xz,yz` symmetric generators;
оба off-diagonal coefficients равны единице. Common depth rotation и полный
материальный `grad(beta)` входят в `G_i`. Historical angular rate не
подставляется вместо active COMMON rate. Improper donor quarter frame
обрабатывается матричным сопряжением, с сохранением reflection sign.

Производные здесь по семи metre amplitudes данного тела:

```text
D_i y = psi_i                 D_ij y = 0
D_i F = G_i                   D_ij F = 0
E     = (F^T*F-I)/2
D_i E = (G_i^T*F+F^T*G_i)/2
D_ij E= (G_i^T*G_j+G_j^T*G_i)/2
```

Нулевая вторая производная `F` не означает нулевую вторую производную
cofactor, area ratio, нормализованной нормали или finite Green strain.
At/beyond `beta=b` семь displacement modes и их gradients точно нулевые;
query, straddling clamp, требует явного разделения.

## Nanson transport двух отдельных сторон

Для каждой стороны берётся её собственная конечная карта `F` и frozen
reference unit normal `N`. Cofactor convention фиксирована:

```text
F = [f1 f2 f3]
cof(F) = [f2 cross f3, f3 cross f1, f1 cross f2]  # COLUMNS
v       = cof(F)*N
j_s     = sqrt(v dot v) > 0
n_now   = v/j_s
J_now   = J_ref*j_s
```

Отдельно требуется строгий `det(F)>0`. Положительная длина area-vector не
исключает inversion. `J_ref` — настоящая mapped reference coarea; сохранённая
reference measure не отождествляется с ней.

Donor reference normal равна минус receiving normal. Две finite sides
сохраняются в отдельных `donor` и `receiver` DTO. При точном `F=I` известная
коррелированная unit-normal identity сохраняет `j_s=1` точно; это не
нормализация произвольного trial vector и не обнуление его body derivatives.

При `F_i=G_i`, `F_ij=0` cofactor имеет точный quadratic body 2-jet. Для
первой cofactor column:

```text
D_i c1  = g_i2 cross f3 + f2 cross g_i3
D_ij c1 = g_i2 cross g_j3 + g_j2 cross g_i3
```

Остальные columns получаются cyclically; `g_ik` — column `k` матрицы `G_i`.
Это математическая нумерация `1,2,3`, не индексы Python.

```text
j_i  = (v dot v_i)/j_s
j_ij = (v_i dot v_j + v dot v_ij)/j_s
       - (v dot v_i)*(v dot v_j)/j_s^3
n_i  = v_i/j_s - v*j_i/j_s^2
n_ij = v_ij/j_s - (v_i*j_j+v_j*j_i+v*j_ij)/j_s^2
       + 2*v*j_i*j_j/j_s^3
```

Сохраняются все первые и вторые body derivatives обеих сторон; genuine
interval uncertainty не заменяется `I(0)` лишь потому, что enclosure
содержит ноль. Эти jets относятся к локальному finite body query. Они не
объявляются full92 force/Hessian или finite owner trajectory.

`n_now` — Nanson image выбранной материальной reference поверхности.
Его вычисление не устанавливает новый follower-normal contact law,
исторический world traction или физический rest gap. Rigid covariance
проверяется отдельно от конечной истории owner motion.

## Reference measure и current receiving memory

На cold cuts используется уже явно объявленная NEW conditional depth модель:

```text
C      = R*source_contact_height/sin(literal_source_dip)
zeta   = (tau-tau_lo)/Delta_tau
mu_ref = C*abs(D_p theta_comp)*profile(zeta)/Delta_tau
rho_receiver_now = mu_ref/J_receiver_now
```

Nonnegative Bernstein profile имеет integral one. `mu_ref` фиксирована в
material/reference coordinates; division by current receiving coarea
сохраняет её локальный measure flux. Второй множитель `J_ref` либо `J_now`
не добавляется к reference density. Donor и receiver coarea после finite
jump не смешиваются. Receiver density использует именно receiver-side
Nanson result:

```text
rho_i  = -rho*j_i/j_s
rho_ij = rho*(2*j_i*j_j/j_s^2-j_ij/j_s)
```

Source density берётся из подлинного существующего
`NativeCommonSourceAllocation`: сохраняются four-family symbolic source
budget closure, active/inactive support cells и separate16-atom ledger.
Новый ответ связывает `receiving_cell_id`, `allocation_binding_sha256` и
`density_binding_sha256` с тем же current receiving state. Это allocated
reference density, а не повторное распределение полного бюджета на каждый
cut. Donor-side Nanson geometry имеет нулевой reference-measure input;
единственная выдаваемая current measure density относится к receiver.

Полные current pieces `p(theta_comp), c(theta_comp)`, anchors, source trace,
stores и отдельные clocks сохраняются. Новый displacement не меняет
material coordinate и не заменяет current memory previous/dense/nodal
интерполяцией. Observation zero-duration: нет source clock advance,
return-map adoption или нового debit.

Если API выдаёт `p*F*t_down`, это NEW кинематический вектор из нынешнего
скаляра `p` и выбранного reference tangent. Он не является восстановленной
историей пластического вектора, интегралом деформации по времени или
исторической cumulative work. В нынешнем
`NativeCommonSurfaceTransportResponse` такой дополнительный вектор не
выдаётся: сохранены только `plastic_slip_bounds_m` и
`cumulative_slip_bounds_m` как прежние материальные скаляры.

Компонент использует существующее once-only symbolic accounting четырёх
continuous receiver budgets; оно не объявляется full physical face
memory/measure/work handover. Literal old Gauss areas,
continuous `C dtheta_comp` и swept material coarea остаются разными мерами.
Ни новая конечная density, ни `p,c` не создают дополнительные old work stores.

## Исходные 32 метки и явное продолжение поля

По одному `sub` и `over` attachment каждого из 16 source point IDs сохраняют
точный literal `q`, original parent support, material ID и настоящий owner.
Для геометрии используется только `qhat=q/sqrt(q dot q)` и полный текущий
owner prefix `O*qhat`. Source mechanics продолжает использовать literal
`q` и виртуальный tip `R*q`; исходные directors и binary norm defects не
нормализуются вместе с геометрическим ray.

Проверяются глубины `z=0` и `z=H/2`, конечные `u`, `F`, Green strain и обе
reference-gradient chains:

```text
D_y/D_current_owner_chart  = F*D_Phi
D_y/D_original_Cartesian   = F*D_Phi*O
```

`D_Phi` — Cartesian gradient общей reference map после source owner pose;
`O` — отдельно сохранённый полный owner matrix. Точные значения явно
раскрывают chart domains:
`reference_gradient_input_frame=OWNER_CURRENT_CARTESIAN_INPUT_TO_COMMON_PHI`,
`total_original_gradient_input_frame=ORIGINAL_MATERIAL_CARTESIAN_BEFORE_OWNER_POSE`.
`total_material_gradient_bounds=F*D_Phi`,
`total_original_cartesian_gradient_bounds=F*D_Phi*O`.

Восемь archived donor attachments используют только declared seven-mode
extension своего original quarter. Их `original_live` geometry field ID
идентифицирует происхождение формул, а не нынешнюю retained-mask membership.
Archive receiver ancestry не переassigns material owner. HOT-birth material
не подставляется вместо archive. Нынешние cut queries и original attachment
queries имеют разные authority и не могут взаимно transplant responses.

Исторические sheet poses остаются различными. Injective COMMON Phi и
конечное field extension не доказывают gluing original donor/receiver sheets,
наличие archive точки на actual current cut или физическое движение всего
archived 3D body. Их энергетический handover, historical stresses и
распределение старой работы этим локальным запросом не восстанавливаются.

## Контракт API и подлинность query

Core имеет два точных formats:
`native-common-finite-material-surface-transport-1` и
`native-common-original-material-finite-field-transport-1`.
`extension_law` равен
`SAME_ORIGINAL_PARENT_SEVEN_MODE_COMMON_FIELD_EXTENSION_ON_AUTHENTIC_ORIGINAL_QUARTER`.
Дополнительного нового energetic system class нет: используются прежние
authenticated reference/port/allocation/material-system inputs.

| DTO / API | Точные ключевые поля или назначение |
|---|---|
| `NativeCommonFiniteSurfaceSide` | `incremental_gradient_bounds`, `determinant_bounds`, `determinant_gradient_bounds`, `determinant_hessian_bounds`, `cofactor_bounds`, `cofactor_first_derivative_bounds`, `cofactor_second_derivative_bounds` |
| Side Nanson 2-jet | `cofactor_normal_bounds`, `cofactor_normal_first_derivative_bounds`, `cofactor_normal_second_derivative_bounds`, `surface_stretch_bounds`, `surface_stretch_gradient_bounds`, `surface_stretch_hessian_bounds` |
| Side normal/coarea 2-jet | `current_unit_normal_bounds`, `current_normal_first_derivative_bounds`, `current_normal_second_derivative_bounds`, `reference_area_density_bounds`, `current_area_density_bounds`, `current_area_density_gradient_bounds`, `current_area_density_hessian_bounds` |
| `NativeCommonSurfaceTransportResponse` | `donor`, `receiver`, `reference_measure_density_bounds`, `receiver_current_measure_density_bounds`, `receiver_current_measure_density_gradient_bounds`, `receiver_current_measure_density_hessian_bounds`, `receiver_measure_conservation_residual_bounds` |
| Current receiving memory | `source_current_memory_trace_json`, `memory_piece_intersections`, `plastic_slip_bounds_m`, `cumulative_slip_bounds_m`, source memory/atom hashes и native/memory/contact clocks |
| `NativeCommonOriginalMaterialTransportResponse` | original parent/material/owner/cohort, `original_direction_literal`, `normalized_material_ray_bounds`, `current_world_material_ray_bounds`, `coordinates_m`, `displacement_increment_bounds_m`, `incremental_gradient_bounds`, `incremental_green_strain_bounds`, обе total-gradient chains |
| Finite cut evaluator / typed replay | `evaluate_native_common_surface_transport`; `native_common_surface_transport_to_dict/from_dict` |
| Original material evaluator / typed replay | `evaluate_native_common_original_material_transport`; `native_common_original_material_transport_to_dict/from_dict` |

Surface DTO явно сохраняет
`current_normals_used_as_constitutive_directors=False`; original DTO —
`current_live_mask_membership_authenticated=False`,
`current_contact_cut_membership_claimed=False` и
`deformed_source_rod_endpoint_equivalence_certified=False`.
Общие flags сохраняют `new_energy_or_force_law_added=False`,
`whole_face_measure_integral_certified=False`,
`deformed_sides_glued_or_injective_certified=False` и
`local_archive_force_or_work_handover_certified=False`.

Owner covariance law:
`FRESH_LEFT_RIGID_OWNER_ROTATION_AFTER_FROZEN_COMMON_REFERENCE_AND_BODY_DEFORMATION`.
В этом свежем rigid chart scalar measure owner first/second derivatives
точно нулевые; finite owner history этим законом не сертифицируется.

Complete query arguments, координаты, сторона, глубина, material/cut identity
и current state/dependency hashes должны быть связаны с ответом. Typed
serialization replay и semantic tamper/query-transplant rejection требуют
exact schemas; один валидный self-seal не разрешает borrowing другой query.

Packet явно содержит `original_cold_body_contexts`: восемь полных fresh96
factory DTO в sorted component order. Native report хранит
`original_cold_body_contexts_sha256`; wrapper связывает полный native output
со своим output manifest. Независимый adapter сравнивает physical/source
поля original body с pinned841 original160 DTO; разрешённые шесть различий —
precision, original geometry seal, owner matrix и три current-quarter
vectors. Fresh96 owner enclosure, source/body SHA и полный mathematical
COMMON body view проверяются отдельно. Original response body SHA относится
к настоящему телу того же original parent.

`source_material_point_binding_sha256` и reference-point geometry прямо
связываются с 64 сохранёнными pinned871 passive-point receipts по точным
point/side/depth. Это сохранённая provenance association, дополненная новой
независимой проверкой конечного поля, а не новый запуск старого control.
`density_binding_sha256` и `state_binding_sha256` аутентифицируются строгим
gate replay; независимый аудит восстанавливает physical density/memory
invariants, но не объявляет эти два токена независимо пересчитанными.

Final `replay_packet` рассчитан на фиксированные десять настоящих surface
queries. Он сравнивает весь serialized port DTO с соответствующим genuine
factory query и передаёт actual registered typed port в новый strict
response `from_dict`. Исторический generic
`native_common_contact_port_response_from_dict` в этом replay не вызывается.
Это полная проверка данного packet contract, а не заявление о новом
generic port decoder.

## Валидация и неизменность прежнего выпуска

До любых новых project imports или вычислений выполнен свежий stdlib
baseline875: **PASS_FINAL875_EVIDENCE_READBACK_ONLY за23,771418646 с**.
Raw report `1bae41cd1b0f5abc11decd794878c4084742e61e68c2e537fa8026275aeaaf1d`,
self-seal `5076cb213fdcef2ab59666ea525c6793d75fa12ea2a5a08011b2f631613b44ca`.
Exact executed reader raw:
`85de6cb6261971ed36e78756c038455a3ec114bccac65f2a6ea3704bcc30355e`.

Закрытые875 map и manifest73 остаются immutable:

| Объект | Raw SHA256 / canonical digest либо self-seal |
|---|---|
| Frozen875 source map | `8f7608219c398a05913d96e80baf685b997dccff466ee9cf9eb1aa89b8124675` / `13c19fe10056af9ab621e2c8a23e63ad36b301c95e121a6835cead019378e385` |
| Closed875 manifest73 | `035c7eeae472676924792d07e1d05b8e3660a4a9d0fa9d8cfe8f55df71862412` / `519d42468d7291b2af3db6894943a2ec91e2c12f490643e2d18f4b16d63777a2` |
| Final reusable875 reader source | `ad75a879a8485a22e21b44b97f42e89c43c408e399b4abd3a11cfcdb1adf8682` |
| Unused reusable-v2 draft source | `c7d6b28ad9f7d0ea6439e7cdf75b742c3866c24bc1c6f7f81f9173e854f59dd1` |

Final `ad75...` reader и unused `c7...` draft сохраняются отдельно с точными
raw bytes. Draft не объявляется исполнявшимся verifier. Перед runtime final
reader выбран явно. Raw copies:
`analysis/geometric_native_common_material_surface_transport_validation/baseline_875_reusable_source.txt`
и `baseline_875_reusable_draft01_source.txt`.
`baseline_875_reusable_draft_reconciliation.json` фиксирует
`PRESERVED_UNUSED_DRAFT_AND_CORRECTED_BEFORE_RUNTIME`,
`draft_executed=False`, `final_executed=False` на момент создания этой
pre-runtime записи; его self-seal
`2472fc28572f465a38be4bf08e2c6ff57ba2602f828d39a9d24b34a16c1ae3d8`.
Эта историческая запись не подменяет фактические последующие runtime
receipts final reader.

Reusable reader проверяет настоящую полную source map879 отдельно, до/после,
и разрешает ровно четыре новые Python paths. Вложенный неизменный reader85de
получает явно disclosed frozen875 subset view; его `actual875` flags относятся
к этой view, а не к фактическому source count879. Все180 recorded raw pins
проверяются wrapper до и nested reader после. Сохранённый before-math1633
baseline и девять strict ancestry closures не объявляются заново выполненной
физикой. Документы связываются отдельно от immutable Python/artifact freeze.

Контракт содержит ровно **14 новых тестов**: настоящие zero/finite cut sides,
независимые tangent-cross/cofactor/determinant/normal/density two-jets,
material memory/clock preservation, rigid covariance, all32/64 original
labels и finite field extension, Cartesian gradient chains, rejection of
inversion/unresolved support, typed replay и query-transplant rejection.
Исправленный frozen набор во второй попытке фактически завершился:
**14 passed in 1378.12s (0:22:58)**. Это полный pytest outcome, а не частичная
progress строка. Native control, единый gate и независимый аудит также
завершились PASS.

До полного gate выполнены три actual local smoke queries за
**373,7903216180275 с**: zero/nonzero cut0 и одна archived original query
в `H/2`. Статус `PASS_THREE_LOCAL_QUERIES_ONLY`, scope
`THREE_LOCAL_QUERIES_ONLY_NOT_RELEASE_GATE`; self-seal
`ccc9570bc25a15f5a6667630b7c0b5f60ad9c8d456a98bb4ffcf46307f8e5702`.
Эти три результата не заменяют десять surface/64 original queries,
14 тестов или независимый release audit.

Отдельный bounded pure AST guard сохранил actual результат для core8e678:
**592 exact component checks и 15 synthetic type checks**; project imports,
real material queries и solvers/quadratures отсутствовали. Его archive
self-seal `f8421c182ee43239c5f3dbf4752dbc8e161b4137f4876305509474b5a99cc5d8`.
Ранний report для core51ed после перезаписи восстановлен из ранее выданных
значений и точного producer formatting; reconstructed raw
`a055b938ee747bdd2671d87c6d97e9755286eb05ec55534c894625748006d028`
совпадает с ранее записанным raw hash. Он явно обозначен реконструкцией,
а не исходным сохранённым filesystem report или новым исполнением.
Normalized console artifacts также раскрыты как реконструированные без
повторного исполнения. Actual latest8e678 JSON сохранён отдельно.

Pure tests4/5/8 имеют сохранённый scope
`PASS_BOUNDED_PURE_AST_SURFACE_ALGEBRA`, guard self-seal
`930293890e69dca344baf08e18dd92f3cc4c5607ac729d237ce47cef60ab8696`.
После исправления test7 их AST и все helpers остались прежними;
эта проверка не повторялась. Сохранённая AST reconciliation относится к
эквивалентности исходников, а не к runtime outcome исправленного test7.

### Первая прерванная попытка и точный объём исправления

В `release_validation_01` фактически завершились **12 PASS и 1 FAIL**.
Failed test7 трактовал harmonic memory coefficients как power basis;
production memory law остаётся
`a0+a1*cos(theta)+a2*sin(theta)+a3*cos(2*theta)+a4*sin(2*theta)`.
Test14 не завершился, новый native control не запускался. Owner прервал
только принадлежащий этой попытке PID155995 в **13:06:52 UTC**: старый
generic port decoder заново восстанавливал всю коллекцию шести cuts/ports
для каждой serialized row. Это не полный результат 14 тестов и не PASS
native/gate.

Сохранённые pytest/gate durations первой попытки составляют соответственно
**1569,610437876021 с / 1636,7888344140374 с**. Exact frozen sources4,
map, log/report и before/after875 baselines сохранены в
`failed_validation_01` и неизменном `release_validation_01`.
Preservation self-seal:
`bfb74073330e244bd5eeb121c5e441b964149d7a076ff6620ca0319e40fc06da`.
Его исторический status string
`PRESERVED_FAILED_FROZEN_GATE_01_BEFORE_TEST7_ONLY_CORRECTION`
сохраняется буквально, но реальный объём явно указан отдельным
`correction_scope=TEST7_HARMONIC_ORACLE_AND_FIXED_PACKET_REPLAY_WITHOUT_REPEATED_OLD_WHOLE_PORT_COLLECTION_BINDING`.

Исправлены ровно два top-level AST names в двух новых файлах:
`test_literal_current_memory_pieces_and_all_clocks_advect_at_the_same_material_label`
и `replay_packet`. Core и validation wrapper не изменились, все875 прежних
Python sources и закрытые73 artifacts сохранены. Новый test7 использует
150-digit Decimal harmonic oracle с 180-term sine/cosine, next-term absolute
Taylor remainder и явно выведенной абсолютной forward-rounding границей.
Его source reconciliation self-seal:
`0fb53e42c30861a9a8bf6b86b27455bc3b011c20947645a0dbb923bba5028ce5`.
Final fixed packet replay сохраняет полный raw-port DTO equality, но
использует соответствующий genuine typed query вместо повторного
whole-port-collection binding; его AST reconciliation self-seal
`83cbf766db5e49b1a983bf7c77eb2e2198f18aa6136708d563ed04894b4f003d`.
Общий source-epoch transition self-seal
`c95e62859735e4efe99a937c46f07376a5f10c6d533b6a6854da939e4f4dd110`.

Первая879 canonical map
`93193719c3f64e9912b71c08be4418918b86139f9865c9bb95715dd6985d58ef`
и raw `a77717f3cbde54537fb37d4fc5c909e407881d09c90f2bb6cdc92882f442ee20`
сохранены отдельно. Исправленная879 map приведена в таблице ниже.
Предыдущий independent producer
`75285eacdaf72d01c5203daa2b863644906468ea3cb3228bea97ca9aff9b45cd`
сохранён как **UNEXECUTED_GATE01_DRAFT**. Final producer
`c38dd9f2e9d0efeac42b64d50c15601d40fb1f56232edafcf34de19f47c11502`
меняет source-map pins, docstring и два serialization-authentication
descriptors в 105-field report contract; mathematical function AST и
source schemas прежние. Producer reconciliation self-seal
`62bed7a2beea89d4a6ae57d15be038000232f28bcbfcfc554c21b720f32fd208`.
Предварительная compile-only/AST запись сохранила подготовку producer.
Затем именно final c38 producer был исполнен одним отдельным worker и
завершился PASS за1073,233603071014 с; ранний75285 draft не исполнялся.
Подготовительная запись не подменяет этот фактический runtime receipt.

Текущая исправленная frozen попытка: **release_validation_02**.
Свежий pytest фактически завершился: **PASS14/14 за1378,12 с**
(22 мин58 с; точная напечатанная pytest duration). `pytest.log` содержит
полную строку `14 passed in 1378.12s (0:22:58)`.
Native control фактически завершился: **PASS10/10 surface и64/64 original
queries за350,96334030799335 с**. Полный wrapper/gate завершился PASS,
root strict проверка закрытого отчёта и outputs выполнена. Независимый
аудит завершился PASS в отдельном единственном разрешённом worker
с exit code0. Exact wrapper-measured pytest duration:
**1380,5565931419842 с**; native callback duration:
**351,019289251999 с**.
Native: **PASS;10/10 actual cut queries,64/64 original material queries
за350,96334030799335 с**.
Gate: **PASS за2246,0146472550114 с**.
Независимый stdlib audit: **PASS за 1073,233603071014 с**;
фактические счётчики: **10 queries /20 finite sides /5 genuine cold cuts;64 original queries /32 labels
(8 archived donors,8 live donors,16 receivers);8 complete cold body contexts;
7 first derivatives и49 second coordinate pairs на сторону;64 prior passive-point
associations;193 recorded raw dependency pins, включая old180, и10 ancestry closures**.
Before/after-gate baseline: **PASS за38,49952138203662 с /
30,391725720954128 с**. Перед независимой математикой выполнен отдельный baseline:
**PASS за20,669048584997654 с**, nested875 projection за11,045058068993967 с.
Raw baseline219cc04943bf857ec2f6b7961f2e533145a7033299b716fe92284caee1b01175;
self-seal16e1325a22bcc16260bd2d963075080663837c2cd362fe52b32992d67b1803c7.
После аудита reusable baseline main отдельно не повторялся: непосредственно
проверены все879 source bytes,193 raw dependencies и10 strict ancestry closures —
PASS. Эти guards не объявляются повторным исполнением старого baseline/physics. Фактический результат отдельного
финального readback определяется связанной квитанцией, указанной ниже.

Gate PASS относится к фактически завершённой второй попытке; независимый
аудит PASS подтверждён завершённым отдельным исполнением. Прежние tests, physical solvers и
quadratures не повторяются; source875 и все73 закрытых artifacts не
перезаписываются. Восстановление подлинных factories/DTO отделяется от новой
физики. Фактически случившиеся failures/corrections сохранены с исходными
sources, logs и mappings в `failed_validation_01`, `release_validation_01`
и трёх test7/replay/source-epoch reconciliation receipts, описанных выше.

| Новый объект | Raw SHA256 / canonical digest либо self-seal |
|---|---|
| Frozen879 source map | `1b948eace979f3125756f6f0dea803fb55d0d2e87650890a2236c07502b0c336` / `4a75d4bc7d404fc11d5d252302fe6b6d961797b4a637f921c38acb679485b79c` |
| Core | `8e6781187f5c8afad8dd201dfbf3c2165c2f07e862bf72ef1017cd5d20433438` |
| Tests | `5d5aee42586ea783ef049dc3c9de517bdbd4c16d066819cded5cfa2c65e859d0` |
| Native control | `ad39571acffa6a88ddc0b14f4163b81cdb622b38f92831aed854c49718a41151` |
| Wrapper | `790f81d7edca699a54988eab0ed9e6e9a94650f572f0e466cc1b4ebbd259779b` |
| Gate report | `82f3b165a6a4da350ae83a587ea77ce114aef97f7794d7e6e7a871dfd900715c` / `03bd4048e8d4515c61433b7605fcff775f9058aa767c5c076a1f83a251fb767d` |
| Native report/packet | `25796983f80d0647613bfc31fafb2f7f1ca423f736c04a6baa6310fd3a94ff39` / `8727b6924e77f790b7a85b7caa6407a6bd1729a712f481fdfa2b11dd7bc140e7`; `f2884284f663564d304537f91614efbb01e5b7df5ebdab86ac7eb6e77287ee64` / `368a524a954ac5f2ba3a6b92c182d34d1c3567ba7df1314c0afd02b2cff2c23e` |
| Independent audit source/result | `c38dd9f2e9d0efeac42b64d50c15601d40fb1f56232edafcf34de19f47c11502` / `39b58a92972e8b570f59ca3d7550edb794b6ae01f052d15678048dc729ab988f` / `c857af3af356f873c863f20334bb562f5a1074b57586f7e480ce89686d67e38b` |
| Closed879 evidence manifest | `1347310d09095ce8e3dff7db970262524c323a5542861fa2402bb1431c7f7e4b` / `651f204118935dadff0a6326b5534772e30a7e2af9374c41ba5ac099641cde66` |

Evidence directory:
`analysis/geometric_native_common_material_surface_transport_validation/`.
Четыре новые Python paths:
`tectonics/geometric_native_common_material_surface_transport.py`,
`tests/test_geometric_native_common_material_surface_transport.py`,
`analysis/geometric_native_common_material_surface_transport_validation/native_control.py`,
`analysis/geometric_native_common_material_surface_transport_validation/validate.py`.
## Итоговая проверка сохранённых файлов

Источником результата финального stdlib readback служит отдельный
[readback.json](analysis/geometric_native_common_material_surface_transport_final_readback/readback.json).
В нём фиксируются фактический статус, длительность и проверенные raw SHA256:
полная карта 879 Python sources, закрытый evidence manifest, прежние
зависимости и оба окончательных документа. Эта проверка читает сохранённые
байты и seals; тесты, физический solver и квадратуры она не выполняет.

[archive.json](analysis/geometric_native_common_material_surface_transport_final_readback/archive.json)
связывает точные копии исполнявшегося reader, его JSON и console с исходными
путями, размерами и raw hashes. Копирование артефактов в постоянный каталог
не объявляется повторным исполнением. Final readback хранится отдельно от
закрытого component manifest и не добавляется в него после запечатывания.

Отчёт и обе текущие части `CLOUD_HANDOFF.md` окончательно записываются до
единственного финального запуска reader; команда передаёт оба файла через
`--document`. Reader проверяет именно эти байты, после чего документы не
переписываются. Фактические result pins и длительность reader берутся из
связанной квитанции, поэтому их копия в этом отчёте не создаёт цикл из
проверки документа и последующего изменения его SHA256. Наличие ссылки
само по себе не объявляет статус проверки: статус определяется содержимым
квитанции и её проверенным seal.

## Продолжение в новой облачной сессии

Финальный reader raw source:
`f23843c4065d8d8bd4095a9f0b1ecbf948386e92ea1047506f09104bbf7a66f1`.
Manifest producer raw source:
`e55afa87cf8bc7d94241ab436b14f85e74d73d5ddea36d89018a0e9bd3f31c6d`.
Точные исходники доступны в отдельном final readback archive как
`readback_source.txt` и `manifest_producer_source.txt`. Его шесть файлов:
два producer TXT, `readback.json`, `console.log`, `manifest_console.log`
и `archive.json`. Этот каталог находится вне component manifest83.
Первоначальный reader исполняется из закреплённого scratch source; постоянный
TXT сохраняет его точные bytes. Origin paths и raw-copy provenance хранятся
в `archive.json`; само копирование не объявляется повторным исполнением.

### Восстановление восьми текущих зависимостей


В новой сессии восстановите только отсутствующие восемь CURRENT `/tmp`-зависимостей из закрытых постоянных копий. Raw SHA256 и self-seal manifest закреплены в коде; пользователю не требуется подставлять новые значения. Все восемь текущих путей и пинов выводятся из закрытого manifest и связанных gate02/audit receipts; результат финального readback не используется как вход. Код сначала проверяет все источники и существующие назначения. Неверный существующий файл означает FAIL без перезаписи. Исторические TMP-пути прерванного `failed_validation_01` остаются метаданными и не восстанавливаются. Физика, baseline, тесты и producer не исполняются.

```bash
cd /workspace/Habitable-Moon-Project
python - <<'PY'
from pathlib import Path
import hashlib, json, os, shutil, sys

ROOT = Path('/workspace/Habitable-Moon-Project')
PARENT = ROOT/'analysis/geometric_native_common_material_surface_transport_validation'
OLD = ROOT/'analysis/geometric_native_common_strain_projection_validation'
MANIFEST_RAW = '1347310d09095ce8e3dff7db970262524c323a5542861fa2402bb1431c7f7e4b'
MANIFEST_SEAL = '651f204118935dadff0a6326b5534772e30a7e2af9374c41ba5ac099641cde66'
AUDITOR_RAW = 'c38dd9f2e9d0efeac42b64d50c15601d40fb1f56232edafcf34de19f47c11502'
OLD_BASELINE_RAW = 'b048664d22012a46ca95715e60586424568995c37f4637b18e7b51aa40cd685d'
OLD_MANIFEST_RAW = '035c7eeae472676924792d07e1d05b8e3660a4a9d0fa9d8cfe8f55df71862412'
OLD_MANIFEST_SEAL = '519d42468d7291b2af3db6894943a2ec91e2c12f490643e2d18f4b16d63777a2'

def enc(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()

def vd(value):
    return hashlib.sha256(enc(value)).hexdigest()

def exact(actual, expected, label):
    assert type(actual) is type(expected) and actual == expected, label
    if type(actual) in (dict, list, tuple):
        assert enc(actual) == enc(expected), label + ': literal nested kinds'

def raw(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()

def sealed(path, key, expected_raw=None, expected_seal=None):
    if expected_raw is not None:
        exact(raw(path), expected_raw, 'closed raw bytes: '+str(path))
    result = json.loads(Path(path).read_bytes())
    assert type(result) is dict and key in result, str(path)
    body = dict(result); seal = body.pop(key)
    exact(seal, vd(body), 'closed selfseal: '+str(path))
    if expected_seal is not None:
        exact(seal, expected_seal, 'pinned selfseal: '+str(path))
    return result

for pin in (MANIFEST_RAW, MANIFEST_SEAL):
    assert len(pin) == 64 and all(c in '0123456789abcdef' for c in pin), 'Fixed final879 manifest pins must be complete'
assert Path.cwd().resolve() == ROOT and ROOT.resolve() == ROOT
manifest_path = PARENT/'artifact_manifest.json'
manifest = sealed(manifest_path, 'manifest_sha256', MANIFEST_RAW, MANIFEST_SEAL)
exact(manifest['status'], 'PASS_COMPONENT_EVIDENCE_CLOSED', 'closed component manifest')
exact(manifest['source_count'], 879, 'manifest879')
exact(manifest['source_map_canonical_sha256'], '4a75d4bc7d404fc11d5d252302fe6b6d961797b4a637f921c38acb679485b79c', 'closed final879 source epoch')
exact(manifest['source_map_raw_sha256'], '1b948eace979f3125756f6f0dea803fb55d0d2e87650890a2236c07502b0c336', 'closed final879 raw map')
exact(manifest['exact_independent_producer_raw_sha256'], AUDITOR_RAW, 'closed manifest exact executed auditor')
exact(manifest['closed_component_file_count'], 83, 'completed explicit83 closure')
exact(len(manifest['files']), 83, 'explicit83 manifest files')
old_manifest_path = OLD/'artifact_manifest.json'
old_manifest = sealed(old_manifest_path, 'manifest_sha256', OLD_MANIFEST_RAW, OLD_MANIFEST_SEAL)
exact(len(old_manifest['files']), 73, 'immutable old73 manifest')

def closed_file(path, pin):
    assert path.resolve() == path and path.is_file() and not path.is_symlink(), 'canonical closed file: '+str(path)
    name = str(path.relative_to(ROOT))
    rows = old_manifest['files'] if path.is_relative_to(OLD) else manifest['files']
    row = rows[name]
    exact(row['sha256'], pin, 'manifest permanent raw pin: '+name)
    exact(path.stat().st_size, row['bytes'], 'manifest permanent byte count: '+name)
    exact(raw(path), pin, 'verified permanent raw bytes: '+name)

wrapper_path = PARENT/'release_validation_02/report.json'
audit_path = PARENT/'independent_audit.json'
for path in (wrapper_path, audit_path):
    closed_file(path, manifest['files'][str(path.relative_to(ROOT))]['sha256'])
wrapper = sealed(wrapper_path, 'report_sha256', expected_seal=manifest['wrapper_selfseal'])
audit = sealed(audit_path, 'audit_packet_sha256', expected_seal=manifest['independent_audit_selfseal'])
exact(wrapper['status'], 'PASS', 'CURRENT gate02 PASS')
exact(wrapper['source_count'], 879, 'CURRENT gate879')
exact(wrapper['pytest_counts'], {'passed':14}, 'CURRENT completed14 tests')
exact(wrapper['changed_python_paths'], [], 'CURRENT frozen source execution')
exact(wrapper['source_map_before_sha256'], manifest['source_map_canonical_sha256'], 'CURRENT gate before source epoch')
exact(wrapper['source_map_after_sha256'], manifest['source_map_canonical_sha256'], 'CURRENT gate after source epoch')
exact(audit['status'], 'PASS', 'CURRENT independent audit PASS')
exact(audit['auditor_script_sha256'], AUDITOR_RAW, 'CURRENT executed scratch auditor pin')
exact(audit['source_map_canonical_sha256'], manifest['source_map_canonical_sha256'], 'CURRENT audit source epoch')
exact(audit['wrapper_raw_sha256'], manifest['files'][str(wrapper_path.relative_to(ROOT))]['sha256'], 'CURRENT audit gate raw association')
exact(audit['wrapper_selfseal'], wrapper['report_sha256'], 'CURRENT audit gate selfseal association')
exact(audit['output_manifest'], wrapper['outputs'], 'CURRENT audit same gate output closure')
expected = {}

def association(temporary, permanent, pin):
    temporary = Path(temporary)
    assert temporary.is_absolute() and temporary.parent == Path('/tmp') and temporary.resolve(strict=False) == temporary, 'direct canonical CURRENT TMP path'
    assert str(temporary) not in expected, 'each CURRENT temporary target appears exactly once'
    closed_file(permanent, pin)
    expected[str(temporary)] = dict(permanent_copy=str(permanent), sha256=pin)

for when, basename in (('before', 'baseline_before_gate'), ('after', 'baseline_after_gate')):
    path = PARENT/(basename+'.json')
    receipt = sealed(path, 'audit_sha256', wrapper['prior_baseline_'+when+'_raw_sha256'], wrapper['prior_baseline_'+when+'_selfseal'])
    exact(wrapper['prior_baseline_'+when], receipt, 'CURRENT gate embedded '+when+' receipt')
    association(wrapper['prior_baseline_'+when+'_path'], path, wrapper['prior_baseline_'+when+'_raw_sha256'])
    projected_path = PARENT/(basename+'_projection.json')
    sealed(projected_path, 'audit_sha256', receipt['projected_reader_report_raw_sha256'], receipt['projected_reader_selfseal'])
    association(receipt['projected_reader_report_path'], projected_path, receipt['projected_reader_report_raw_sha256'])
gate_tmp_paths = set(expected)
path = PARENT/'baseline_before_independent_audit.json'
receipt = sealed(path, 'audit_sha256', audit['baseline875_readback_raw_sha256'], audit['baseline875_readback_selfseal'])
association(audit['baseline875_readback_path'], path, audit['baseline875_readback_raw_sha256'])
projected_path = PARENT/'baseline_before_independent_audit_projection.json'
exact(audit['baseline875_projected_reader_report_path'], receipt['projected_reader_report_path'], 'CURRENT audit projected TMP path')
exact(audit['baseline875_projected_reader_report_raw_sha256'], receipt['projected_reader_report_raw_sha256'], 'CURRENT audit projected raw association')
sealed(projected_path, 'audit_sha256', audit['baseline875_projected_reader_report_raw_sha256'], audit['baseline875_projected_reader_selfseal'])
association(audit['baseline875_projected_reader_report_path'], projected_path, audit['baseline875_projected_reader_report_raw_sha256'])
association('/tmp/native_common_material_surface_transport_independent_audit_20261006.py', PARENT/'independent_audit_source.txt', AUDITOR_RAW)
association('/tmp/native_common_strain_projection_final_independent_audit_20261006_baseline871_readback.json', OLD/'baseline_before_independent_audit.json', OLD_BASELINE_RAW)
exact(len(expected), 8, 'exact8 CURRENT dependencies')
audit_temporary_pins = {name: pin for name, pin in audit['dependency_raw_sha256'].items() if Path(name).is_relative_to(Path('/tmp'))}
exact(audit_temporary_pins, {name: row['sha256'] for name, row in expected.items() if name not in gate_tmp_paths}, 'ALL four audit-recorded CURRENT TMP dependencies plus four CURRENT gate receipt dependencies')

# Preflight every existing target before creating any missing target.
missing = []
for name, row in sorted(expected.items()):
    target = Path(name)
    if os.path.lexists(name):
        assert target.is_file() and not target.is_symlink(), 'wrong existing CURRENT TMP target: '+name
        exact(raw(target), row['sha256'], 'wrong existing CURRENT TMP bytes: FAIL, never overwrite '+name)
    else:
        missing.append(name)
for name in missing:
    row = expected[name]
    with Path(row['permanent_copy']).open('rb') as source, Path(name).open('xb') as target:
        shutil.copyfileobj(source, target, 1 << 20)
for name, row in expected.items():
    exact(raw(name), row['sha256'], 'CURRENT restored/already-present TMP bytes: '+name)
    closed_file(Path(row['permanent_copy']), row['sha256'])
exact(raw(manifest_path), MANIFEST_RAW, 'closed manifest unchanged after recovery')
exact(raw(old_manifest_path), OLD_MANIFEST_RAW, 'old73 manifest unchanged after recovery')
for path in (wrapper_path, audit_path):
    closed_file(path, manifest['files'][str(path.relative_to(ROOT))]['sha256'])
assert all(not (name == 'tectonics' or name.startswith('tectonics.') or name == 'tests' or name.startswith('tests.')) for name in sys.modules), 'no project/test imports'
print(json.dumps(dict(status='RESTORED_ONLY_MISSING_CURRENT879_TEMP_CLOSED_COPIES',
    current_temp_dependency_count=8, restored=missing, already_valid=sorted(set(expected)-set(missing)),
    failed_gate_historical_tmp_paths_restored=False, regenerated_baseline_or_physics_or_tests=False,
    project_imports=False), sort_keys=True))
PY
```


### Чтение закрытых артефактов без повторения физики

После восстановления зависимостей команда проверяет точный source pin
постоянной копии и запускает только stdlib reader. Указан новый output
`/tmp/native_common_material_surface_transport_followup_readback.json`;
если он уже существует, выберите другое отсутствующее имя. Reader создаёт
output исключительно через `xb`, прежние receipts не перезаписывает.
В оба `--document` передаются окончательные опубликованные файлы.

```bash
cd /workspace/Habitable-Moon-Project
python - <<'PY'
from pathlib import Path
import hashlib
reader = Path('analysis/geometric_native_common_material_surface_transport_final_readback/readback_source.txt')
assert hashlib.sha256(reader.read_bytes()).hexdigest() == 'f23843c4065d8d8bd4095a9f0b1ecbf948386e92ea1047506f09104bbf7a66f1'
PY
python analysis/geometric_native_common_material_surface_transport_final_readback/readback_source.txt \
  --manifest analysis/geometric_native_common_material_surface_transport_validation/artifact_manifest.json \
  --document GEOMETRIC_NATIVE_COMMON_MATERIAL_SURFACE_TRANSPORT_IMPLEMENTATION.md \
  --document CLOUD_HANDOFF.md \
  --out /tmp/native_common_material_surface_transport_followup_readback.json
```

Reader проверяет карту 879 и старый subset875, component manifest83,
старый manifest73/180 raw pins, десять ancestry release closures, полные
сохранённые DTO и связи запросов. Сохранённый baseline1633 учитывается как
историческая квитанция до независимой математики; его solver/tests не
исполняются вновь. Новая квитанция отдельно связывает bytes двух документов.
Фактические status/time/result pins исходной финальной проверки остаются
в связанном `readback.json`; приведённая команда их не подменяет.


## Остаток пункта1

Пункт1 открыт: **два прежних блока завершены; один блок остаётся с теми же
тремя условиями**. Внутри первого условия компонент879 добавляет локальный
finite field/Nanson measure/current-memory transport. Это не завершает
whole-source physical force/prestress/work handover.

1. Физическая связь с текущими телами/архивами и настоящими COMMON faces:
   связать полученные finite `F`, local surface measure и current memory с
   once-owned source work/energy и переносом сил/prestress; согласовать
   column/bulk energy. Затем закрыть HOT law/finite mean chart, полный
   физический archive handover, loads/drag и NEW mixed equilibrium.
2. Whole-clock all-pair admission, включая reverse C5→C1.
3. Принятое продолжение с независимой path work, балансами и joint restart.

Следующий физический шаг — использовать конечный локальный перенос
в source-force/work/energy handover с явным material/reference/measure и
once-only ownership contract. Новые этапы к прежнему плану не добавлены.
Принят **один прежний / ноль новых** интервалов. Пункты2–6 остаются открытыми;
объём остатка текущего пункта указывать в каждом отчёте.
