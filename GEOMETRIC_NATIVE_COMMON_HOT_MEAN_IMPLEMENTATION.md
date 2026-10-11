# Полный HOT-закон и конечное среднее основания

Ветка `fix/genesis-mantle-convection`. Компонент903, 07.10.2026 МСК.
Scope: `DECLARED_COMPLETE_HOT_FINITE_BASE_MEAN_ENDPOINT_ONLY`.

## Результат и границы

Реализованы полный материальный закон пяти исходных HOT-тел и конечный chart
с30 координатами: физическое среднее радиальное смещение основания и пять
бесследовых амплитуд каждого тела. Получены энергия,30 обобщённых внутренних сил (градиент энергии) и полный Hessian
30×30, включая вторые производные неявного среднего. Это компонент для
следующей совместной силовой сборки. Равновесие, перенос старых сил/работы,
работа подготовки reference, whole-clock all-pair admission и новый принятый
физический интервал не удостоверены.

Полный холодный компонент899 используется по закрытым сохранённым источникам.
Его энергия,56 сил и Hessian не пересчитывались. Whole-system mixed response
из холодного, HOT, контактов и нагрузок этим выпуском ещё не собран.

## Подлинные источники и геометрия

Сохранены все пять birth/mask/foundation состояний, исходные material/parent/
owner IDs,100-coordinate catalogue, material/memory/work histories и часы.
Подлинные HOT30 amplitudes в891 ровно нулевые. НенулевыеCOMMON Cartesian
пробы и два ненулевых old19-foundation результата остаются непринятыми
диагностическими состояниями. Они не заменяют исходное принятое состояние.

Все исходные площади строго положительны, в том числе три около1e-47 sr.
Каждый настоящий native polygon входит целиком. Положительная paired curved
поправка HOT295 учитывается один раз со своей площадью около3.9161e-72 sr.
Containing chart ограничивает integrand; его площадь не назначается материалу.
Параметризация настоящей границы этой кривой в источниках отсутствует и не
объявляется восстановленной. Сумма площадей polygons+delta согласована с
подлинным total area. Никаких area floors или удаления малых тел нет.

Новое подготовленное размещение `X0=s_owner*Qshared(z)*r_original(z)*u_world`.
`r_original^3=R^3-3R^2*z`, R100000m, HOT areal thickness2000m, roofs0.
Сохранены фиксированные owner scales895 и общий depth rate879. Native polygon
pose применяется один раз при восстановлении source-world ray; сохранённая
owner history служит источником истории и не добавляет второй поворот.
Отдельно сохранены исходные HOT body и foundation DTO. Этот scale reference
не объявляется бесплатным физическим переходом или новым допуском тел.

Prepared original-material Jacobian `s^3`; после деформации — `s^3*det(F)`.
Плотность переносится как `rho_original/(s^3*det(F))`. Нормосохраняющий depth
rotation, монотонный радиус и постоянный положительный F дают инъективность
каждого целого HOT-тела. Проверка зазоров со всеми COLD/HOT телами остаётся
открытой; индивидуальная инъективность не выдаётся за all-pair admission.

## Закон, среднее и производные

`F=(1+h/R)*I+T(b)/R`, T симметричная бесследовая матрица пяти амплитуд.
Она задаёт одно поле для bulk и основания. СтарыеCartesian mode labels
`xx,yy,zz,xy,xz,yz` проверены по полному оригинальному body/kernel/catalogue.

Bulk: `E_Green=(F^T*F-I)/2`,
`W=mu*(E:E)+lambda*(tr(E))^2/2`, mu=lambda4e8Pa из подлинных E1e9/G4e8.
Это явно объявленное изотропное StVK-продолжение исходных коэффициентов.
Интеграл берётся по полному исходному материальному объёму `R^2*dOmega*dz`;
новый current-volume множитель и старые scalar foundation energies не добавляются.
Исторические 3D prestress/work поля этим продолжением не восстанавливаются.

Finite mean: `q=rb_prepared*mean_actual_mask(|F*v|-1)` на настоящем основании,
с нормировкой подлинной полной площадью. q заменяет trace coordinate h;
седьмой неплатный DOF не вводится. Mean не является reference-normal
projection или средним по объёму. При b0 точная ветвь `h=R*q/rb_prepared`.

Basal law: `.5*rho*g*R^2*integral_actual_mask([rb_prepared*(|F*v|-1)-birth]^2)`.
rho2800kg/m3, g1.5m/s2, birth0. При b0 восстанавливается исходный Winkler
коэффициент для q. При b!=0 сохраняется вся радиальная дисперсия, включая
случай q0 с положительной энергией. Это NEW objective finite-radial extension;
прежний reference-normal affine law сохранён отдельно как исходный receipt.

Mean root ограничен глобальным SPD bracket с rho_T>=norm(T).
`M_h>=(rb/R)*lambda_min(F)/lambda_max(F)>0` доказывает единственность.
Whole-mask interval Newton с внешним округлением сужает настоящий bracket.
Midpoint используется только как Newton trial, не как физический root.
Полные first/second implicit jets: `h_q=1/M_h`, `h_b=-M_b/M_h`,
`h_ab=-(J_a^T*Hess(M)*J_b)/M_h`.
Силы `J^T*g`, Hessian `J^T*H*J+g_h*Hess(h)` сохраняют force-curvature term.

Выпускные integrals ограничивают каждый целый настоящий polygon/correction
через certified cone ranges и его собственную площадь. Fraction arithmetic
и направленное округление сохраняют весь support. РазностьGauss интегралов
не используется как доказательство границ. Cone/root/force bounds могут быть
широкими; для mixed solve их достаточную точность требуется проверить.

## Проверки и сохранения

Fresh899 evidence-only baseline PASS27.155103914s выполнен до добавления новых
источников. Frozen903release02: **16/16 tests PASS64.98s** (same-process phase
68.037924374s); четыре полных HOT30 запроса и15 material
label queries **nativePASS19.523803725s**; gate **PASS124.837578361s**,
exit0 (outer wrapper132.413205726s). Полные system/response restart проверены.

Независимый аудит **PASS424.318071865s**, exit0 (outer wrapper
424.470972374s),220 decimal digits: **6324 scalar comparisons всего** —
6144 operator containments (силы/Hessian, mean roots иimplicit jets) и
180 point-channel comparisons для15 literal material labels. Все1441 raw
pins и весь903 source map проверены BEFORE/AFTER. Native source geometry
восстановлена независимо из literal planes иRodrigues; integrals считают
целые сферические triangles правилом16/32. Изотропные постоянные каналы
используют polygons+literal positive correction area INTERVAL. Для остальных
curved moments сохраняется containing-chart uncertainty; midpoint material
не создаётся. Actual stdout байт-в-байт совпадает с полным274936B report;
stderr пуст. Gauss difference — диагностическое сравнение, а неnative proof.

Границы честно широки: max nonzero mean-root width≈.489714458m,
force width≈3.62723e7N, Hessian width≈2.18619e8N/m. Они удостоверяют полный
компонент, но не высокоточный solve. Изотропная ветвь среднего имеет width
до1.98e-100m. Достаточную точность cold/HOT сил для mixed equilibrium нужно
получить внутри того же первого условия первоначального пункта1.

Failedgate01 сохранён целиком:15/16PASS; synthetic large-amplitude fixture
не достиг установленной ширины1e-12m после12 iterations, хотя exact root/J/H
containment прошёл. В единственном изменённом тесте сохранён бюджет12 и
добавлен бюджет24 с проверкой вложенности и строгого прежнего порога.
Core/native/validate не менялись; native остаётся12iterations.
Сохранены дваfailed auditor attempts с executing sources, фактическими
stdout/stderr/exit facts иbefore/after failure preservation. Первый добавлял
искусственный запас≈1e-90 к точному isotropic Mh с width≈2.397e-99;
второй не учитывал delta mass в angle-independent Hqq и добавлял слишком
широкий запас. Исправлен только workspaceTXT auditor; PY/gate02 не менялись.
Предварительные70 independent isotropic checks после исправления такжеPASS.


Все899 прежних Python источников,899P36/F4,895P50/F4,891P44/F4,887P41/F4,
341raw dependency pins/15TMP associations/11ancestry closures сохранены.
Добавлены ровно четыре Python файла: core,tests,native_control,validate.
Закрытые источники и артефакты не переписывались. Сохраняются все failed
attempts с executing snapshots и фактическими stdout/stderr. Документы
завершены до evidence-only final reader; его результат и seal находятся
в linked readback/archive. Проверка закрытия не запускает старую математику.

При входе остановлен оставшийся диагностический kernel прошлого шага899;
он не писал в закрытые файлы. Неработающие zombie записи не потребляют CPU.
Старые force solves/tests/tensor quadrature controls не перезапускались.

## Остаток первоначального пункта1

Два прежних блока выполнены. Остаётся ОДИН существенный блок с ТЕМИ ЖЕ
тремя условиями; исходный шестипунктный план не расширен:

1. Whole-source material/old-force/prestress/work handover, HOT law/finite
   mean, loads/drag и NEW mixed equilibrium. Cold899 и HOT/finite-base-mean903
   constitutive компоненты теперь готовы. Остаются физический перенос
   исходных сил/prestress/work и работа reference, normal damage/follower
   law, нагрузки/drag и новое совместное равновесие с достаточной точностью.
2. Whole-clock all-pair admission, включая reverse C5→C1.
3. Принятое продолжение, независимая работа всей системы, балансы и joint restart.

Следующий конкретный шаг внутри условия1: согласованная сборка cold56,
HOT30, owner/internal/contact координат и нагрузок/drag для NEW mixed solve.
Finite mean chart сам по себе не означает finite mean equilibrium.
Остаток существенный, а не последние одна-две проверки.

Принят1 прежний и0 новых физических интервалов. Часы source0/contact2e6/
memory1e6/material1e6 секунд сохранены. Новые damage/slip/work не приняты.
Пункты2–6 первоначального плана остаются открыты.

## Доказательства и повторное чтение

- [Gate02](analysis/geometric_native_common_hot_mean_validation/release_validation_02/report.json)
- [Full native packet](analysis/geometric_native_common_hot_mean_validation/release_validation_02/native_control/source_packet.json)
- [Independent audit](analysis/geometric_native_common_hot_mean_validation/independent_audit/report.json)
- [Whole manifest](analysis/geometric_native_common_hot_mean_validation/artifact_manifest.json)
- [Final evidence readback](analysis/geometric_native_common_hot_mean_final_readback/readback.json)
- [Closed final archive](analysis/geometric_native_common_hot_mean_final_readback/archive.json)

Frozen903map canonical SHA256
`c6c36bdaa67772f3ef02e38195c27e0e6326b2d275170fff6d26dd8d40bb4012`;
raw `a0a406a60ebbd0b7f90444aa26330f7272bd9afdd528ee746c898fb56fbbec05`.
Gate seal `4c9ac2e0bd93478fe96839a4cbf2b0d4f32dc258a734a50a244d9472141107ca`.
Native packet raw `0f8a0ee8ae082e3434b136bab7515944354eaf059db95965d0e87259f6791035`;
seal `c53e4cd75219328eeac974f3b9982aa33ad78eaf2e07d823feafb9f07b289e6c`.
Audit seal `c74a66f928078e05bf537362ba304ae4c70244689469cede090ddd81ea55bd0f`.

Final readback/archive являются авторитетным результатом закрытия;
число файлов, executing reader pins иselfseals берутся из них.
Документы и все необходимые материалы завершены ДО запуска final reader.
После закрытия903P/F не дополнять и не переписывать.

Следующая облачная среда: сначала восстановить только недостающие exact
outside staging/TMP aliases из закрытых копий, затем свежо прочитать903.
Restorer stdlib-only проверяет closed903F4,903manifest ивсе исходные pins,
восстанавливает42 outside paths иотказывается заменять неверные existing
bytes. Отсутствовавший permanent copy старого887 preservation receipt
сохранён как `inherited_final887_preservation.json` в903P. Никакой старый
main/solver/test/integrator для восстановления не исполняется.

```bash
python analysis/geometric_native_common_hot_mean_validation/baseline_restore_source.txt --restore --output /workspace/hot_mean_restore_next_cloud_20261007.json
python analysis/geometric_native_common_hot_mean_final_readback/readback_source.txt --output /workspace/hot_mean_fresh903_readback_next_cloud.json
```

Использовать новые exclusive output/capture paths. Сначала получить fresh
closed903 reader PASS иbefore/after сохранность, затем добавлять следующие
источники. Следующий компонент additive; `.git` в этой среде readonly.

