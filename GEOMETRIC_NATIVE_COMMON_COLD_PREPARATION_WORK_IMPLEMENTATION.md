# Холодный закон, работа модального пути и движение подготовительной карты

08.10.2026. Ветка `fix/genesis-mantle-convection`. Компонент919:
четыре новых Python-файла поверх неизменённых915. Перенос915 теперь
сопровождён подлинным холодным нелокальным законом и отдельным гладким
кандидатом физической подготовки. Полная работа подготовки ещё не оплачена.

Сохранён весь объявленный закон899, а не заменён локальной упругой плотностью.
Для `E=H*t+R`, `r=∫H:C:R`, `S=∫R:C:R/2`, фиксированного подлинного
`A=M^-1` и восьми буквальных beam ledgers:

```
p = t + A*r
U = Σ E_literal + f_literal·p + Σ K_i*p_i²/2 + S − r·A*r/2
```

Получены энергия, все первые и вторые производные для произвольных
каналов моментов. Сохранены общая проекция восьми банков, её межбанковые
связи, исходная объёмная мера и кривизна, сопряжённая преднапряжению.
В Hessian входят `f·t_ij` и `(A*f−A*r)·r_ij`. Восемь исходных энергий
входят ровно один раз; ненулевая историческая 3D archive prestress не
восстановлена этим новым объявленным законом.

Для прежних подлинных zero/registered запросов899 восстановлены полные
56×56 операторы непосредственно из сохранённых целых интегральных моментов.
Сохраняется split tip/remainder: независимые interval copies `A*M` нельзя
подставлять вместо точного равенства I. Source raw/canonical seals, все
восемь материальных банков и три непустые архивные когорты проверяются.
Новые supplied moments — явно объявленные, без удостоверения интегрирования.
100-channel curvature fixture — проверка алгебры, не новый mixed equilibrium.

## Работа подлинного модального пути

На существующем895 reference путь `a(λ)=λ*a_registered`, `0≤λ≤1`,
остаётся в удостоверенном895 cold envelope. Green strain квадратична,
проекция r квадратична, S и энергия U — полиномы до четвёртой степени.
Коэффициенты восстановлены из saved899 moments и их directional derivatives.
Определены энергия, производная, вторая производная и интеграл работы
на всём пути. Работа положительного градиента равна `ΔU`; работа
физической реакции имеет противоположный знак. Полные source/moment/
response/path DTO восстанавливаются с буквальными типами и exact replay.

Это работа холодного модального движения в уже подготовленном reference,
а не работа устранения расхождения базовых карт. Подлинный whole895 box
позволяет максимум2.183mm donor/1.854mm receiver; им нельзя закрыть889.604m.
Доказательство исходного объёмного интегрирования899 наследуется, не
выдаётся за новый quadrature control. Временной допуск/часовые балансы отсутствуют.

## Отдельное физическое движение карты

Реализован один гладкий map для всего холодного материала владельца,
включая архив, с подлинными общими R/Ω и фиксированным scale s:

```
τ(x) = (R³ − (|x|/s)³)/(3R²)
Tλ(x) = exp(λ[Ω]τ(x)) x
DTλ = Qλ (I + λ(Ω×x) ⊗ ∇τ)
∇τ = −|x| x/(s³R²)
```

Радиус сохраняется, `det DT=1` точно: `(Ω×x)·∇τ=0`.
Обратная карта использует тот же радиус и противоположный угол.
Композиция с допустимой895 картой сохраняет её injectivity/local J;
это не all-pair admission и не доказательство холодно-горячего контакта.
Одинаковый map для всех холодных банков владельца сохраняет их швы.
Побанковая замена base на old-roof COMMON map была бы разрывной: подлинные
старые крыши внутри одного owner различаются.

Доступны положение, DT/обратный DT, пространственный Hessian,
λ velocity/acceleration, rate DT и перенос всех56 мод/их original gradients.
Последний содержит `D²T[v,A]`; одного вращения градиентов недостаточно.
Связанный API принимает зарегистрированный895 carrier и общий reference;
чистый declared API не присваивает материальной маске удостоверение.
Нельзя применять движение к HOT, который уже имеет COMMON depth reference.
Для свежих owner rotations подготовка задаётся в исходном prepared world
frame, а fresh owner motion применяется снаружи; fixed Ω не остаётся
неподвижной мировой осью при смене общего механического frame.

Положения реально перемещаются; это не relabel915 и не новая stress-free
reference. Для `d=(Ω×x)⊗∇τ`:

```
ΔE_relative(λ) = λ sym(d) + λ² d^T d/2
ΔE_original(λ) = A_old^T ΔE_relative(λ) A_old
```

Это приращения старой деформации. Они дают основу для интегральных
моментов нового пути по тому же899 закону. Пространственный двухджет и
перенос первых модальных градиентов ещё не являются полным mixed tangent
для всех nonlinear mode compositions; его вторым вариациям градиента
потребуются соответствующие дополнительные производные.

На подлинной frozen913 точке `genuine_zero_hot_source/start`, HOT-depth1000m:
normal gap приλ0 `−889.603992527m`, приλ1/2 `−441.629054261m`, приλ1
**`−7.761031059·10^-8m`**. Зазор не обнулялся; малая отрицательная
penetration сохранена. Расстояние между сопоставленными точками в3D
**0.1000929324m**: малая normal projection не означает совпадения точек.
Это проверка одной подлинной точки, не всего curved C6–HOT C1 face.
[Точные интервалы](analysis/geometric_native_common_cold_variational_work_validation/preparation_gap_observation.json).

## Проверки и источники

Итог: **17 вариационных тестов PASS98.37s +18 map-тестов PASS35.57s**;
оба actualexit0/stderr0. Вариационный native PASS51.268325141s:
два authentic56 operators, объявленный100-channel fixture, пять λ states,
forward/reverse work и restart. Его независимый stdlib-аудит
**PASS23.456091697s,32987 exact comparisons,95 interval checks**, source
данные и operator algebra проверены отдельно, imports проекта отсутствуют.

Depth native **PASS82.229202480s**, три λ запроса исходного913 trace,
без замены material/root midpoint. Независимый220-digit mpmath.diff-аудит
**PASS9.897639119s,2578 enclosure checks,69 identities,108 composed-mode
controls**, все56 columns и Green coefficients. Его compact inputs
побайтно связаны с sealed913 источником. Оба настоящих audit stdout
совпадают с report, stderr пуст; исходники и execution facts сохранены.

Development01 вариаций:15PASS/1FAIL95.25s, сохранён. Повторное DTO
восстановление после отказа ошибочно снимало регистрацию прежнего идентичного
объекта. Исправлена дедупликация; development02 focused2PASS8.69s и final17
прошли. Добавлена проверка обхода frozen objects. Map16PASS22.01s до
расширения Green проверок и final18 сохранены отдельно. Неудача не заменена
успешным журналом.

Fresh915 restoration/readback и closed ancestry guard выполнены до новых
PY. AFTER guardPASS36.950118558s: все915 старых PY, закрытые915P37/F4 и
вся913/909/907 ancestry неизменны; source_count919. Старые force/quad/test
controls и математические mains не запускались. Документы завершаются до
final evidence-only readback/archive. Коммита и push не выполнялось.

[Gate](analysis/geometric_native_common_cold_variational_work_validation/report.json),
[variational audit](analysis/geometric_native_common_cold_variational_work_validation/audit_report.json),
[depth audit](analysis/geometric_native_common_cold_variational_work_validation/depth_audit_report.json),
[manifest](analysis/geometric_native_common_cold_variational_work_validation/artifact_manifest.json).

## Остаток исходного пункта1

Пункт1 открыт: один существенный блок, прежние3 условия:

1. Physical material/old-force/prestress/work handover, работа полной
   подготовки, loads/drag, точный general-state operator и NEW equilibrium.
2. Whole-clock all-pair admission, включая reverse C5→C1.
3. Accepted continuation с independent temporal path work, balances/restart.

Следующий конкретный шаг внутри первого условия: интегрировать **r1/r2 и
S2/S3/S4 именно пути Tλ** на всех исходных cold supports/архиве, затем
согласовать эту работу с891 local источниками, HOT/contact и нагрузками.
Работа прежнего модального пути не может быть зачислена как работа Tλ.
Нужны full curved receiving association/admission и достаточная force
precision для нового равновесия. Source0/contact2e6/memory1e6/material1e6s
не продвинуты; acceptedold1/new0, damage/slip не приняты. Пункты2–6 открыты.

В новой среде из корня репозитория сначала stdlib byte-restorer, затем
reader с разными новыми отсутствующими `--output`:

```
python analysis/geometric_native_common_cold_variational_work_validation/baseline_restore_source.txt --output /workspace/cold_preparation_restore_next.json
python analysis/geometric_native_common_cold_variational_work_final_readback/readback_source.txt --output /workspace/cold_preparation_readback_next.json
```

После закрытия919/915/913/909/907 validation/final_readback не дополнять.
TXT native/audit controls не запускать повторно для чтения байтов.
