# Короткий совместный сферический пилот завершён

База `9bb3f0178b6e8d32ff36367ed313d7948e38d379`.
На320 гранях подключены существующие native thermal/phase ports, implicit
enthalpy, thermoelastic work, Maxwell-память, whole-cell birth, momentum
capture и edge energy reserve. Новый экспериментальный adapter использует
сферические mass/strain/basal operators. Законы, beta и физические пороги
не изменены; defaults/canonical и старые checkpoints сохранены.

**Результат — технически воспроизводимая связь компонентов.** Пространственная
сходимость не доказана (прежние30.6% load-response discrepancy остаются);
кроме того, интегральное drag heat пока заметно зависит от шага. Это не
подтверждение физического прогноза разломов или долгой геологической эволюции.

## Явный отдельный перенос

Исходник `/workspace/genesis-angular-coupled-01/initial.json`, SHA256
`ae107b4a2a5725c0e9b7eece15de71f234460de7e7d5f5ae6ef59616510ba590`.
Использована уже принятая minimum-angular-variance реконструкция исходной
истории, не выдумана недостающая angular history. Перенос интерполирует
поле strain в исходных angular/radial координатах, переводит engineering
shear через настоящий тензор в мировую систему и вычисляет его в сферических
квадратурных точках.16 тыс. точек памяти — локальные значения,972 глобальных DOF.

Thermal supports и phase/material фрагменты получены существующим
conservative refiner; равные четверти заменены spherical-area fractions.
Глобальные reservoirs не размножены, source IDs сохранены как parent IDs.
Начальный clock/source history не обнулён. Старые800 cohorts представлены
3200 фрагментами, это не новые рождения.

| Измерение переноса | Результат |
|---|---:|
| Relative thermal mass change |1.29e-19|
| Relative enthalpy change |1.25e-19|
| U до/после, J |3.1893236255915614e19 /3.1893236255915622e19|
| Signed ΔU |8192J, relative2.57e-16|
| K до/после |5014.98 /64579.66J|
| Restricted generalized momentum error |2.72e-15|
| Physical linear/angular momentum error |2.81e-13 /1.26e-12|
| Изменение restricted internal force |6.2381%|
| Изменение restricted external force |14.8480%|
| Mapping heat |ровно0|

Velocity prolongation сохраняет значения/радиальную производную исходного
FE поля в новых вершинах, затем корректируется для сохранения исходного
restricted generalized momentum. Геометрическое изменение M делает K
неинвариантной: **+59564.68J явно является ценой инициализации**, не нагревом
и не скрытым изменением старого thermal H. Сохранение U не означает
сохранения всех внутренних сил или восстановления неизвестной истории.
Физические resultant momenta независимо проверены в
`spherical_coupled_momentum.json`; их вычисляет `rigid_momenta()` runner.

## Повторное использование подсистем и численная устойчивость

`geometry_provider` — необязательный hook существующего native advance для
material M, moving basal assembly, birth mass и preparation phase inertia.
При отсутствии hook прежний путь остаётся прежним. Spherical-tagged state
отказывается идти через старые плоские операторы. Thermal ports, water EOS,
8-iteration thermal corrector, фазовое admission и capture не переписаны.

Moving basal geometry обновляет points/normals/area из того же spherical map
и displacement gradient, а после admission — из выросшей несущей массы.
Beta1e14Pa s/m, seed/amplitude, transmission law и damage thresholds прежние.
Геометрия и текущая толщина меняют дискретную внешнюю силу; её изменение
выше измерено, а не скрыто под утверждением о неизменном nodal F.

Newborn strain memory создаётся строго нулевой в50 angular/radial samples;
на следующих шагах используется тот же Maxwell и thermal-work путь.
Collapsed radial memories поддерживают старые stress/edge diagnostics;
энергия и release всегда вычисляются по полной квадратурной памяти.

Адресный thermoelastic test на малом dt выявил потерю точности Cartesian
матричного drag action при почти нормальных движениях. Допуск не ослаблялся
после отказа. В экспериментальном вызове используется положительное
quadrature factor action и до3 residual corrections линейного решения.
Dense D остаётся preconditioner/assembly; физический закон не изменён.
Опции в existing core выключены по умолчанию, включая сохранение прежнего
порядка арифметики default drag energy. Factored work проверен известным
линейным решением. Максимумы в принятой real-серии:
mechanical energy relative1.01e-14, momentum relative8.06e-10.

## Зафиксированный пилот и результат

Протокол до запуска — `SPHERICAL_COUPLED_PROTOCOL.md`:
0.0625year, dt0.03125/0.015625year,2+4steps,320 граней;
180s solve/360s total budget. Ни расширения горизонта, ни сгущения.

Fine final: max ΔT0.00821760K, max Δold-D3.1913e-7,
max face-effective-D0.0196432. Реальных admissions0; progressive solid
mass в mush+6.29619e14kg. Shell остаётся одной, `NOT_SEPARATING`.
Native latent heat повторно не добавлялась. Thermal mass arrays неизменны
при каждом принятом шаге; H меняется только по действующему ledger.

Максимальный whole residual3.4108% разрешённого native allowance,
который учитывает ulps огромного H и физический energy scale. Это не
заявление о точности3.4% всей физики. В конце fine U3.104455020009e19J,
K7.90276e5J. Maxwell heat1.113636216171e15J, thermal work3.482300657072e14J,
drag heat1.368661646847e16J, edge release2.291720111999e17J уходит в
механический reserve, edge heat0. Numerical losses не зачислены в тепло.

Контроль шага в конце интервала:
world mean stress relative0.071410%, max T difference1.12757e-6K,
old-D5.45e-10, face-effective-D0.000215546. На первом общем времени
stress difference0.104788%. Это норма полного напряжения с унаследованной
памятью, **не** новая оценка пространственной точности малого mantle signal.

**Существенная временная граница:** coarse drag heat8.8414724e15J против
fine1.3686616e16J, разность35.4006%. Fine numerical-memory dissipation
6.0506276e17J — примерно40.9× суммы физического Maxwell+drag heat.
Начальная перестройка сильно гасится BE; близкие полные stress/T этого
не опровергают. Нельзя объявить энергетический переход временно сошедшимся.
Следующий контроль, если продолжать, должен разрешать этот стартовый
переход по энергии/работе с адаптивным шагом, а не только по полной T/stress.
Дополнительный прогон сейчас не запускался.

Независимая загрузка checkpoint после первого fine шага и следующий шаг
побитно совпали с прямым продолжением полного model state.

## Отдельный фазовый контроль

В отдельной копии старта первый phase support охлаждён до solidus−1K;
явно внесено **ΔH=−1.285513304235e24J**. Это искусственная подготовка,
не событие реальной траектории. Admission:1 cell, mass3.631486220316e18kg,
без добавления mass/H inventory или повторного latent heat.
Initial newborn memory ровно0. Capture heat1.097239155488e10J,
generalized momentum error3.73e-16. На следующем шаге norm памяти4.35481e-6,
повторных admissions нет. Restart после birth также побитный.
Максимальный whole residual fraction этого контроля1.711% allowance.
Preparation и оба checkpoints находятся отдельно от реального пилота.

## Скорость, проверки и deliverables

Перенос16.87s. Первичный benchmark совместного шага5.90s; после устойчивого
factored solve реальные принятые шаги6.57–8.21s (thermal corrector2 iterations).
Вся серия111.94s, solve вместе с restart и искусственным birth71.13s —
в пределах обоих заранее заданных бюджетов. Эти времена относятся к320
и текущей аппаратуре; перенос на геологические сроки с фиксированным
малым dt практически непригоден. Адаптивное время ещё не реализовано.

19 адресных/регрессионных тестов прошли за26.32s:
`test_spherical_coupled.py`, `test_experimental_angular_coupled.py`,
`test_genesis_aggregate_coupled.py`, `test_simplified_genesis_nodal.py`.
Дополнительно повторён guard test после явного запрета старого geometry
fallback:1 passed5.92s. Проверены thermoelastic work, modulus-change heat,
полная energy release, linear solve и native default regressions.

Численный отчёт `spherical_coupled_check.json`; полные новые файлы в
`/workspace/genesis-spherical-coupled-01`. Fine final56723963 bytes, SHA256:
`0f79f9d10d4db9308228f1f0a13cc968e8b5425eb0f60fa83b1fb9e9435b67a1`.
В отчёте SHA256/размеры всех initial/restart/final/control/fields файлов.
Checkpoints сохраняют recipe/source hash, config, full quadrature memory,
thermal/phase state и mapping ledger; exclusive save защищает предыдущие.

Воспроизведение: Python `/workspace/habitable-moon-venv/bin/python`, env
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1`.
Старт создаёт `SphericalCoupled.initialize()` и exclusive `.save(new_path)`;
runner `-m analysis.genesis_geological_handoff.spherical_coupled_check`
использует сохранённый preflight initial и требует новый OUT. Старые
checkpoint bytes, defaults/canonical и архивы не переписаны; push/deletion нет.

Оставшиеся ограничения конкретны: неизвестная исходная angular history,
изменение сил/K при geometry mapping, spatial stress accuracy30.6%,
несошедшийся стартовый drag-energy integral, reference-chart/small-strain
пределы, whole-cell birth и mesh-dependent edge weakening. Contact evolution
после принятого split отсутствует. При выполнении цели «одна оболочка →
самостоятельный разлом → геологическая эволюция» эти ограничения нельзя
заменить ссылкой на machine-level balance. Техническая интеграция теперь
реализована и воспроизводима; физическая готовность не заявляется.
