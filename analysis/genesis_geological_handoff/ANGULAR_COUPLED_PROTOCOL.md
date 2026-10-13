# До короткой совместной серии

База1f7176f2c8fce97039cf1570ea4a82245192ff05. Начало — новый сохранённый
/workspace/genesis-source-angular-01/initial.json, а не frozen1yr final:
у initial нет нераспределённой термодинамической истории и heat buffer=0.
Реконструкция угловой памяти остаётся прежней явно приближённой.

Повторно используются native thermal ports/orbit, implicit enthalpy graph,
water EOS, temperature-dependent Maxwell rates, old damage/healing/moduli,
thermoelastic work, moving-basal drag/traction и phase admission/capture.
Тепловые поля по углу остаются общими для трёх точек каждой исходной
тепловой опоры — это существующее разрешение thermal graph, не новая
неизвестная температурная структура. Новый материал получает3 нулевых
угловых памяти по существующему zero-stress birth, без выдуманной предыстории.

Maxwell+drag−thermal_work−old_damage_elastic_change входят в H один раз;
кинематический capture входит один раз. Edge-release идёт в mechanical
reserve, численные потери остаются численными. Frozen heat buffer не импортируется.

Первый замер80faces/.0625year:0.258s,2 thermal corrector iterations.
План:1year при dt=.125/.0625;24 штатных шага, один restart, отдельный
контролируемый birth-case с2 шагами. С подготовкой/сохранением оценка<30s,
cap120s.5120faces и долгая эволюция исключены.

До решения: max common-time world-stress relative difference<5%,
edge-D absolute difference<.02, old-D difference<.002, T difference<.05K;
phase admission count должен совпадать на общих временах. Весь native
энергетический/геометрический/фазовый допуск обязателен, не подбирать его.
При несовпадении остановить длинное продолжение, записать фактические числа.
Birth в контроле явно пометить как подготовленное thermal initial condition,
не как случившийся в реальном интервале. Сохранить source/checkpoints/defaults.
