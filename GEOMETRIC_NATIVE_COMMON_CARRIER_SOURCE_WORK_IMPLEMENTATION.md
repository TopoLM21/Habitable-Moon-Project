# Перенос локальных источников COMMON на совместный carrier материала

Дата: **07.10.2026, московское время**. Ветка `fix/genesis-mantle-convection`.

Область компонента: `DECLARED_ARCHIVE_CARRIER_REGISTERED_COMMON_COHESIVE_PREPARATION_ONLY`.
Это кандидат подготовки геометрии с явно объявленным начальным состоянием
нормального разрушения. Принятого продолжения физического времени здесь нет.

## Что реализовано

Все 32 исходные материальные метки, включая восемь архивных attachments,
связаны с подлинным carrier выпуска887. Сохранены исходные родители,
material IDs, owners и когорты; повторного списания архивного материала нет.
Полные исходные состояния материала из883/879 сравниваются с carrier887,
а не только по совпадению имён меток.

Старый градиент COMMON задан в собственной системе координат. Поэтому
новая производная относительно старого material reference равна
`T = A_original (B_old Q_owner)^−1`. Используются полные старые крыши,
глубинный сдвиг и упорядоченный owner pose. Подстановка одного
`A_original` дала бы неверные силы и смешанные производные.

На объявленном пути подготовки λ:

```
Xλ = Xold + λ (χcarrier − Xold)
Tλ = I + λ (T − I)
x = Xλ + Σ a Ψold
F = Tλ + Σ a Gold
y = x + F (lold + r vold)
lold = p0 − Xold
```

Свежие owner rotations применяются к точкам и принимающим механическим
директорам с полными первыми/вторыми вариациями. Получены локальные
силы, сопряжённые материальные тензоры, моменты и полный оператор
100 координат: 6 owner, 56 cold, 30 HOT и 8 внутренних source residuals.
Первые92 значения остаются подлинным зарегистрированным ненулевым
COMMON probe; неизвестный оператор при body=0 не подставляется.

В сборке ровно16 исходных контактных механизмов, 8 исходных изгибных
связей и 8 зарегистрированных объёмных remainders, каждый один раз.
Непрерывная плотность контактов из прежних атомарных площадей не создаётся.
Для всех32 локальных combined F/J проверяется положительный Jacobian.
Это проверка локальных меток, а не допуск полных деформированных объёмов.

## Раскрытие и работа разрушения

Новый верхний reference принимающей плиты раскрывает прежние сжатые
контакты. Продолжать старую сжатую квадратичную ветвь до этого положения
нельзя: она породила бы неподдержанную растягивающую пружину.

Используется исходный bilinear Mode-I закон с буквальными параметрами
`Kn=10^6 Pa/m`, `T=2·10^6 Pa`, `Gc=2·10^7 J/m²` и суммарной
column compliance `C=2·10^−6 m/Pa`. Контактное раскрытие g отличается
от полного свободного раскрытия d: `d=g+Cτ`. Получаются:

```
Keff = 1/(1/Kn+C) = 10^6/3 Pa/m
onset: g=2m, d=6m
failure: g=d=20m
τ(d) = Keff d                 при d<6m
τ(d) = T (20m−d)/14m          при 6m<d<20m
τ(d) = 0                     при d>20m
```

Стабильная ветвь размягчения существует без snapback. Интервалы,
пересекающие6m/20m, отклоняются: единственного классического Hessian
там нет. Неподдержанные friction/stick/unloading контексты не вычисляются.

Примитив Φ имеет производную τ и достигает Gc после разрушения.
`E_literal + Area·(Φ(d)−Φ(d_anchor))` сохраняет буквальный старый prestress
и исходную поправку округления энергии. При λ=0 весь100-coordinate
оператор сравнивается с сохранённым883 receipt без старого force replay.

На λ-пути работа положительного градиента равна изменению loading
potential; работа физических реакций имеет противоположный знак.
Отдельно считаются recoverable energy и irreversible fracture work:

```
Δ loading work = Δ recoverable energy + Δ fracture work
```

Hessian относится к монотонной loading envelope. Он не объявляется
оператором разгрузки или Hessian одной только запасённой упругой энергии.
В конечном новом размещении все16 нормальных контактов полностью
разрушены в предложенном состоянии: τ=0, tangent=0, damage=1 и
fracture work/Area=Gc. Это следствие закона раскрытия, а не сброс старого
преднапряжения. Геометрический normal driver каждого контакта положителен
на всём объявленном аффинном λ-пути, около7914,77–7914,79m.

**Ограничение истории:** исходные source DTO содержат память сдвига,
но не normal max-opening/damage history. Поэтому virgin Mode-I seed
объявлен НОВЫМ предположением. Восстановление прежней нормальной истории
разрушения не заявлено; ненулевые или иного типа начальные damage/opening
отклоняются. Предложенные damage/slip/fracture stores не приняты и не
переписывают старые состояния, memory и work ledgers.

## Проверки и сохранение источников

Итоговый frozen gate попытки02: **15/15 тестов PASS585,83s**.
**Native control4/4 queries /64 atom queries PASS81,902999375s**;
wrapper native phase81,955109064s. **Same-process gate
PASS688,864870870s**, exit0; pytest wrapper587,806478258s.
**Независимый аудит PASS17,074628386s**, exit0:32 физических
χ/F,32 контрпримера подстановке A вместо T,4 queries/64 atoms,
1408 первых и30976 вторых вариаций,1408 physical-work координат и
112 piecewise λ-integral segments. Geometry160 и AD100 decimal digits.
Ни core, ни прежний audit main не импортируются/не запускаются.
Audit selfseal `1f59076935909b49b06c3ba548a11c5892c8ce97289cac420763278896dc19e2`;
report и настоящий stdout побайтно одинаковы, raw
`13d4060320d96454d2ca8e2c4c307649abab4e9fd42c1d082fbd5e86866af70e`.

Первый запуск auditor остановился ДО математики: общий rglob включил
Python-файлы установленной .venv. Второй использовал подлинные четыре
source roots tectonics/tests/scripts/analysis. Сохранены executing TXT,
actual stdout и observation первой попытки; Python-источники891 не менялись.

Frozen891 canonical `4fe39c88e8a0f00c0380bf68f0ccdb195db341ea84bdaa3e587560606f840fcc`,
raw `1491b406a5e373afc95112bb14947d709053639e2576570997a7db1a24f8c08e`.
Native packet raw `b45d5d4d80c74c509bfac033c4a60c5b96c8327704e0c727fcf680406d82e57e`,
selfseal `1f1b07b977bbb092e08281d8d200a26f8bf9ab85e2a18f001b2061275f742c40`.
Gate selfseal `176d4073301771b7b60e0304d17b91b05e4165bebf41def4b889d8ecc9108101`.

Попытка01: **8/15 PASS, 7FAIL**. Guard ошибочно сравнивал cohort
всего родителя с архивной принадлежностью конкретной точки. У четырёх
живых меток родитель имеет непустую когорту, а attachment cohort=None.
Исправлена только семантика guard: отдельно проверяются parent cohort,
фактическое live/archive и точное списание0/1. Все FAIL/logs/frozen891/
executing snapshots и preservation receipts первой попытки сохранены.
Прежние887 исходников не изменены; в повторной попытке менялись только
новый core и его тест (проверка подлинного старого reference).

Только четыре новых Python: core, его тесты, native_control и validate.
Все887 прежних Python и прежние закрытые P41/F4, 341 raw dependencies,
15 временных associations и11 ancestry closures проверяются до/после
нового gate. Прежние force solves, тесты и tensor-quadrature controls
не запускаются. Public saved material/geometry restoration и свежая
сертификация32 first-entry point contexts выполняют геометрические проверки.

Независимый auditor восстанавливает χ/F, полный старый chart inverse,
owner/body/internal twojets, силы/моменты и piecewise λ-work своей
арифметикой; его main не импортирует проект. Final reader проверяет
только сохранённые байты, seals и closure, не запускает физику.

## Остаток исходного пункта1

Пункт1 открыт. Два прежних блока завершены; остаётся ОДИН существенный
блок с ТЕМИ ЖЕ тремя условиями:

1. Физическая сборка всего исходного материала и архива, перенос сил/
   prestress/работы, HOT law и finite mean chart, нагрузки/drag и новое
   смешанное равновесие. Теперь локальные32 источника перенесены в
   объявленном endpoint-кандидате и работа их раскрытия учтена. Остаются
   полные материальные поддержки и история повреждения, current follower
   law, HOT/mean, нагрузки/drag и новое равновесие.
2. Допуск всех пар на всём временном интервале, включая обратную C5→C1.
3. Принятое продолжение, работа всей системы, балансы и совместный restart.

Следующий шаг внутри условия1 — проверить и собрать полные ненулевые
cold material fields на этом размещении; локальные32 положительных J
не заменяют whole-support проверку. Затем HOT/mean, нагрузки и новый
смешанный solve остаются частью того же условия. Число условий не выросло.

Принят1 прежний и0 новых интервалов. Часы source geometry0,
contact geometry2e6, memory1e6 и material geometry1e6 секунд не продвинуты.
API source подготовки ограничен подлинным journal endpoint `t=1`:
старые COMMON поля/ancestry не удостоверяют более ранние времена.
Пункты2–6 исходного шестипунктного плана открыты.


## Артефакты и продолжение

[Gate](analysis/geometric_native_common_carrier_source_work_validation/release_validation_02/report.json),
[native packet](analysis/geometric_native_common_carrier_source_work_validation/release_validation_02/native_control/source_packet.json),
[independent audit](analysis/geometric_native_common_carrier_source_work_validation/independent_audit/report.json),
[manifest](analysis/geometric_native_common_carrier_source_work_validation/artifact_manifest.json),
[final readback](analysis/geometric_native_common_carrier_source_work_final_readback/readback.json),
[отдельный закрытый архив4](analysis/geometric_native_common_carrier_source_work_final_readback/archive.json).

Свежий baseline887 до новых источников — PASS5,902696601s. Восстановлены
15 отсутствовавших TMP associations только из подлинных постоянных копий;
receipt, executing reader TXT и настоящий stdout архивированы в новом P.
Отсутствие TMP первой попытки baseline не выдаётся за успешный запуск.

Final reader — stdlib-only чтение запечатанных файлов. Его источник
содержит точные manifest raw/selfseal, проверяет891 фактический source map,
все четыре текущих output payloads (включая native report), snapshots,
старые P41/F4/341 dependencies/15TMP/11closures, docs до/после и caller
module identity. Папки P и F после закрытия не дополнять и не изменять.

Для повторного readback после закрытия, из корня репозитория:

```bash
.venv/bin/python analysis/geometric_native_common_carrier_source_work_final_readback/readback_source.txt --output /workspace/carrier_source_work_baseline_20261007/fresh891_readback_next.json
```

Выход должен быть новым отсутствующим файлом. Если workspace staging
не сохранился, восстановить его восемь baseline файлов из нового P
побайтно, с raw verification по manifest. TMP восстанавливать исключительно
по `temporary_alias_restoration.json` и прежним authenticated associations;
прежние reader main/solver/quad не запускать для обхода отсутствующих файлов.
Все четыре новых Python заморожены текущим release02. Следующий компонент
делать отдельными additive источниками; не переписывать891 или закрытые P/F.
