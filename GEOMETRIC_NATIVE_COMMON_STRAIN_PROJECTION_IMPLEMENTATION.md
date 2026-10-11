# Нелинейная проекция исходного механического источника на поля COMMON

Дата: 06.10.2026 (UTC). Ветка: `fix/genesis-mantle-convection`.

## Результат и граница выпуска

Статус выпуска: **PASS_COMPONENT_EVIDENCE_CLOSED**. Scope:
`COMMON_BANKWISE_COLD_STRAIN_PROJECTION_REDUCTION_COMPONENT_ONLY`.

Восемь внутренних каналов предыдущей системы сведены к полям восьми настоящих
cold bodies COMMON. Теперь используется 92 координаты: шесть общих owner
angles, 56 cold coordinates — по семь на тело — и 30 HOT coordinates — по
шесть на тело. Отдельных независимых `source_internal` coordinates в новой
системе нет. Исходные 14 координат, шесть owner angles и восемь radial channels,
получаются одной общей нелинейной картой `Pi` из этих 92 координат.

Связь с cold fields ненулевая; все 56 cold coordinates входят в каталог
проекции. Source coupling всех 30 HOT coordinates точно нулевая. Это реальная
редукция внутренних каналов, а не ещё одно расширение размерности. Выбранная
связь — новая нелокальная калибровка по деформации целого cold bank. Она ещё
не доказывает эквивалентность распределённых tractions на COMMON faces,
движение трёхмерного архивного тела или закон приёма контакта на HOT.

Сохраняются все 16 исходных контактно-колонных механизмов, восемь отдельных
global beams, исходные материальные IDs, live/archive cohorts, состояния и
work lineages. Восемь архивных исходных точек не заменяются HOT-birth материалом.
Подлинность исходного материала и наличие нынешней retained cold области
проверяются как разные свойства.

## Одна проекция и один потенциал

Каждый из восьми блоков использует зарегистрированные whole-bank COMMON
strain integrals. Пусть `a` — семь координат данного cold body, `H0` — его
опорный tip strain, а `r = E_rem = Green_increment(a) - a0*H0` — остаток
деформации. Скалярное
произведение здесь включает настоящий retained material mask и material law
целого банка. Из сохранённых интегралов берутся

```text
kappa = <H0, H0> > 0
p     = <H0, r>
N     = <r, r>
s(a)  = a0 + p/kappa
Ds    = e0 + Dp/kappa
D2s   = D2p/kappa
Rperp = N/2 - p^2/(2*kappa)
```

`s(a)` имеет размерность метров и задаёт соответствующий исходный radial
increment. Первые шесть компонентов `Pi` — общие owner angles; следующие
восемь — `s(a)` восьми банков. Owner increments выражены в радианах, body
increments — в метрах. Матрицы и тензоры сохраняют эти смешанные единицы.

### Строгое Cauchy enclosure проекции

Первый frozen gate обнаружил чрезмерно широкий enclosure `p/kappa`: он не
позволял доказать неизменность normal branch во всех исходных точках.
Обычная и factored формы давали одинаковый итог — четыре строго compressed
точки и 12 enclosures, пересекающих opening threshold. Пересечение с
квадратичными тождествами `P1/P2` этот итог не улучшило. Пересечение порога
в enclosure не доказывает физическое открытие контакта; strict normal-branch
guard обоснованно блокировал полный scalar response.

Исправление использует Cauchy–Schwarz для того же самого положительного
strain inner product, с теми же mask и material law:

```text
p^2 <= kappa*N
abs(p/kappa) <= sqrt(N_hi/kappa_lo).
```

Здесь `N` — исходный полный raw channel37
`raw.remainder_strain_inner_product_j = <E_rem,E_rem>`, где
`E_rem = Green_increment(a) - a0*H0`. Используется именно сохранённый
остаточный norm, а не восстановленный norm полного strain increment
и не energy `N/2`.
Upward cap вычисляется только целочисленной арифметикой. При dyadic scale
`S` строятся `M=ceil((N_hi/kappa_lo)*S^2)`, `m=ceil(sqrt(M))`, `C=m/S`;
проверяется точный сертификат `C^2 >= N_hi/kappa_lo`. Enclosure исходного
quotient пересекается с `[-C,C]`.

Это сокращение строгого enclosure неизменного `p/kappa`, а не constitutive
clipping. Функция `s`, её `Ds/D2s` и material law не изменены; производные cap
не подставляются в правило цепочки. Восемь блоков сохраняют два новых
derived fields: `projection_quotient_bounds_m` и
`projection_quotient_cauchy_absolute_upper_bound_m`. Все 73 исходных raw
integral intervals остаются неизменными; новая quadrature не требуется.

Точная read-only диагностика с этим сертификатом дала 16/16 strictly
compressed source points: максимальная верхняя граница normal traction
около `-2254.174 Pa`. Во всех восьми блоках `Ds_tip.lo > 0`. Эта диагностика
подтверждает branch enclosure, но не заменяет свежий полный gate или аудит.

Raw remainder gradient и Hessian являются производными `N/2`, а не `N`.
Поэтому

```text
gperp = g(N/2) - p*Dp/kappa
Hperp = H(N/2) - (Dp outer Dp + p*D2p)/kappa
```

Энергия новой системы определена одним выражением:

```text
U92(q) = U871(Pi(q)) + sum_b Rperp_b(q_b).
```

`U871` содержит исходные 16 contact stores, их serial columns и восемь
отдельных global beams. Каждый из этих механизмов оплачивается один раз.
Проекционная часть calibrated cold beam energy заменена исходным beam
potential; дополнительно оставлен только `Rperp`. Полная calibrated COMMON
cold energy не добавляется параллельно. Continuous `C dtheta` также не
энергизируется вторым представлением тех же исходных контактных механизмов.
Контракт API отклоняет явное повторное включение этих энергий.

Сохранённые reference energies предыдущего выпуска:

| Механизм | Literal reference energy, J |
|---|---:|
| 16 contact stores | 14 285 931 143,033392 |
| Их serial columns | 28 571 862 286,06678 |
| Восемь global beams | 251 841 262 157,0276 |
| Общий исходный источник | 294 699 055 586,12775 |

Таблица показывает десятичное представление сохранённых literal values;
точные Fraction receipts входят в DTO. Beam ledger остаётся константой
относительно точной квадратичной модели. Signed difference между literal
beam ledger и суммой `.5*k*w_ref^2` равен `1.988602611261443e-5 J`.
При `k=K_old/R^2` переменная часть равна
`sum(k*w_ref*s + k*s^2/2)`. Константа сохраняет ledger, не меняя силы или tangent.
Old solved radial q отсутствует; совпадение старых binary beam forces не
объявляется. Ранее сохранённые signed arithmetic residues не стираются.

## Полные первые и вторые вариации

Для `J = DPi` вся система использует одно правило цепочки:

```text
g92 = J^T*gsource + gperp
H92 = J^T*Hsource*J + sum_r gsource_r*D2Pi_r + Hperp.
```

Член `gsource*D2Pi` обязателен: исходная преднагрузка вносит кривизну даже
при неизменном source tangent. Его исключение меняет вторую вариацию и не
даёт исходный 14-coordinate operator. Сохраняются полные source scalar B/H,
вращение receiver director, owner couples и соответствующая virtual work.

Owner covariance объявлена для свежего общего rigid rotation, приложенного
после frozen reference geometry и body deformation, с совместно повёрнутыми
reference generators. При такой смене мирового кадра `kappa` и `p` инвариантны.
Конечная история owner increments в этом выпуске не допускается.

Relaxed zero-strength shear и frozen elastic predictor — две разные ветви
запроса. В relaxed ветви shear traction/tangent равны нулю; пластическое
изменение остаётся предложенным. Predictor — диагностическая упругая ветвь.
Новый return-map state, прочность, damage или принятое разрушение не выводятся
из проверки этого компонента.

## Один обратный 2-jet: начало и настоящий ненулевой запрос

Используется один общий обратный 2-jet восьми банков, а не per-node
pseudoinverse. Его два локальных anchors различаются явно; конечная обратная
траектория этим сертификатом не объявляется.

В structural Green zero reference `Ds_tip=1`. Первая производная обратной
карты равна единице, а tip curvature равна `-D2s_tip,tip`. Это структурное
следствие Green polynomial; новая нулевая bulk quadrature не изобретена.

В настоящем сохранённом ненулевом запросе требуется положительный enclosure
`Ds_tip`. Его inverse derivatives равны

```text
Dpsi_tip  = 1/Ds_tip
D2psi_tip = -D2s_tip,tip/(Ds_tip)^3.
```

Проверяются композиция `DPi*DPsi = I14` и её полная вторая вариация. Исходный
source Hessian восстанавливается с curvature term обратной карты:

```text
Hsource = DPsi^T*Hprojected_source*DPsi
          + sum_i gprojected_source_i*D2psi_i.
```

Один линейный selector `T` без `D2psi` этого равенства не обеспечивает.
Сертификат относится только к локальным первой и второй вариациям source
part в этих двух anchors; конечная path work из inverse2jet не выводится.
`Rperp` остаётся отдельной физической bulk energy и работой. Полная конечная работа новой системы не объявляется
равной работе старого источника. Structural origin certificate также не
объявляется восстановленным трёхмерным архивным prestress.

## Подлинные NONZERO contexts и precision replay

Native control использует ровно восемь настоящих сохранённых COMMON cold
contexts выпуска841. Их семь координат ненулевые; они не заменяются нулевым
или свежим произвольным probe. Полный query vector92 привязан к этим typed
integral receipts. Evaluator принимает только этот точный вектор, с нулевыми
конечными owner/HOT increments; owner gradients и Hessians при нём полные.
Другой конечный body vector потребовал бы нового зарегистрированного raw
integral receipt.

Закрытая reference841 имеет declared geometry precision160, а reference871
— precision96. Сравниваются полные физические определения reference и всех
13 банков. Исключены только declared precision и перечисленные binding IDs:
reference `geometry_enclosure_precision`, `reference_snapshot_sha256`;
bank `bank_snapshot_sha256`, `reference_snapshot_sha256`. Эквивалентность не
ограничена совпадением `R/A`.

Исходные полные typed contexts160 восстанавливаются как неизменяемое
доказательство. Отдельно настоящая factory регистрирует fresh96 contexts из
тех же 73 числовых integral intervals и тех же cell/tail/correction receipts.
Для каждого полного body/material/raw/diagnostic DTO разрешены только
35 именованных binding-identity paths. Их список сохранён в native receipt;
после этих замен все остальные поля совпадают полностью. Исходные DTO160 не
переписаны и не перезапечатаны. Новая tensor quadrature не запускается.

Полные hashes mutable dependencies и точные рекурсивные Python types
проверяются до возврата cached result. JSON-одинаковая поддельная dataclass
или `list` вместо исходного `tuple` не получает доверие через cache hit.

Производные `p` описывают квадратичную структуру Green projection. Единственный
receipt `N`, `DN`, `D2N` в ненулевом probe не восстанавливает произвольную
quartic bulk energy во всём пространстве координат. Таких запросов и такого
сертификата этот выпуск не предоставляет.

## Геометрия, состояния и время

Выбрана прежняя `archival_old_reference` force basis с geometry0. Source
contact clock2e6/step2, memory clock1e6 и material geometry clock1e6 сохраняют
свои разные значения. Current objective force basis не принята взамен
архивной. Все исходные stores, before/unaccepted work ledgers, материальные
резервуары и thermal associations сохранены; новых списаний или state advances
нет.

Компонент не выполняет physical face/memory handover, finite HOT mean law,
load/drag assembly или новое mixed equilibrium. Whole-clock all-pair admission
не разрешён, новый интервал не принят: **один прежний / ноль новых интервалов**.

## Проверки и закрытая воспроизводимость

Frozen попытка01 завершилась **FAIL: 10 failed / 4 passed** за449,79 с pytest;
wrapper —450,393 с, gate —571,638608827 с. Десять failed tests имели общий
блокирующий strict normal-branch response; это не десять независимых ошибок
производных. Точные четыре исходных Python-файла, source map, console,
after-baseline и preservation receipt сохранены в
`failed_release01_preserved`. Исходные выходы `release01` оставлены неизменными.

Frozen попытка02 после Cauchy correction завершилась **FAIL: 1 failed /
13 passed** за544,43 с pytest; wrapper —545,266456937 с, gate —645,941903837 с.
Все остальные source/full92/state/replay checks прошли. Единственная ошибка
была в тестовом требовании строгого знака восьми tip-curvature enclosures:
сохранённые raw bounds этого знака не доказывают. Изменение физического ядра
для исправления этой проверки не требовалось.

Исправлен только test6: теперь во всех 784 entries восьми `7x7` curvature
blocks двух shear branches сборка обязана содержать целиком независимо
полученный enclosure `source_force*(raw_Hp/raw_kappa)`. Невырожденный
uncertainty interval не может быть заменён `I(0)`, даже когда он содержит ноль.
Source forces отдельно сверяются с исходным potential. Строгое доказательство
prestress curvature `11/2` в независимом polynomial oracle не изменено.
Core, native control и wrapper сохранили прежние hashes.

Точные failed02 sources, map, audit producer, console и baseline сохранены:
10 payloads, отдельный preservation receipt и два gate files. Итоговый manifest
**73 файла** включает обе неудачные попытки. Выходы failed02
не переписаны после исправления теста.

После этого frozen попытка03 и единственный независимый аудит завершились
PASS. Между её freeze, gate и аудитом четыре исходных Python-файла не менялись.

Авторитетная исправленная frozen попытка: **03, PASS**. Wrapper measurement
pytest —556,812071810 с; отдельно printed pytest —555,94 с. Native callback
wrapper —23,893818156 с, собственное native measurement —23,832926630 с.
Свежий pytest: **PASS, 14/14 за 555,94 с**.
Native: **PASS, 2/2 whole-system queries за
23,832926630 с**; обе shear branches используют тот же настоящий
NONZERO vector92. Gate: **PASS за 767,635476494 с**.
Независимый stdlib-аудит: **PASS, с одного запуска за 160,207512894 с**;
Он независимо проверил восемь настоящих NONZERO projections / 584 raw
channels, полный precision bridge всех13 банков с35 named ID paths,32 source
atom responses,64 scalar B и64 scalar H,16 beam responses,8 Rperp blocks,
два full92 energy/gradient/Hessian с prestress curvature, один reference-origin
и два actual-nonzero inverse2jets, по два source-only иtotal-bulk inverse
pullbacks, HOT30 zeros и16 work lineages. Project imports, старые tests,
force solvers иквадратуры в независимом запуске отсутствовали.

14 новых тестов проверяют raw-derived projections и Rperp, один source/beam/
bulk potential, все source scalar B/H, full92 gradients/Hessians с prestress
curvature, mixed directional variations, source-only work и дополнительную
bulk work, оба inverse anchors, HOT30 zeros, material/clocks/work histories,
полный typed replay и semantic tamper/query-transplant rejection. Fraction
polynomial oracle дифференцирует независимый потенциал; Decimal differences
используют выведенные truncation/rounding bounds.

Pure helper AST guard: **PASS за5,382548 с** на test source `f16e8965...`.
Он проверил 32 integer
upward-square-root cases, независимый polynomial potential, full92 chain,
шесть Decimal finite differences и inverse curvature. Предыдущий pre-Cauchy
guard сохранён как исторический prefinal receipt: он ещё не проверял этот
cap и не объявляется проверкой окончательного frozen source.

`cauchy_pure_math_guard_final_test_reconciliation.json` фиксирует ровно одно
изменение test AST — исправленный test6. Все выбранные math helpers и
polynomial oracle сохранены. Результат5,382548 с не объявляется повторным
выполнением guard на новом test source `408e127e...`.

Pure guard — отдельная дешёвая проверка чистой арифметики; она не заменяет
полный pytest, подлинный typed replay или независимый аудит. Dual-context typed preflight:
**PASS за 334,143480123 с**. Его восстановление
factory и полного старого diagnostic algebra также не объявляется новым solver
или quadrature. Исторические тесты/решения не повторяются.

Frozen source875: **PASS**, полный source map неизменен до/после gate и
аудита. After-gate baseline PASS за51,071513509 с; before-audit baseline
PASS за102,245726576 с. Это stdlib readback прежнего выпуска871 и его
закрытой ancestry, а не повторное вычисление новой математики.
Прежние источники871, их закрытый выпуск и объявленные зависимости:
**PASS: 871 источники, прежние23 manifest files / 51 dependencies и9 строгих ancestry closures сохранены**. Final closure: **PASS_FINAL875_EVIDENCE_READBACK_ONLY за28,043987216 с**: проверены73
manifest files,180 recorded raw dependency pins, прежние51 dependencies,
9 строгих ancestry closures иполный875-source map до/после. Старые1633
artifacts повторно после математики не перечитывались: используется закрытый
before-math baseline receipt. Новые физические вычисления не выполнялись.
Preserved failed01, failed02 и preflight corrections:
**оба frozen FAIL01/02, bootstrap mismatch, development diagnostic failures, initial160/96 mismatch иprefinal/final pure-guard receipts сохранены отдельно с точными sources, consoles иself-seals; исправления раскрыты выше**.

| Объект | Raw SHA256 / canonical digest либо self-seal |
|---|---|
| Final frozen source875 map — raw / canonical | `8f7608219c398a05913d96e80baf685b997dccff466ee9cf9eb1aa89b8124675` / `13c19fe10056af9ab621e2c8a23e63ad36b301c95e121a6835cead019378e385` |
| Native control report — raw / self-seal | `cf4982541a7eabf263a430ebee0c20e68979154af135f6bec3050b1d24184668` / `bc6620ef7b9afd6f1cee4b8f571a63c8738548e305b2a01342415ddb577651a4` |
| Native packet | 2ddf12415b9abecbbeead78a203ba82d7e18fea84bc9197aa789a607514939b2 / f4c5f9519698c28bf12b9854dc05f78704f25068fca427478a0cfc98786f1a8d |
| Gate wrapper result | a7e74fc5d8774eaaabdcf3919357d27dbd6c35f0c79ea741cb0a89057e51e10e / a20d3dcfa9e158e5e57829f4b81e7bf3206e604e3070eed0514b097666507cbb |
| Independent audit source/result | 74090c447f56ff25f8bf3fc94b28dce3b468c3a19954ef16c4bf85550566e22e / f7280d7b1edee1bdf1467daa9d97271d378b3b8a43b197cf606729eeef499532 / 8b325f6dca075438445085d77ba6255cfb0bdf762c653c39e71d308ec6d13be1 |
| Audit console raw | `be1259238365ef484e4ee544b794bc0cfa593e3c452b3be6fd864136c64519fb` |
| Final readback source raw | `85de6cb6261971ed36e78756c038455a3ec114bccac65f2a6ea3704bcc30355e` |
| Final readback JSON — raw / self-seal | `190d3720c566c2ea494eaa601bbac0a7ae93e842df14938ab4dd4d62b42e6c71` / `83e50c3fa39550ca657224f0cf659669802dd1085f7748a07954c52b6e53c44b` |
| Before-audit baseline — raw / self-seal | `b048664d22012a46ca95715e60586424568995c37f4637b18e7b51aa40cd685d` / `fd6e0048bd24748a5c07c30d6a58371641b2a973955b68ab38e2805e36f35cc7` |
| After-gate baseline — raw / self-seal | `b3f664f6e91c2278659a74ec0bfcae0113ffb257021ba2f9b1ec6136b5d147ed` / `4525742b21561731dc20cc8d18d0b882219da2a2959fbeda8d5dd04468be6f4e` |
| Closed manifest | 035c7eeae472676924792d07e1d05b8e3660a4a9d0fa9d8cfe8f55df71862412 / 519d42468d7291b2af3db6894943a2ec91e2c12f490643e2d18f4b16d63777a2 |
| Native precision-equivalence receipt | ae5465c0ca0faf5ff00cd0d31c5aa5d2e17588399a6e22a720835a07507c884a |
| Core raw | `4b7659d50d73e3c60b7b8c2888277813489034e866dfb4161515a79e37514488` |
| Tests raw | `408e127ee1f5931eb4803d613bf67a9dd56e4bf7682cd18b530b10fd9eeb5f8d` |
| Native control raw | `54a8de7ec6aea6abda6fafcb4a42c3b1d796343889623ef9fd452fc01e191068` |
| Wrapper raw | `7b52887679c420bb2d24a3406ccaa761a9fd31b7763a1eb31832e3227e7600b2` |

Evidence directory: **analysis/geometric_native_common_strain_projection_validation/**. Frozen validation files:
`tectonics/geometric_native_common_strain_projection.py`,
`tests/test_geometric_native_common_strain_projection.py`,
`analysis/geometric_native_common_strain_projection_validation/native_control.py`
и `analysis/geometric_native_common_strain_projection_validation/validate.py`.
Закрытый каталог не перезаписывать; новый readback сохранять отдельно.

## Продолжение через сохранённый reader

Точный исполнявшийся stdlib reader, его JSON иconsole сохранены raw copies
в `analysis/geometric_native_common_strain_projection_final_readback/`:
`readback_source.txt`, `readback.json`, `readback_console.log`.
`readback_archive.json` с self-seal
`3f18a1faac70dd8e34fa2558c076e12cbacd1277db28bc7cbea15940e4a42676`
связывает три исходных `/tmp` paths, размеры иraw hashes; копирование не
запускало producer повторно. Вreceipt сохранён SHA256 первоначального
producer; его исходный `/tmp` path записан вarchive. Эти четыре archive
artifacts находятся отдельно
от запечатанного73-file component manifest.

В первоначальном audit dependency list сохранена одна временная ассоциация:
`/tmp/native_common_strain_projection_final_independent_audit_20261006_baseline871_readback.json`.
В новой cloud session её можно восстановить точной raw copy постоянного
`baseline_before_independent_audit.json`. Сначала проверяется pinned SHA256;
существующий временный файл должен совпадать byte-for-byte. Отличающийся файл
не перезаписывается. Source pins иself-seals прежних receipts сохраняются:
новый baseline или физическое вычисление не запускаются.

Команды для нового readback из корня репозитория:

```bash
python - <<'PY'
from hashlib import sha256
from pathlib import Path
saved = Path('analysis/geometric_native_common_strain_projection_validation/baseline_before_independent_audit.json')
target = Path('/tmp/native_common_strain_projection_final_independent_audit_20261006_baseline871_readback.json')
data = saved.read_bytes()
assert sha256(data).hexdigest() == 'b048664d22012a46ca95715e60586424568995c37f4637b18e7b51aa40cd685d'
if target.exists():
    assert target.read_bytes() == data, 'Existing temporary baseline differs; preserved without overwrite'
else:
    with target.open('xb') as stream:
        stream.write(data)
PY
python analysis/geometric_native_common_strain_projection_final_readback/readback_source.txt \
  --out /tmp/native_common_strain_projection_continuation_readback.json \
  --document GEOMETRIC_NATIVE_COMMON_STRAIN_PROJECTION_IMPLEMENTATION.md \
  --document CLOUD_HANDOFF.md \
  > /tmp/native_common_strain_projection_continuation_readback.log
```

Reader проверяет сохранённые bytes, exact schemas, self-seals, source maps и
при передаче `--document` отдельно связывает текущие docs. Project imports,
математика, тесты, solver иquadratures не запускаются. Hash association docs
не добавляет их в component manifest. После перезапуска выполнять этот
readback; прежние физические решения повторять не требуется.

## Остаток пункта1

Пункт1 открыт: **два прежних блока завершены; один блок остаётся с теми же
тремя условиями**. Внутри первого условия теперь выполнена ненулевая bankwise
связь исходных восьми каналов с cold fields и их once-only энергетическая
сборка. Полный physical handover не завершён.

1. Физическая связь с полями текущих тел/архивов и настоящими COMMON faces:
   после выполненной coarse cold coupling остаются actual face measure/memory,
   локальный архивный material handover, согласование сил/работы, column/bulk
   energy и prestress, HOT law/finite mean chart, loads/drag и новое mixed equilibrium.
2. Whole-clock all-pair admission, включая reverse C5→C1.
3. Принятое продолжение с независимой path work, балансами и joint restart.

Следующий конкретный шаг — локальная связь с настоящими faces и архивным
материалом с корректной мерой/памятью, HOT law/finite mean и once-only energy;
далее loads/drag и новое mixed equilibrium. Новые этапы к прежнему плану не
добавлены. Принят один прежний / ноль новых интервалов. Пункты2–6 остаются
открытыми; объём остатка указывать в каждом отчёте.
