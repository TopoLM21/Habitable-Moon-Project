# Исходный материал и единый сокращённый механический источник

Дата: 06.10.2026 (МСК). Ветка: `fix/genesis-mantle-convection`.

## Результат и граница выпуска

Статус выпуска: **PASS_COMPONENT_EVIDENCE_CLOSED**. Scope:
`COMMON_ORIGINAL_MATERIAL_EXTENDED_REDUCED_SOURCE_COMPONENT_ONLY`.

Все 16 старых контактно-колонных механизмов привязаны к своим исходным
материальным меткам. Сохраняются восемь живых donor points, восемь архивных
donor points и все 16 живых receiver points. Архивные материалы сохраняют
первоначальные IDs, владельца, когорты, свойства и резервуары. Пять новых
HOT-birth материалов не используются как замена старого архивированного материала.

Объявлена новая пассивная карта архивов через тот же COMMON flow и исходную
траекторию владельца. Это самостоятельное расширение опорной карты исходного
материала. Оно не восстанавливает движение трёхмерного архивного тела или
прежние дальние концы колонн. Подлинная принадлежность исходному материалу
проверяется отдельно от принадлежности нынешним 13 COMMON bodies.

Новая механическая система имеет 100 координат: шесть общих owner angles,
86 координат нынешних тел и восемь независимых внутренних source channels.
Эти внутренние каналы ещё не сведены к полям 13 тел. Finite queries требуют
нулевых owner/body increments; полные первые и вторые owner derivatives
вычислены. Нулевая source coupling остальных 86 координат — явная граница модели.

## Материал, геометрия и механика

Для каждой из 32 привязок проверяются исходный parent support, материал,
owner, live/archive domain, allocation и при необходимости cohort. Состояние
donor определяется знаком собственного исходного полинома `g(q,u)/u` на
всём времени и `g(q,1)`, затем сверяется с запечатанным независимым аудитом.
Наличие когорты parent5 не переводит его четыре старые точки в архив:
все donor4/5 остаются живыми, donor6/7 архивированы.

В геометрической карте используется луч `q/||q||`, текущая полная owner pose
и `Phi`. Исходные буквальные `q`, normal/down directors и механические
операторы не нормализуются. Верхняя точка — явный выбор `z=0`; карта также
определена по всей исходной areal thickness. Глубина не подменяет длину старой
1000m колонны или восстановленный far endpoint.

Материальная точка `X` и virtual source point `y` связаны перенесённым
rigid lever. Материальные local directions/directors получены полным
pullback через настоящую ненулевую текущую owner pose. В reference exact
correlation `Q Q^T=I` даёт `y=p_archive`, inherited director и `eta=0`.
Finite internal motion задаёт `X+delta*v_source`; запрос material point
возвращает frozen reference chart, а не эту деформированную позицию.

Выбрана только архивная geometry0 force basis. Source contact clock2e6/step2,
memory clock1e6 и material geometry clock1e6 остаются разными. Переход к
current objective force basis требует отдельного доказательства переноса сил
и учёта энергии/работы.

При приложении сил к действительным `X` учитываются обе lever couples и
вращение receiver director. Их сумма воспроизводит owner moments virtual
source points. Все силы и полные B/H следуют из одного потенциала; inactive
элементы сохраняют преднагрузку и меняющуюся энергию. Relaxed shear имеет
нулевой tangent и proposed plastic increment; frozen elastic predictor —
отдельная диагностическая ветвь. Новый return-map state не принимается.

## Один оператор и энергия

Один глобальный `T` вкладывает исходные 14 координат в новые 100: shared owners
занимают позиции0–5, internal channels92–99. Это объявленное расширение,
а не доказанная редукция в прежние 92 координаты или per-node pseudoinverse.

Для выбранной буквальной source mechanical basis проверяются
`B_new*T=B_source`, virtual work и полная вторая вариация:
`H_source=T^T*H_new*T + B_new*D2Psi`, где `D2Psi=0`.
Полные global derivatives хранятся в sparse14 форме и раскрываются API до100.
Старые binary64 operators, собранные через sin/cos и horizontal row,
не объявляются тождественными точным literal-director derivatives.

В потенциал входят один раз:

| Механизм | Сохранённая reference energy, J |
|---|---:|
| 16 contact stores | 14 285 931 143,033392 |
| Их локальные serial columns | 28 571 862 286,06678 |
| Восемь отдельных global beams, literal ledger | 251 841 262 157,0276 |
| Общая reference energy | 294 699 055 586,12775 |

Beam tangent `k=K_old/R²` и reference force `k*w_ref` вычисляются из исходной
квадратичной модели. Literal beam ledger сохраняется отдельной константой
относительно точной суммы `.5*k*w_ref²`; signed difference равен
`1.988602611261443e-5 J`. Энергия при изменении координат равна
`E_beam(delta)=E_literal+sum(k*w_ref*delta + k*delta²/2)`;
поправка reference energy не заменяет beam forces или tangent.
Binary reduced evaluations сохраняются отдельно:
старое solved radial q отсутствует, поэтому старый binary beam force не
объявляется воспроизведённым. Contact force binary residues также сохранены.

Контракт сборки запрещает повторную энергию этих beams в calibrated COMMON
cold response; API отклоняет её явное включение. Общая сборка с cold bulk
ещё не выполнена. Continuous `C dtheta` не энергизируется
дополнительным источником. Родительские материальные резервуары не списываются
на nodal springs. Состояния, anchors, clock, thermal association и work ledgers
не изменяются и не принимают новый интервал.

## Проверки и воспроизводимость

Авторитетная попытка01 завершена: **14/14 свежих тестов PASS за398,92 с**;
native **6/6 global и64/64 material point queries PASS за49,410405 с**,
callback49,428598 с. Full typed replay в том же процессе PASS.
Единый gate **PASS за593,974112 с**, источник871 неизменен до/после.
Независимый stdlib-аудит **PASS за816,774 с**, с первого запуска:
32 исходные метки/верхние привязки,64 полные point/gradient queries,
6 global gradients/Hessians и T14→100,96 механизмов/моментов/virtual work,
192 первых и192 вторых scalar variations,8 beams/48 beam responses,
6 exact quadratic internal path works. Project imports и прежние физические
solver/квадратуры не использовались.

After-gate stdlib baseline PASS за29,818632 с; отдельный baseline до независимой
математики PASS за41,959 с. После неё проверены все871 источники,
закрытые выходы, прежние24 файла выпуска867 и hashes объявленных зависимостей.
Три сохранённых preflight-файла относятся к проверке чистых test helpers до
последнего tamper/restore теста: они не объявляются полным pytest.
После freeze исходный код не менялся; неудачных попыток нового gate нет.

Native control проверяет шесть whole-system запросов: обе shear branches
при нулевых и двух совместных finite8 increments. Ещё 64 запроса проверяют
32 материальные метки при `z=0` и половине areal thickness. Fresh tests используют
независимые Fraction/Decimal формулы scalar jets, SO3, Phi и quadratic path work.

Полный DTO replay отклоняет semantic reseal, материал/HOT substitution,
перестановку genuine response к другому query и изменение runtime dependencies.
Подписи отдельных compact mechanical/geometry contexts проверяются перед кешем;
проверки in-place mutation восстанавливают источники в `finally`.

Старые тесты и физические solve/сертифицированные квадратуры не повторяются.
Для аутентификации исходника заново восстановлены typed867 source DTOs и
буквальные carrier operators; это отдельно от stdlib baseline readback,
который читает только прежние hashes, receipts, seals и output closures.

Baseline867 до добавления файлов: PASS за69,868208 с. Reusable явно раскрывает
frozen867 projection и независимо проверяет полную карту871 до/после,
прежний manifest24, strict4 и сохранённую цепочку1633. Независимый audit
использует только stdlib, отдельно проверяет новую математику и после неё
повторяет hashes источников, выходов и прежних зависимостей.

Манифест закрывает23 файла; он и independent audit лежат вне четырёх
контрольных выходов. Исходники bootstrap/reusable, readbacks и три preflight
файла сохранены в том же родительском каталоге. Документ и handoff являются
указателями и не входят в закрытый evidence manifest.

| Артефакт | Raw SHA256 | Canonical/selfseal SHA256 |
|---|---|---|
| Frozen871 | `340716e8267d63974e3d8c27a94ac53af60d90e6f9f2382b86874a00c0c3fe9b` | `5ba59a6e8231e7cf2408b2614b19601cc631bd204eb4f436b4241cf4fa535915` |
| Wrapper | `44d12b3fb10edfef90a83c50385278b45135d1b051a5733d2b2f7d98f2755b56` | `8c3accb44756caea0cecd5a5573d57c007bccc1ce33f77d82fd38bcaf72ce75e` |
| Native | `0fb94534790bc3c9da9a89bea75e9c9f68fba6fda492f22b59c8427837b098d8` | `4ca0373b5bf9a344e545291b8a1128e2851323d6d4b20b874c2118d3b5210d51` |
| Source packet | `a9fcc7874c0e68940e20ded658748300f55d9f1d296db5c9d67b59269ac8b3ed` | `cd1e2457a857b47891cd40a89219abfac3aa2228028aa9636f0e77f7a10082dd` |
| Independent audit | `caa8033c5a5c84f18bf29f9d1f91ddf7e46b0efd4dcef529a949dbcb14d13466` | `c7c66901dfde16fc2e693dfff9d260e94b28b9d9bb4ad45a74fd13071c3b4bcc` |
| Exact stdlib producer | `06d07d554a7265f96e132327e0526e2a7baf83d18b29d0d620413e3acf7eaaf7` | raw only |
| Manifest23 | `c2a57aa83f8b93ff03bf45779aae830897244e40060cc8604b54510cf253685a` | `eadbc0fe6f437975128973ae52290dcf85d7ebce7ab5faa7cd503b964884edfc` |

| Новый Python-файл | Raw SHA256 |
|---|---|
| `analysis/geometric_native_common_material_transfer_validation/native_control.py` | `890bfeca731387981b94ef9b86f6825e0c5caa51341bcb522d0adcdb16895b30` |
| `analysis/geometric_native_common_material_transfer_validation/validate.py` | `0fc5cba450c2af869b395f17e0eb240984cafca6b33a2ecad9a1ba4d0e3e115a` |
| `tectonics/geometric_native_common_material_transfer.py` | `9c534d3b664570fd7d61f4f26601be3af7882de373264b64ba8e071856af9929` |
| `tests/test_geometric_native_common_material_transfer.py` | `c9f99dddf08ae7a204c366f4f90217035e0f270de6ea9bceb34afc4e322b18c9` |

Прежняя frozen867: canonical
`bfabfa58dbe49ce1102ef9ab72680a301f8cea6fb30093442150d8d4bda45ef8`,
raw `4b2b98b406cf1f955260dcc24770b9a860cf330e49adc72b4d47e018a140ffc6`.
Прежний manifest24 raw
`e0ebc4fbd1086285035fc868181f3ba7cb34833df897d8c576537b71716a91bb`,
selfseal `503525b0c461b4e905844aff4d0178536b763446b56bc6c2b4a4a44d3f7f9160`.

Команда дешёвого stdlib readback без повторной физической математики:

```bash
.venv/bin/python analysis/geometric_native_common_material_transfer_validation/baseline_867_reusable_source.txt \
  --expected-added-path tectonics/geometric_native_common_material_transfer.py \
  --expected-added-path tests/test_geometric_native_common_material_transfer.py \
  --expected-added-path analysis/geometric_native_common_material_transfer_validation/native_control.py \
  --expected-added-path analysis/geometric_native_common_material_transfer_validation/validate.py \
  --out /tmp/native_common_material_transfer_baseline871_readback.json
```

Команда завершённого gate:

```bash
.venv/bin/python analysis/geometric_native_common_material_transfer_validation/validate.py \
  --out analysis/geometric_native_common_material_transfer_validation/release_validation_01 \
  --control analysis.geometric_native_common_material_transfer_validation.native_control:run \
  tests/test_geometric_native_common_material_transfer.py
```

Завершённый каталог не перезаписывать. Для readback использовать сохранённый
stdlib producer и новый путь `/tmp`; прежние физические проверки не повторять.

## Остаток пункта1

Пункт1 открыт: **два прежних блока завершены; один блок остаётся с теми же
тремя условиями**. Новый выпуск закрывает внутри первого условия исходную
материальную идентичность/пассивную карту и единый сокращённый механический
потенциал с отдельным изгибом. Полный физический handover остаётся незавершённым.

1. Физическая связь внутренних каналов с полями текущих тел/архивов и настоящими
   COMMON faces; согласование меры/памяти, сил/работы, column/bulk energy и
   prestress, HOT law/mean chart, loads/drag и новое mixed equilibrium.
2. Whole-clock all-pair admission, включая reverse C5→C1.
3. Принятое продолжение с независимой path work, балансами и joint restart.

Следующий конкретный шаг — рабочая связь этой подсистемы с body/HOT/mean
координатами, настоящими faces и once-only bulk energy; затем loads/drag и новое смешанное
равновесие. Принят один прежний / ноль новых интервалов. План из шести пунктов
не расширен; объём остатка указывать в каждом отчёте.
