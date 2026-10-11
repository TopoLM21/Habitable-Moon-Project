# Единое распределение всей исходной опорной меры и памяти

Дата: 06.10.2026 UTC. Ветка: `fix/genesis-mantle-convection`.

Продолжение первого условия последнего блока исходного пункта 1 после
[опорной меры и последовательной цепочки 859](GEOMETRIC_NATIVE_COMMON_CONTACT_TRANSFER_IMPLEMENTATION.md).
План из шести пунктов не расширен.

## Статус

Реализован один аутентифицированный реестр всех четырёх исходных контактных семейств:
четыре непрерывных семейства, 12 общих границ, восемь частей распределения,
все 16 буквальных историй и четыре текущих поля памяти. Десять genuine
проб плотности относятся к пяти настоящим холодным контактным участкам.

Bootstrap859 readback **PASS за 50,679 с**: прежние 859 Python-файлов,
цепочка 1633 артефактов, закрытые выпуски и текущие raw/selfseal сохранены.
Авторитетная свежая попытка02: **17/17 тестов PASS за 318,55 с**, единый
gate **PASS за 587,293 с**, native control **PASS за 55,441 с**,
callback **55,462 с**. В том же процессе полностью восстановлены
четыре семейства, восемь частей, 16 source histories, четыре CURRENT
memory queries и десять genuine density queries. Выходы:
`analysis/geometric_native_common_source_allocation_validation/release_validation_02/`.
Независимый stdlib-аудит **PASS за 72,091 с**, с первого запуска.
Проверены полный source map 863 до/после, все прежние 859 источников,
цепочка 1633 артефактов и закрытые выпуски, четыре точных projective
границы, все 16 историй/48 node epochs/288 literal work-store scalars,
десять независимых source-direction/reference-density/coarea/RN ответов,
четыре CURRENT memory queries и восемь harmonic fields.
Компонентный учёт источника завершён.

Первая попытка01: 17 тестов и native control прошли, общий gate остался
**FAIL** из-за двух ошибок нового source-replay wrapper: checksum key
удалён помощником `sealed`, а density DTO хранит `receiving_cell_id`,
не `cell_id`. Исправлен только validator; он также требует ровно 17 PASS.
Основной модуль, тесты и native control не менялись. Первая попытка
сохранена в `release_validation_01/`; её frozen map, точный validator TXT,
внешний журнал и `attempt01_preservation.json` сохранены в родительском
каталоге. Она не переобъявлена успешным выпуском.

Scope: `COMMON_WHOLE_SOURCE_ACCOUNTING_COMPONENT_ONLY`.
Это единое распределение и наблюдение исходных reference budgets;
полный физический перенос материала, старых сил и преднапряжений,
HOT-закон, совместное равновесие и новый интервал не приняты.

Frozen863 canonical:
`83ebcc92428688af96c4f104b5ce9bbd96f29625e8a46d2b08e77b2fbad99406`;
raw `frozen_source_map_863.json`:
`e9e1dc52ae057b29f06e1e345e8ac055fd67aa8314f147d2dd5105d75a9bd2bf`.
Прежние 859 Python-файлов неизменны; ровно четыре добавленных пути:

- `tectonics/geometric_native_common_source_allocation.py`;
- `tests/test_geometric_native_common_source_allocation.py`;
- `analysis/geometric_native_common_source_allocation_validation/native_control.py`;
- `analysis/geometric_native_common_source_allocation_validation/validate.py`.

## Полный раздел исходных семейств

Каждое семейство содержит буквальные `C=R*source_height/source_frame_sine`
и `Gamma`, полный current `step.memory` trace, endpoint anchors, исходный
контактный закон/колонны и четыре original atom identities. Cells ссылаются
на единственного владельца этого семейства; они не получают полный `CGamma`
повторно. Oct0 сохраняется целиком, хотя у него нет нового холодного контакта.

| Исходный receiver | Угловая часть | Назначение source support |
|---|---|---|
| Oct0 | `[0,Gamma0]` | Вне активных контактов |
| Oct1/C7 | `[0,gamma7]` | Вне активных контактов |
| Oct1/C7 | `[gamma7,Gamma7]` | Настоящий terminal-trim C6→C7 |
| Oct2/C8 | `[0,sigma8]` | Микроскопический C10→C8 |
| Oct2/C8 | `[sigma8,alpha8]` | Длинный C9→C8 |
| Oct2/C8 | `[alpha8,Gamma8]` | Положительный остаток вне контактов |
| Oct3/C11 | `[0,sigma11]` | Микроскопический C6→C11 |
| Oct3/C11 | `[sigma11,Gamma11]` | Длинный C10→C11 |

Границы sigma8/sigma11 — общие exact projective rays соседних настоящих
native arcs. Их идентичность доказывается канонизацией planes/rays и
сокращением только соседних точных обратных вращений. Равенство широких
интервальных чисел не заменяет эту идентичность. gamma7 задаётся истинной
terminal-envelope expression при `u=1`, а не старым untrimmed native end.

Для каждой части хранится формальная площадь `C*(upper_endpoint-lower_endpoint)`.
При суммировании именованные внутренние границы сокращаются точно:
остаётся `C*(Gamma-0)`. Численные площади частей — наружные интервальные
границы; их сумма не выдаётся за точное равенство при потерянной корреляции.

Все восемь частей имеют строго положительную непрерывную площадь. Остаток
C8 порядка `6.021e-20 rad` сохранён. Нулевое число старых Gauss-узлов на
микроскопических участках и этом остатке не позволяет объявить их пустыми.
Замкнутые швы имеют нулевую непрерывную меру; исходный атом на неразрешённой
границе сохраняется один раз в отдельной unresolved receipt, без midpoint.

## Один профиль на семейство и настоящая геометрия

`bind_native_common_source_allocation` требует явный словарь всех четырёх
trace IDs и точных нормированных неотрицательных Bernstein profiles.
Все части и пробы одного семейства используют одинаковый профиль.
Контроль объявляет uniform для Oct0/Oct2 и linear `(0,2)` для Oct1/Oct3.
Профиль по глубине — явно новая условная модель, сохраняющая исходную
одномерную меру `C dtheta`; исходное распределение материала по 3D-глубине
не восстановлено.

Разные настоящие пересечения глубин сохраняются. На каждом cut:

`tau=(R^3-r^3)/(3R^2)`,
`zeta=(tau-tau_low)/depth_span`,
`mu=C*abs(theta_comp_prime)*f(zeta)/depth_span`.

Native `theta_prime=-1` следует из exact original receiver plane и
положительной ориентации настоящего параметра. В ответе также сохраняется
общая projective differential `(x*y_prime-y*x_prime)/(x^2+y^2)`,
которая должна заключать этот точный результат. Масштаб computational
angle — буквальное `Gamma/(analytic_pi/2)`.

Физическая coarea `J` и отношение `mu/J` остаются отдельными величинами.
Дополнительная доля cell area не применяется: аналитическое интегрирование
mu по всей назначенной глубине и угловому support даёт `C*DeltaTheta`.
Это тождество нормированной модели, не выполненная численная квадратура.
J также не применяется
вторым множителем. Проба не получает конечную площадь одного узла.

`validate_native_common_source_allocation_measure` связывает существующую
декларацию 859 с настоящим cut, семейным профилем, памятью и часами.
Подмена профиля даже в другом cut того же семейства отвергается.
Это совместимая связь с локальной последовательной цепочкой, а не новая
сборка её 20-coordinate force/Hessian или общего равновесия из 92 координат.

## Все 16 историй и текущая память

Отдельный literal atom ledger сохраняет все source point IDs, computational
angles, исходные площади, prepared/previous/unaccepted-current states,
истории и шесть полей энергии/накопленной работы. Суммы и разности
current−previous хранятся как точные Fractions буквальных binary64 значений.
Все узлы входят в учёт ровно один раз. Старые donor labels не меняются.

Угловая классификация даёт восемь попаданий на две длинные границы и восемь
вне активных supports. Она отличается от прежней incoming-donor
классификации: восемь попаданий совпадают со старой donor «archived» группой.
Площадь и состояние этих узлов не назначаются новым физическим точкам.
Старые Gauss weights не являются квадратурой новых усечённых границ.
Сумма четырёх буквальных старых Gauss-площадей отличается от `CGamma`
примерно на `-3.19128e-8 m^2` в каждом семействе; обе меры сохраняются отдельно.

Четыре full-family memory queries читают actual endpoint `step.memory`,
включая Oct0. Сохраняются все замкнутые пересечения настоящих current pieces,
буквальные пять harmonic coefficients p/c и наружные bounds. Dense endpoint,
previous-state память и endpoint anchor не подменяют current memory.
В текущем источнике действительно одна current piece на семейство;
проверяются настоящие замкнутые endpoints, без выдуманной второй piece.

Memory/native clock — 1 000 000 с/stage1, contact clock — 2 000 000 с/step2.
Current source node epoch остаётся `unaccepted_candidate`. Новые пробы
не обновляют память, тепловое состояние или счётчик принятых интервалов.

## Восстановление и проверки

Все allocation/density/memory DTO зарегистрированы и запечатаны. From-dict
восстанавливает ответы из полного аутентифицированного источника и
сверяет весь export. Повторное запечатывание изменённых clocks, p/c,
добавленной cell, source atom area, profile или support не даёт допуска.

Новые 17 тестов проверяют точное замыкание общих границ, положительные
микроскопические/неактивные области, full4memory, все 16 histories/stores,
family profile agreement и отказы при подмене. Независимый Decimal oracle
проверяет literal harmonic fields. Тесты, силовые расчёты и квадратуры
прежнего выпуска 859 повторно не запускаются. Native control: одна allocation, четыре полных
CURRENT memory queries и десять genuine density queries; затем authentic replay.

В родительском validation-каталоге, вне четырёх выходов, сохранены
`baseline_859_bootstrap_source.txt` (raw
`240127f21ee6077212795b512bdab2cb1cd83cf88f6f2293cfd8d1f44a91730d`),
`baseline_859_bootstrap.json` (raw
`e44d4f4512e090469fe2204f64f9f5d943e99447e57e1d99ec3c38bc0076e655`,
selfseal `81c4cbca06a4e15c6ef67f8f5d7999294f678e2f40d4849a82350659416bbb55`)
и reusable baseline producer (raw
`778787b5231c6d03e1c87857038b52889077be0e37290a542fc88acda7c34836`).
Последний допускает ровно четыре заявленных новых пути и сохраняет
все 859 старых источников/закрытых выпусков до/после.

Авторитетный gate02 содержит ровно четыре выхода плюс `report.json`:

| Выход | Bytes | Raw SHA256 |
|---|---:|---|
| `pytest.log` | 111 | `22095aabfd2322bcb9bfba0ef1bb00f7e7ffde039bf80e0b2aabfd1c4411f659` |
| `control_console.log` | 507 | `2edf4dd86ac73de18dd70cc98de3b3edbbfab3a1637463129eddbff4aa4831e7` |
| `native_control/report.json` | 2344 | `b554486533b7faa9bde66ac5ca6daecdb0b80000342de903410f4d536352d704` |
| `native_control/source_packet.json` | 108204947 | `dfda996840a53d54610055dc0a427a12365dda6d3d550971190b0b73a0815dcb` |
| `report.json` | 4919 | `7df81606a168656cd51208a71d7ad425714751f736d9ed4ca50218263e106f95` |

Wrapper selfseal:
`f81c0ab0dbd2aebf59124274a6b8a5211934fdd36363e7b6aebe4fc1ea9ef9a4`.
Native report selfseal:
`35cfaf3711db0aec9905e1dd41ee7b6beb6fe4721c16b79a73e1132fe4ff4fd0`.
Source packet selfseal:
`b55d3e4c55c940eb95d55ba714e1b13a7c4c833615686559822119b9251358d0`.
Allocation binding:
`063f7583032b2ac40539c7ab2b296672baceb1c079b7cfd07a4b6be4f1e1d03b`.
Density queries hash:
`fcce8c9c57b3ac7aa934cb65b0a39de8e5d297adc18c9dfacff9a5f5402cdf2e`;
полный CURRENT memory query list:
`737c33857388bbddd302ab14da5d6535498242cf87bf20b35e19596d321e3e0e`.

В родительском каталоге, вне четырёх выходов:

- `independent_audit.json`: 8554 bytes, raw
  `2d9af7d2a38dcf55afa1269b0fe983fa5f51a6dcec5db93b6c4026dcda29a009`,
  selfseal `64dabc8a7a9d23e1ced792db03ba413f17cbbb7c43a5a0f0be7e727de5b2bf9a`;
- точный `independent_audit_source.txt`: 53327 bytes, raw
  `eafbd34a81e9a6ee1f6d06c368797e4cec6291f445c8efc5e024ac16ffda4572`;
- `baseline_859_independent_readback.json`: raw
  `33819495578b55a5cf8cd2f21ab36b87988eb3d35a327493342c123473883de4`,
  selfseal `6dbd748b1ba0ded094900b0e6ffe22b6a1215c9e60cea1c8a1dff52f0efe8b3f`;
- `baseline_859_gate_readback.json`: raw
  `f048d0fd0f71a5539c9c2e57b3a57e2217483096b6ddc3fe7c496c879553ab10`,
  selfseal `291daf04c1903e29294cf1f9d74c562a38dbefa86d2678e4779ae6e4c3e7703b`.

Для воспроизведения обязательны archived producers/JSON прошлого выпуска
859: `source_theta_allocation_design_source.txt`/JSON (raw
`374f620725cd2b396c2c9fd981e5477b14bdc4470633820760c06481561616f9` /
`0eb62b8a9bd6b24a154b1764f61f78492364b682e9b1f1ba5b3dc247f1645396`),
`original_atom_receiver_accounting_source.txt`/JSON (raw
`8b9d1b641f85345ee450a0e7213ea827104075f316efbf922b2a79c13a8a7999` /
`cd3a28d4d38e43af4bdd4aec194f382409a1f035a5e9b8c5832ce5faab93e8ba`),
stdlib `geometric_native_cold_body_validation/independent_audit_source.txt`
(raw `ee031938177a3497aa7821c883650a89397a9942f6b177e48b9a655153297f6e`)
и весь неизменный bootstrap/reusable ancestry. Полный список raw pins
находится в `baseline859_dependency_raw_sha256` итогового аудитора.
Перед фактическим запуском read-only review исправил nested COMMON-reference
ключи только в scratch producer. Обе прежние версии сохранены как TXT;
они не запускались. Проектные Python-файлы/gate02 не менялись.

Readback без повторного pytest, force/Hessian или сертифицированной квадратуры:

```bash
.venv/bin/python analysis/geometric_native_common_source_allocation_validation/independent_audit_source.txt \
  --release analysis/geometric_native_common_source_allocation_validation/release_validation_02 \
  --baseline analysis/geometric_native_common_source_allocation_validation/baseline_859_reusable_source.txt \
  --out /tmp/native_common_source_allocation_readback.json
```

Контракт следующего физического переноса сохранён как
`physical_handover_design_source.txt`, raw
`43f3107f46d1ddb144bf1e30367d6e3637704ae2eca29ee32bdebf08a4d4d255`.
Это read-only дизайн ближайшей реализации, без нового физического PASS.

## Остаток исходного пункта 1

**Готовы два из трёх прежних блоков; остаётся один блок с тремя условиями.**
В первом условии теперь реализован полный source accounting, но физический
перенос ещё требует согласования исходной площади/памяти с настоящим 3D
материалом, преднапряжениями, мировыми силами и работой. Неактивный receipt
или постоянный архив энергии не создаёт необходимую restoring force.
Далее в том же условии остаются HOT law/mean chart, нагрузки/drag и новое
смешанное равновесие. Затем два прежних условия: whole-clock all-pair
admission/reverse C5→C1 и принятое продолжение с независимой работой,
балансами и joint restart.

Принятых интервалов: **1 прежний / 0 новых**. Пункт 1 остаётся открытым.

Ближайший конкретный контракт внутри первого условия — work-conjugate
carrier replay всех 16 original nodes. До замены их непрерывной моделью
нужно восстановить исходные bank-tip позиции, receiver-carried `n/down`
и полный кинематический Jacobian `B_old`, включая couple от изменения
receiver frame. Доступны буквальные areas/states/traction/work, исходные
directions/owner histories/radial tip values и существующие E/L/G.
Диагностические bank tips не являются аутентифицированными дальними
3D концами исходных 1000-метровых стержней.

Существующий `geometric_native_contact_reference_transport.py` сохраняет
free anchors переносом полного конфигурационного смещения:
`free_current=old_free+(h_current-h_new)`. Это сохраняет strain/energy
в момент передачи, но само по себе не сохраняет мировую силу. Требуется
явный материальный attachment на настоящем cut и сравнение `B_new*T=B_old`
для вложения прежних координат в общую систему из 92 координат, затем
preload/owner-couple и вторые вариации. Исходный frame/director нельзя
молча заменить новым. Carrier использует единственную последовательную
энергию; старые rods не добавляются к ней параллельно. Переход от literal
Gauss areas к `C dtheta` требует отдельного доказательства сохранения
сил/работы; постоянный архив энергии или само угловое распределение
эту проверку не заменяют.
