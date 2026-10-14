# Ограниченное совместное продолжение угловой памяти

База1f7176f2c8fce97039cf1570ea4a82245192ff05; среда после запуска доступна.
Реализован отдельный opt-in `coupled-source-angular-moments/v1`.
Начало — сохранённый `/workspace/genesis-source-angular-01/initial.json`
(SHA2569d8534a3593257625a95f94197c8d60ea0b32d01657c391ee3a115c2024bba90).
Это прежняя **приближённая** минимально-дисперсионная инициализация реального
source, не восстановленная полная угловая история.

Взято initial, а не frozen1yr final: там нет нераспределённого heat buffer
и пропущенной тепловой траектории. Импорт frozen final теперь явно отвергается
с `UNALLOCATED_THERMAL_HISTORY`. Прежний результат/история не переписаны.

## Что теперь активно

Используется существующий `simplified_genesis_nodal.advance` с небольшими
явными experimental hooks механики, фиксации угловой памяти и edge-release.
Его обычный путь остался прежним. Прямой вызов обычного driver для angular
checkpoint запрещён, чтобы не потерять угловую память молча.

- Existing thermal ports: радиогенный и приливный ввод, орбитальная эволюция,
  поглощённое излучение звезды/гиганта, поверхностное излучение и water EOS.
- Existing implicit enthalpy solver на исходном консервативном графе.
- Термодеформация, температурно-зависимая Maxwell-вязкость; прежние
  old-D damage/healing, tidal loading и изменение effective stiffness.
  Отдельная новая зависимость E(T) не вводилась: E/ν остаются исходными.
- Moving-basal traction/drag и геометрия на принятом начале шага.
- Existing phase/mush fractions, whole-cell load-bearing admission,
  conservative generalized-momentum capture и его тепловой вклад.
- Полная угловая память, прежний edge-D law и fracture reserve.

Радиальная температура каждой thermal support общая для трёх угловых точек:
это существующее разрешение thermal graph. Новая угловая температурная
структура или новая физическая ширина повреждения не вводятся.

## Энергия без двойного нагрева

Механические blocks содержат треть исходной объёмной метрики на каждую
угловую точку. В исходные radial thermal nodes поступает угловая сумма
`Q_Maxwell − W_thermal − ΔU_old_damage`, плюс basal drag heat. Для born support
используется его единственный thermal node и три angular memory contributions.
Изменение модулей учитывает **всю** угловую энергию, не только среднюю.

Новая phase-cell уже владела массой и энтальпией до механического рождения.
Admission меняет распределение несущей массы, не добавляет thermal mass/H
или latent heat. Память при рождении равна нулю во всех трёх точках по
прежнему zero-stress правилу; на следующем шаге она развивается. Capture
сохраняет Mv и один раз переносит потерю кинетической энергии в thermal H.

Edge-release считается по полной угловой энергии и один раз поступает в
mechanical reserve; edge heat=0. Численные потери памяти и скорости
сохраняются отдельными ledger-строками и **не** становятся теплом.
Старый frozen heat buffer не добавлялся. Общий шаговый баланс:
`ΔH + ΔU + ΔK + Δfracture_reserve + numerical = thermal_ports + external_work`.

Проверены раздельно node-work identity, modulus-change energy, положительная
диссипация, полная angular edge-release с неизменным H и нулевым повторным
release, native whole balance и независимая cumulative проверка по сохранённым
полным полям/массе. Сохранены фазовые, strain/chart и STEP_RANGE guards;
ошибка откатывает весь angular state.

## Реальный короткий интервал

80 исходных граней, номинально1 год: от0.8684000000000005 до
0.8684009999999998 Myr в fine-series. Шаги .125/.0625 года,8/16 шагов;
сравнение во всех8 общих временах. Первый предварительный шаг.258 s,
предварительная оценка серии с подготовкой/записью<30 s при cap120 s.
Фактически runner с двумя сериями, restart и отдельным birth-case занял
**17.97 s**.5120 граней и долгой эволюции не было.

Основные численные массивы reference M/L/angular memory≈945792 bytes;
checkpoint с reference и текущим состоянием≈16 MB. Это не оценка peakRSS:
Python-объекты, копии и факторизации сюда не входят. Семь основных файлов
суммарно96 708 537 bytes плюс небольшой manifest контрольной подготовки;
свободно≈5.28 GB на заключительном замере.

| Поле, fine-series за1 год | Изменение / результат |
|---|---:|
|Максимальное изменение T среди2722 узлов|0.131464988 K|
|Изменение полной H|−6.34548859550e23 J|
|Net thermal ports|−6.34548882413e23 J|
|Максимальное изменение old-D|1.25596545e-5|
|Финальный max edge-D|0.178914638|
|Насыщенные рёбра|0|
|Новые несущие phase-cells|0|
|Накопленный прирост кристаллической массы в mush|1.00731680e16 kg|
|Fracture reserve|3.77509479233e18 J|
|Изменение eccentricity|−7.66006e-11|
|Орбитальная приливная диссипация за интервал|8.20843931e18 J|

Масса всех thermal nodes сохранена точно. Температура, энтальпия, фаза,
вязкость, old-D/модули, angular stress/memory, edge-D, движение и орбита
теперь меняются совместно. В реальном интервале увеличилась доля твёрдой
фазы, но **whole-cell несущий фронт ещё не продвинулся**: births=0.
Во всех шагах обеих серий одна оболочка и `NOT_SEPARATING`.

## Временная проверка и точность баланса

| Максимум на общих временах | Получено | Предзаданный допуск |
|---|---:|---:|
|World-stress relative L2|0.2219163%|5%|
|Edge-D absolute difference|0.003554314|0.02|
|Old-D absolute difference|2.04350e-7|0.002|
|T absolute difference|6.05893e-5 K|0.05 K|
|Число birth-cells|совпало,0|совпадение|

Все временные допуски выполнены. Это проверка двух шагов на80 гранях,
не пространственная сходимость и не оценка истинности исходной реконструкции.

Механический balance relative ≤1.99e-16, momentum ≤2.35e-15. Общий
термомеханический баланс **не** заявляется с такой точностью: большие
thermal inventories ограничивают float-разрешение. В fine-series его
cumulative остаток2.2800e16 J при суммарном native allowance≈5.9925e17 J
(3.805% allowance); максимальный отдельный шаг использовал6.21% allowance.
Остаток относительно net thermal ports≈3.59e-8. Allowance включает
spacing(H), а не произвольно подобранный новый допуск. Очень малые работы
нельзя доказать одним вычитанием глобальных H; для них отдельно проверены
механический ledger и распределение Q/W по thermal supports.

## Контролируемое рождение — отдельное доказательство

Подготовлено **искусственное начальное тепловое состояние одной ячейки**:
T=solidus−1 K, imposed ΔH=−5.23946167144e24 J. Это не событие реальной серии
и не новая внешняя нагрузка, применённая к её checkpoint.

- Родилась1 ячейка face0/node1 массой1.48011170316e19 kg, с прежним material ID.
- Material front q:0.001731975→0.03125; глубина74.3254→1341.3920 m.
  Это существующее дискретное whole-cell admission, не новая модель
  непрерывного фронта. Moving-basal geometry учитывает его на следующем шаге.
- Angular memory при рождении **точно нулевая** во всех3 точках.
- Capture momentum relative4.02e-16; capture heat0.005869 J, added latent=0.
  Столь малый heat ниже разрешения большой H; запись проверяется локально
  и в существующем общем roundoff allowance, не объявляется различимой
  глобальной энтальпийной прибавкой.
- На следующем шаге норма angular newborn memory4.07527e-8, повторного
  admission нет. Restart после рождения совпал точно с прямым продолжением.
- Whole allowance fractions в двух шагах5.41% и2.61%; thermal mass неизменна.

## Проверки, файлы и ограничения

14 адресных тестов прошли за20.84 s, включая прежние native/default tests.
После добавления проверки фронта, полного modulus-change и STEP_RANGE:
5 coupled-тестов повторно прошли за14.04 s. В реальной fine-series также
проверен disk restart с полной повторной сборкой операторов, exact equality.

Код: `tectonics/experimental_angular_coupled.py`. Минимальные hooks:
`simplified_genesis_nodal.advance` и optional commit solver в edge-rule;
обычные/default пути численно не заменены. Протокол, runner, JSON:
`ANGULAR_COUPLED_PROTOCOL.md`, `angular_coupled_check.py/json`.

Новый каталог `/workspace/genesis-angular-coupled-01`: initial, coarse/fine
final, restart-before, controlled birth/after-birth и compressed fields.
`controlled_preparation.json` связывает хеши двух controlled checkpoints
с явно заданным искусственным ΔH и результатом перехода. Исходные snapshots
не переписывались для добавления этой provenance; manifest хранится рядом.
Хеши/размеры всех восьми файлов и обоих входных checkpoints проверены.
Основной `fine_final.json`,16050748 bytes, SHA256:
`70fac6c0e6df5c091a49eb09a7e81bd486be42392697d61720262f0e13f2ff25`.
Исходный source-derived checkpoint SHA256 не изменён; оригинальный реальный
source также сохранён с SHA256
`94fbbb3cb88a15147d33ab04c8f791470dd92e3d73cf8cba4afc0f361cd583fc`.

Команды из repo с `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PYTHONDONTWRITEBYTECODE=1`, Python `/workspace/habitable-moon-venv/bin/python`:
`-m analysis.genesis_geological_handoff.angular_coupled_check`
(требует новый OUT, не перезаписывает старый опыт);
`-m pytest -q tests/test_experimental_angular_coupled.py tests/test_simplified_genesis_nodal.py tests/test_experimental_edge_weakening.py`.

**Теперь проверен совместный thermal/phase/mechanical путь** с явной угловой
памятью и zero-stress birth. Сохраняются ограничения: неизвестная исходная
угловая история,80-гранная пространственная точность, fixed reference chart,
whole-cell/front discretization, нерегуляризованный по физической ширине
edge-law и отсутствие контактной эволюции после сертифицированного split.
Перенос позднего старого source с неизвестной newborn angular history не
разрешён этим тестом. Четыре пространственных контроля и первый физический
этап остаются открыты. Следующий практический вопрос — конечный контроль
пространственного отклика и чувствительности реконструкции в уже совместном
режиме, не увеличение горизонта ради появления разлома. Defaults/canonical,
архивы и прежние checkpoints сохранены; публикации/платных ресурсов не было.

Продолжение: [ограниченный совместный пространственный контроль80/320 и
годовая временная пара](COUPLED_SPATIAL_RESULT.md). Перенос и replay проверены;
границы применимости и остающиеся физические вопросы перечислены отдельно.
