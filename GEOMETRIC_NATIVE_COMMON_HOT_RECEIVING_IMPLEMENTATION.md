# Поля принимающей HOT стороны и перенос заданной работы

Работа начата07.10.2026. Ветка `fix/genesis-mantle-convection`, компонент909.

Реализован следующий необходимый элемент физической связи COLD–HOT:
полные jets положения и original-material gradient горячей стороны,
а также перенос заданных пространственных сил/моментов в общие координаты.
Модуль: `tectonics/geometric_native_common_hot_receiving.py`.

Получены12 пространственных channels:3 положения и9 компонент материального
градиента. У каждого полный first/second jet по9 активным координатам:
3 fresh owner rotations и6 HOT finite-mean coordinates. Глобальные индексы
соответствуют существующему100-coordinate каталогу; новые степени свободы
не добавлены. Поддерживаются ровно4 настоящих saved charts903 и15 настоящих
original polygon labels: roof, interior, true base для всех5 HOT тел.
Реальные маленькие birth bodies не исключены.

## Математика и единицы

Prepared X0 и A0 используют ORIGINAL_SOURCE_WORLD_CARTESIAN input.
Polygon pose уже применён один раз; source owner history не даёт второго
поворота. Для исходных R,z,q и shared angular rate omega:

`r³=R³−3R²*z`, `Q=exp(z*[omega]cross)`,
`X0=s*r*Q*q`,
`A0=s*Q*(I−r³/R²*(omega×q)⊗q)`.

Источник authentic903 сохраняется целиком с seal
`c53e4cd75219328eeac974f3b9982aa33ad78eaf2e07d823feafb9f07b289e6c`.
Источник имеет OTHER input frame, чем COLD895 original-parent Cartesian.
При сопряжении нормалей потребуется явное согласование этих input frames;
простая подстановка одного A вместо другого недопустима.

`Fhot=I+(h*I+sum(b_k*G_k))/R`, `y=Fhot*X0`, `A=Fhot*A0`.
Saved implicit chart Jacobian J и scalar Hessian hs используются один раз:
`y_,j=F_,j*X0`, `y_,jk=X0*hs_jk/R`; те же производные присутствуют у A.
Fresh left-exponential owner rotation действует на y и на строки A.
Все owner/body cross derivatives, symmetric owner second variations и
finite-mean curvature retained. Direct determinants сверены с inherited
positive whole-body Jacobian theorem; theorem не объявляется заново доказанной.

Обобщённая работа для заданного potential jet на12 channels:

`g_a=sum_i f_i*x_i,a`,
`H_ab=sum_ij x_i,a*K_ij*x_j,b + sum_i f_i*x_i,ab`.

Channels x содержат и положение, и material gradient. Поэтому f включает
силы N и gradient-conjugate orientation hyperstress N*m; последний член
сохраняет prestress/chart curvature. Owner generalized force — N*m,
HOT generalized force — N. Переданный K обязан сохранять все12×12 mixed
entries. Замена его одним normal spring tangent или удаление A channels
теряет ориентационную работу. Сам адаптер не определяет physical traction law.

## Фактические проверки

Fresh closed907 reader PASS до новых источников; отдельный pre-math
907/ancestry guard PASS26.608663680s. Старые907PY/P135F4 и вся903/899/
895/891/887 ancestry сохранены. Ровно2 additive Python sources →909.

Frozen gate **10/10tests PASS50.45s**, actual exit0/stderr0;
**60 full field queries**:4 charts ×15 labels.
Native point/restored-response control PASS98.646469473s;
gate PASS215.339746564s. Готовые mean roots и tensor integrals903
используются из immutable DTO; старые force solves/tests/body quadrature
controls не исполняются. Full point restart требует exact query/label/field/
history; changed/resealed sources и буквальные неподдержанные kinds отклоняются.

Independent audit **PASS68.215278536s**,220digits,
**71640 numerical comparisons**. Собственные dense
second-order dual numbers воспроизводят y/A и owner rotations; исходный
world gradient отдельно восстановлен через Rodrigues rotation/rank-one map.
Все12×9 first и12×9×9 second entries, supplied potential force9 иH9×9
сверены по всем60 запросам. Actual stdout побайтово равен report, stderr0.
Представители input intervals служат algebra diagnostics, а не новыми
physical exact material points; geometric/mean-root proof остаётся inherited.

До frozen gate сохранён development failure:8 tests passed,2 failed
из-за неверного equality assertion для outward binary enclosure дробей1/5
и2/5. Исправлены требования containment; формулы не подменены exact-point
вычислением. Также сохранён pre-test import failure system Python безpytest;
тесты исполнены project virtualenv, audit — system Python сmpmath.

[Gate](analysis/geometric_native_common_hot_receiving_validation/report.json),
[audit](analysis/geometric_native_common_hot_receiving_validation/audit_report.json),
[packet](analysis/geometric_native_common_hot_receiving_validation/source_packet.json),
[manifest](analysis/geometric_native_common_hot_receiving_validation/artifact_manifest.json),
[final readback](analysis/geometric_native_common_hot_receiving_final_readback/readback.json),
[archive4](analysis/geometric_native_common_hot_receiving_final_readback/archive.json).

Документы завершены до manifest/readback closure. Actual final statuses,
file counts и timestamps находятся в этих JSON. Закрытые909 и907 bundles
после закрытия не дополнять и не переписывать.

## Остаток пункта1

**Пункт1 открыт: один существенный оставшийся блок, те же3 условия.**

1. Physical material/old-force/prestress/work handover, HOT law/mean,
   physical loads/drag и NEW mixed equilibrium. Cold899/HOT903/mixed907
   и HOT receiving field/work adapter909 готовы. Осталось сопрячь обе
   стороны всей genuine COLD–HOT границы, определить и интегрировать её
   physical receiving traction law; удостоверить handover/reference work,
   modal physical loads, достаточную точность/general states и mixed solve.
2. Whole-clock all-pair admission, включая reverse C5→C1.
3. Accepted continuation, независимая temporal work, балансы и joint restart.

Проверенные15 polygon labels не покрывают весь curved receiving cut и
positive paired correction. Complete receiving-face coverage и physical
receiving law не заявлены. Diagnostic supplied work не становится physical
contact work. New mixed equilibrium, source-work handover, admission и
clocks False; принято1 old и0 new intervals. Пункты2–6 исходного плана открыты.

**Следующий конкретный шаг внутри того же условия1:** собрать согласованные
материальные traces обеих сторон полного подлинного COLD–HOT cut и связать
их с contact traction law. HOT point curvature/work pullback для этого готов.

В следующей среде:

```bash
python analysis/geometric_native_common_hot_receiving_validation/baseline_restore_source.txt --output /workspace/hot_receiving_restore_next.json
python analysis/geometric_native_common_hot_receiving_final_readback/readback_source.txt --output /workspace/hot_receiving_fresh_next.json
```

Stdlib restorer проверяет closed909 и восстанавливает54 outside paths:
50 inherited907 paths плюс4 fresh907 baseline files. Он вызывает сохранённый
stdlib byte-restorer907; старые reader/guard/auditor/physical mains не
исполняет, repository files не пишет. Reader проверяет evidence/byte namespaces,
не повторяя математику. Авторитетный final archive —4 файла.
