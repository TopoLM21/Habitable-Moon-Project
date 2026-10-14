# Зафиксированный ограниченный совместный пилот

База9bb3f0178b6e8d32ff36367ed313d7948e38d379. Только320 граней.
Preflight initialization16.87s, совместный dt.03125year шаг5.90s,
thermal corrector2 iterations. Свободно4.57GB перед новыми файлами.

После адресных тестов:0.0625year, dt.03125/.015625year (2+4steps),
один restart между fine шагами; budget180s solve, budget360s всей серии
включая загрузки/сохранения. Оценка real solves~36s, restart~6s,
контролируемое birth2steps~12s. Новые каталоги, exclusive checkpoints.
Это проверка связи подсистем, не горизонт рождения плит.

Технические критерии: source bytes неизменны; conserved thermal mass/H
переноса до округления; U/kinetic/generalized forces/momenta change явным
ledger, mapping heat0; native whole-energy guards, nonnegative physical
losses, никакого numerical loss as heat; restart exact. Native birth test
в отдельном искусственном thermal state, с учётом внесённого deltaH,
zero newborn memory и generalized momentum capture. Остановиться при
сертифицированном split или любом guard; пороги/допуски не подбирать.

Практический контроль времени в real interval: общие world mean stresses,
T, damage, mass/birth counts. Не объявлять spatial convergence: прежняя
сферическая80/320 stress difference30.6% остаётся ограничением.
