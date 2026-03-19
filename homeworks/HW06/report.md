# HW06 – Report


> Файл: `homeworks/HW06/report.md`  
> Важно: не меняйте названия разделов (заголовков). Заполняйте текстом и/или вставляйте результаты.


## 1. Dataset


- Какой датасет выбран: `S06-hw-dataset-04.csv`
- Размер: (25000, 62) — 25 000 строк и 62 столбца (включая `id` и `target`)
- Целевая переменная: `target` (бинарная, классы 0/1)
  - Доли классов: класс 0 — 0.9508, класс 1 — 0.0492 (сильный дисбаланс)
- Признаки: все признаки, кроме `id` и `target`, числовые (float); `id` — идентификатор и не использовался как признак


## 2. Protocol


- Разбиение: train/test = 80/20, `random_state=42`, `stratify=y` (чтобы сохранить доли классов в обеих выборках при дисбалансе)
- Подбор: CV на train — `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` + `GridSearchCV`, оптимизация по `average_precision` (AP), `refit="ap"`
- Метрики: accuracy, F1, ROC-AUC (+ average_precision)
  - Почему эти метрики уместны: из‑за дисбаланса accuracy может быть высокой даже у «пустой» модели, поэтому важнее смотреть F1, ROC-AUC и особенно average_precision (качество по PR-логике и по редкому классу)


## 3. Models


Сравнивались модели:

- DummyClassifier (baseline)
  - `strategy="most_frequent"`
- LogisticRegression (baseline из S05)
  - `Pipeline(StandardScaler → LogisticRegression(max_iter=2000, random_state=42))`
- DecisionTreeClassifier (контроль сложности)
  - Подбирались: `max_depth`, `min_samples_leaf`, `ccp_alpha`
- RandomForestClassifier
  - `n_estimators=150`
  - Подбирались: `max_depth`, `min_samples_leaf`, `max_features`
- Boosting: HistGradientBoostingClassifier
  - Подбирались: `max_depth`, `learning_rate`, `max_leaf_nodes`

Опционально:
- StackingClassifier не использовался.


## 4. Results


Финальные метрики на test:

| Model               | accuracy | f1     | roc_auc | average_precision |
|---------------------|---------:|-------:|--------:|------------------:|
| HistGB              | 0.9792   | 0.7374 | 0.8865  | 0.7827           |
| RandomForest        | 0.9716   | 0.5943 | 0.9016  | 0.7750           |
| DecisionTree        | 0.9672   | 0.5900 | 0.8010  | 0.5437           |
| LogReg              | 0.9632   | 0.4286 | 0.8340  | 0.5088           |
| Dummy_most_frequent | 0.9508   | 0.0000 | 0.5000  | 0.0492           |

- Победитель: HistGradientBoostingClassifier (лучший по согласованному критерию `average_precision`, также лучший по F1)
- Короткое объяснение: на дисбалансной задаче `average_precision` и F1 лучше отражают качество по редкому классу, чем одна accuracy


## 5. Analysis


- Устойчивость:
  - В основной работе использовался фиксированный `random_state=42` для воспроизводимости.
  - Дополнительные 5 прогонов с разными `random_state` (опциональная часть) не выполнялись; ожидается, что ансамбли (RandomForest/HistGB) будут более стабильны, чем одиночное дерево.

- Ошибки:
  - Для лучшей модели построена confusion matrix при пороге 0.5.
  - Комментарий: при сильном дисбалансе важно смотреть ошибки по классу 1 (ложные пропуски и ложные срабатывания), поэтому дополнительно использовались PR-кривая и метрика average_precision.

- Интерпретация (permutation importance):
  - Для лучшей модели рассчитан permutation importance по метрике `average_precision`.
  - Top-10 признаков:
    1) f25
    2) f58
    3) f54
    4) f38
    5) f47
    6) f53
    7) f04
    8) f33
    9) f13
    10) f11
  - Вывод: признаки f25 и f58 оказывают наибольшее влияние на качество (при их перестановке качество падает сильнее всего); остальные признаки также важны, но в меньшей степени.


## 6. Conclusion


- Одиночное дерево легко переобучается, поэтому нужен контроль сложности (max_depth/min_samples_leaf/ccp_alpha).
- Ансамбли (RandomForest и boosting) дают лучшее качество на табличных данных, чем линейный baseline и одиночное дерево.
- На сильном дисбалансе одна accuracy не показывает реальное качество — наивная модель может иметь высокую accuracy, но нулевой F1.
- Для задач с редким позитивным классом полезно ориентироваться на F1 и average_precision (PR-логика).
- Честный протокол (фиксированный train/test + CV только на train + test один раз) помогает избежать утечек и переоценки качества.
- Permutation importance позволяет интерпретировать лучшую модель и понять, какие признаки вносят основной вклад.
