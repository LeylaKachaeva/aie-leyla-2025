# S03 – eda_cli: мини-EDA для CSV

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

## Запуск API (FastAPI + Uvicorn)

Помимо CLI, в проекте есть HTTP API на FastAPI.  
Приложение описано в модуле eda_cli.api:app.

Из корня проекта (S03/HW04):

```bash
uv sync
uv run uvicorn eda_cli.api:app –reload –port 
```

8000После запуска документация доступна по адресу http://127.0.0.1:8000/docs.

### Эндпоинты API

Базовые маршруты:

- GET /health — проверка доступности сервиса (health-check).
- POST /api/summary — принимает CSV-файл и возвращает краткий EDA‑summary по колонкам в формате JSON.

Дополнительный эндпоинт (HW04):

- POST /api/summary_extended — принимает CSV‑файл и, помимо базового summary,
  возвращает дополнительные метрики качества данных (например, количество пропусков,
  долю константных колонок и high-cardinality признаков).

### Краткий обзор

```bash
uv run eda-cli overvюiew data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

## Тесты

```bash
uv run pytest -q
```
