FROM python:3.10-slim

WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY configs/ ./configs/

# Устанавливаем переменные окружения
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/artifacts/credit_risk_model.pkl
ENV PREPROCESSOR_PATH=/app/artifacts/preprocessor.pkl

# Открываем порт
EXPOSE 8000

# Запускаем сервис
CMD ["uvicorn", "src.service.app:app", "--host", "0.0.0.0", "--port", "8000"]