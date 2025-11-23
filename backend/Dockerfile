FROM python:3.13-slim

WORKDIR /app

# Установка системных библиотек (нужны для pandas/numpy)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Создаем папки для данных и моделей
RUN mkdir -p /app/data /app/models

# Копируем код
COPY . .

ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["python", "cmd/main.py"]