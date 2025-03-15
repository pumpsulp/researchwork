# Используем официальный образ Python (например, 3.10-slim)
FROM python:3.12-slim

# Устанавливаем рабочую директорию в контейнере
WORKDIR /workspace

# Обновляем пакеты и устанавливаем системные зависимости для сборки
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Копируем файл с зависимостями в контейнер
COPY requirements.txt .

# Обновляем pip и устанавливаем Python-зависимости
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Открываем порт 8888 для Jupyter
EXPOSE 8888

# Команда для запуска JupyterLab (Jupyter сервера)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]