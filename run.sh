#!/bin/sh
# Сборка окружения и запуск приложения в Linux и macOS.
# Использование:  sh run.sh
# Другой интерпретатор:  PYTHON=python3.12 sh run.sh
set -e
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

if [ ! -x .venv/bin/python ]; then
    echo "Создание виртуального окружения .venv ..."
    "$PYTHON" -m venv .venv
    echo "Установка зависимостей ..."
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
fi

exec .venv/bin/python main.py
