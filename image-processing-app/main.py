"""Точка входа: запуск десктоп-приложения для обработки изображений.

Запуск из корня проекта:  python main.py
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import tkinter
    except ImportError:
        print(
            "Не найден модуль tkinter. В Linux установите его командой:\n"
            "    sudo apt install python3-tk\n"
            "В Windows и macOS tkinter входит в установщик Python "
            "с сайта python.org.",
            file=sys.stderr,
        )
        return 1

    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as error:
        print(
            "Не установлены зависимости приложения (%s).\n"
            "Активируйте виртуальное окружение и выполните:\n"
            "    python -m pip install -r requirements.txt" % error,
            file=sys.stderr,
        )
        return 1

    from imageapp.gui import ImageApp

    try:
        app = ImageApp()
    except tkinter.TclError as error:
        print(
            "Не удалось открыть окно приложения: %s" % error, file=sys.stderr
        )
        return 1
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
