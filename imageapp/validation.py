"""Проверка данных, введённых пользователем в поля формы."""

from __future__ import annotations

import math
import re
from typing import Tuple

MAX_ANGLE = 360.0
_INT_PATTERN = re.compile(r"[+-]?[0-9]{1,9}")


class InputError(ValueError):
    """Некорректный ввод; текст исключения показывается пользователю."""


def parse_angle(text: str) -> float:
    """Разобрать угол поворота в градусах.

    Допускаются целые и дробные числа, в качестве разделителя дробной
    части можно использовать точку или запятую. Диапазон: от -360 до 360.
    """
    value = text.strip().replace(",", ".")
    if not value:
        raise InputError("Введите угол поворота в градусах.")
    try:
        angle = float(value)
    except ValueError:
        raise InputError(
            "Угол поворота должен быть числом, например 45 или -12.5. "
            "Введено: «%s»." % text.strip()
        ) from None
    if not math.isfinite(angle):
        raise InputError("Угол поворота должен быть конечным числом.")
    if abs(angle) > MAX_ANGLE:
        raise InputError(
            "Угол поворота должен быть в диапазоне от -360 до 360 градусов."
        )
    return angle


def parse_int(text: str, field_name: str, minimum: int, maximum: int) -> int:
    """Разобрать целое число из поля field_name и проверить диапазон."""
    value = text.strip()
    if not value:
        raise InputError("Заполните поле «%s»." % field_name)
    if not _INT_PATTERN.fullmatch(value):
        raise InputError(
            "В поле «%s» нужно ввести целое число. Введено: «%s»."
            % (field_name, value)
        )
    number = int(value)
    if number < minimum or number > maximum:
        raise InputError(
            "Значение в поле «%s» должно быть от %d до %d. Введено: %d."
            % (field_name, minimum, maximum, number)
        )
    return number


def parse_circle(
    x_text: str, y_text: str, radius_text: str, width: int, height: int
) -> Tuple[int, int, int]:
    """Разобрать параметры круга для изображения размером width x height.

    Центр круга должен находиться внутри изображения, радиус — от 1 пикселя
    до наибольшей стороны изображения.
    """
    x = parse_int(x_text, "X центра", 0, width - 1)
    y = parse_int(y_text, "Y центра", 0, height - 1)
    radius = parse_int(radius_text, "Радиус", 1, max(width, height))
    return x, y, radius
