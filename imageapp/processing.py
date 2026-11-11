"""Операции обработки изображений на основе OpenCV.

Все функции принимают изображение в формате OpenCV (массив NumPy
размером H x W x 3, тип uint8, порядок каналов BGR) и возвращают новое
изображение, не изменяя исходное.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# В OpenCV каналы хранятся в порядке BGR (синий, зелёный, красный).
CHANNEL_INDEX = {"blue": 0, "green": 1, "red": 2}
CHANNEL_TITLES = {"red": "красный", "green": "зелёный", "blue": "синий"}

RED_BGR = (0, 0, 255)

# Ограничение на сторону результата, чтобы повторные повороты
# не привели к исчерпанию памяти.
MAX_SIDE = 12000


class ProcessingError(Exception):
    """Ошибка обработки изображения с сообщением для пользователя."""


def check_image(image: object) -> np.ndarray:
    """Проверить, что передано корректное цветное изображение BGR."""
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or image.size == 0
    ):
        raise ProcessingError("Некорректные данные изображения.")
    return image


def extract_channel(image: np.ndarray, channel: str) -> np.ndarray:
    """Оставить в изображении только один цветовой канал.

    channel: "red", "green" или "blue". Остальные каналы обнуляются,
    поэтому результат отображается в цвете выбранного канала.
    """
    check_image(image)
    if channel not in CHANNEL_INDEX:
        raise ProcessingError("Неизвестный цветовой канал: %r." % (channel,))
    index = CHANNEL_INDEX[channel]
    result = np.zeros_like(image)
    result[:, :, index] = image[:, :, index]
    return result


def negative(image: np.ndarray) -> np.ndarray:
    """Получить негатив: каждое значение v заменяется на 255 - v."""
    check_image(image)
    return cv2.bitwise_not(image)


def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    """Повернуть изображение вокруг центра на angle градусов.

    Положительный угол соответствует повороту против часовой стрелки.
    Размер холста увеличивается так, чтобы изображение не обрезалось;
    появившиеся углы заполняются чёрным цветом.
    """
    check_image(image)
    if not math.isfinite(angle):
        raise ProcessingError("Угол поворота должен быть конечным числом.")

    angle = math.fmod(angle, 360.0)
    if angle == 0:
        return image.copy()

    # Повороты, кратные 90 градусам, выполняются без интерполяции.
    quarters = angle / 90.0
    if quarters == int(quarters):
        return np.ascontiguousarray(np.rot90(image, int(quarters) % 4))

    height, width = image.shape[:2]
    center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0
    matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

    # Размеры прямоугольника, описанного вокруг повёрнутого изображения.
    cos_a, sin_a = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(math.ceil(width * cos_a + height * sin_a - 1e-9))
    new_height = int(math.ceil(width * sin_a + height * cos_a - 1e-9))
    if new_width > MAX_SIDE or new_height > MAX_SIDE:
        raise ProcessingError(
            "Результат поворота слишком велик (%d x %d пикселей). "
            "Сбросьте изменения или выберите изображение меньшего размера."
            % (new_width, new_height)
        )

    # Сдвиг, совмещающий центр изображения с центром нового холста.
    matrix[0, 2] += (new_width - 1) / 2.0 - center_x
    matrix[1, 2] += (new_height - 1) / 2.0 - center_y

    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def default_thickness(image: np.ndarray) -> int:
    """Толщина линии окружности, пропорциональная размеру изображения."""
    height, width = image.shape[:2]
    return max(2, int(round(min(height, width) / 200.0)))


def draw_circle(
    image: np.ndarray,
    center_x: int,
    center_y: int,
    radius: int,
    filled: bool = False,
) -> np.ndarray:
    """Нарисовать красный круг с центром (center_x, center_y).

    filled=False — рисуется контур круга, filled=True — круг закрашивается.
    """
    check_image(image)
    if radius < 1:
        raise ProcessingError("Радиус круга должен быть не меньше 1 пикселя.")
    result = image.copy()
    thickness = -1 if filled else default_thickness(image)
    cv2.circle(
        result,
        (int(center_x), int(center_y)),
        int(radius),
        RED_BGR,
        thickness,
        lineType=cv2.LINE_AA,
    )
    return result
