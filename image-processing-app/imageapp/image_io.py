"""Загрузка, сохранение и подготовка изображений к показу в окне."""

from __future__ import annotations

import os
from typing import Tuple

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg")


class ImageFileError(Exception):
    """Ошибка чтения или записи файла (текст — для пользователя)."""


def _extension(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def load_image(path: str) -> np.ndarray:
    """Загрузить изображение PNG или JPG и вернуть его в формате BGR.

    Файл читается через numpy.fromfile и cv2.imdecode, а не cv2.imread:
    так корректно обрабатываются пути с кириллицей в Windows.
    """
    extension = _extension(path)
    if extension not in SUPPORTED_EXTENSIONS:
        raise ImageFileError(
            "Неподдерживаемый формат файла «%s». "
            "Выберите изображение PNG или JPG."
            % (extension or "без расширения")
        )
    if not os.path.isfile(path):
        raise ImageFileError("Файл не найден: %s" % path)
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as error:
        raise ImageFileError(
            "Не удалось прочитать файл: %s" % error
        ) from error
    if data.size == 0:
        raise ImageFileError("Файл пуст: %s" % path)
    try:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except cv2.error:
        image = None
    if image is None or image.size == 0:
        raise ImageFileError(
            "Файл повреждён или не является изображением PNG/JPG: %s" % path
        )
    return image


def save_image(path: str, image: np.ndarray) -> str:
    """Сохранить изображение в PNG или JPG. Возвращает итоговый путь."""
    extension = _extension(path)
    if not extension:
        path += ".png"
        extension = ".png"
    if extension not in SUPPORTED_EXTENSIONS:
        raise ImageFileError(
            "Сохранение возможно только в форматах PNG и JPG (указано «%s»)."
            % extension
        )
    try:
        ok, buffer = cv2.imencode(extension, image)
    except cv2.error as error:
        raise ImageFileError(
            "Не удалось закодировать изображение: %s" % error
        ) from error
    if not ok:
        raise ImageFileError("Не удалось закодировать изображение.")
    try:
        buffer.tofile(path)
    except OSError as error:
        raise ImageFileError("Не удалось записать файл: %s" % error) from error
    return path


def fit_size(
    width: int, height: int, max_width: int, max_height: int
) -> Tuple[int, int, float]:
    """Вписать изображение в область max_width x max_height без увеличения.

    Возвращает новые ширину, высоту и коэффициент масштабирования.
    """
    max_width, max_height = max(1, max_width), max(1, max_height)
    scale = min(1.0, max_width / float(width), max_height / float(height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return new_width, new_height, scale


def encode_preview(
    image: np.ndarray, max_width: int, max_height: int
) -> Tuple[bytes, float]:
    """Подготовить изображение к показу: уменьшить и закодировать в PNG.

    Возвращает байты PNG и коэффициент масштабирования относительно оригинала.
    """
    height, width = image.shape[:2]
    new_width, new_height, scale = fit_size(
        width, height, max_width, max_height
    )
    if (new_width, new_height) != (width, height):
        image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ImageFileError("Не удалось подготовить изображение к показу.")
    return buffer.tobytes(), scale
