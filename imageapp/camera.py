"""Получение фотоснимка с веб-камеры средствами OpenCV."""

from __future__ import annotations

import sys
from typing import List

import cv2
import numpy as np

MAX_CAMERA_INDEX = 9
WARMUP_FRAMES = 8

TROUBLESHOOTING = (
    "Возможные пути решения:\n"
    "1. Проверьте, что веб-камера подключена и включена "
    "(на ноутбуке — что не закрыта шторка и камера не отключена клавишей).\n"
    "2. Закройте программы, которые могут использовать камеру "
    "(Zoom, Skype, Telegram, вкладки браузера).\n"
    "3. Разрешите доступ к камере в настройках конфиденциальности "
    "операционной системы (в macOS — для приложения «Терминал»).\n"
    "4. Если камер несколько, укажите другой номер камеры (0, 1, 2…).\n"
    "5. Обновите или переустановите драйвер камеры; "
    "в Linux проверьте наличие устройства /dev/video0 и права доступа.\n"
    "6. Если камеры нет, загрузите изображение из файла."
)


class CameraError(Exception):
    """Не удалось получить снимок с веб-камеры."""


def _backends() -> List[int]:
    """Список программных интерфейсов захвата для текущей ОС."""
    if sys.platform.startswith("win"):
        # DirectShow открывает камеру заметно быстрее интерфейса по умолчанию.
        return [cv2.CAP_DSHOW, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def capture_frame(index: int = 0) -> np.ndarray:
    """Сделать снимок с веб-камеры номер index и вернуть его в формате BGR."""
    if index < 0 or index > MAX_CAMERA_INDEX:
        raise CameraError(
            "Номер камеры должен быть от 0 до %d." % MAX_CAMERA_INDEX
        )

    opened = False
    for backend in _backends():
        capture = cv2.VideoCapture(index, backend)
        try:
            if not capture.isOpened():
                continue
            opened = True
            frame = None
            # Первые кадры часто получаются тёмными, пока камера
            # подстраивает экспозицию, поэтому берётся последний из нескольких.
            for _ in range(WARMUP_FRAMES):
                ok, candidate = capture.read()
                if ok and candidate is not None and candidate.size > 0:
                    frame = candidate
            if frame is not None:
                return _to_bgr(frame)
        finally:
            capture.release()

    if opened:
        raise CameraError(
            "Камера №%d найдена, но не передаёт изображение." % index
        )
    raise CameraError("Не удалось подключиться к веб-камере №%d." % index)


def _to_bgr(frame: np.ndarray) -> np.ndarray:
    """Привести кадр к трёхканальному формату BGR uint8."""
    if frame.dtype != np.uint8:
        frame = cv2.convertScaleAbs(frame)
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame
