"""Тесты получения снимка с веб-камеры (камера заменяется имитацией)."""

import unittest
from unittest import mock

import numpy as np

from imageapp import camera


class FakeCapture:
    """Имитация cv2.VideoCapture с заданным поведением."""

    def __init__(self, opened=True, frames=None):
        self._opened = opened
        self._frames = list(frames or [])
        self.released = False

    def isOpened(self):  # noqa: N802 - имя метода задано интерфейсом OpenCV
        return self._opened

    def read(self):
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def release(self):
        self.released = True


def patch_capture(fake):
    return mock.patch.object(camera.cv2, "VideoCapture", return_value=fake)


class CaptureFrameTests(unittest.TestCase):
    def test_returns_last_warmup_frame(self):
        frames = [
            np.full((4, 6, 3), value, dtype=np.uint8)
            for value in range(camera.WARMUP_FRAMES)
        ]
        fake = FakeCapture(frames=frames)
        with patch_capture(fake):
            frame = camera.capture_frame(0)
        self.assertEqual(int(frame[0, 0, 0]), camera.WARMUP_FRAMES - 1)
        self.assertTrue(fake.released)

    def test_grayscale_frame_is_converted_to_bgr(self):
        fake = FakeCapture(frames=[np.full((4, 6), 50, dtype=np.uint8)])
        with patch_capture(fake):
            frame = camera.capture_frame(0)
        self.assertEqual(frame.shape, (4, 6, 3))

    def test_camera_not_opened(self):
        fake = FakeCapture(opened=False)
        with patch_capture(fake):
            with self.assertRaises(camera.CameraError) as context:
                camera.capture_frame(0)
        self.assertIn("Не удалось подключиться", str(context.exception))
        self.assertTrue(fake.released)

    def test_camera_without_frames(self):
        fake = FakeCapture(opened=True, frames=[])
        with patch_capture(fake):
            with self.assertRaises(camera.CameraError) as context:
                camera.capture_frame(1)
        self.assertIn("не передаёт изображение", str(context.exception))
        self.assertTrue(fake.released)

    def test_invalid_index(self):
        for index in (-1, camera.MAX_CAMERA_INDEX + 1):
            with self.assertRaises(camera.CameraError):
                camera.capture_frame(index)

    def test_troubleshooting_text_is_present(self):
        self.assertIn("Возможные пути решения", camera.TROUBLESHOOTING)


if __name__ == "__main__":
    unittest.main()
