"""Тесты операций обработки изображений."""

import unittest
from unittest import mock

import numpy as np

from imageapp import processing


def sample_image(height=60, width=80):
    """Цветное тестовое изображение с разными значениями во всех каналах."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


class ExtractChannelTests(unittest.TestCase):
    def test_only_selected_channel_is_kept(self):
        image = sample_image()
        for name, index in (("blue", 0), ("green", 1), ("red", 2)):
            result = processing.extract_channel(image, name)
            np.testing.assert_array_equal(
                result[:, :, index], image[:, :, index]
            )
            for other in {0, 1, 2} - {index}:
                self.assertEqual(int(result[:, :, other].max()), 0)

    def test_source_image_is_not_modified(self):
        image = sample_image()
        copy = image.copy()
        processing.extract_channel(image, "red")
        np.testing.assert_array_equal(image, copy)

    def test_unknown_channel_is_rejected(self):
        with self.assertRaises(processing.ProcessingError):
            processing.extract_channel(sample_image(), "alpha")

    def test_invalid_image_is_rejected(self):
        with self.assertRaises(processing.ProcessingError):
            processing.extract_channel(
                np.zeros((10, 10), dtype=np.uint8), "red"
            )
        with self.assertRaises(processing.ProcessingError):
            processing.extract_channel(None, "red")


class NegativeTests(unittest.TestCase):
    def test_values_are_inverted(self):
        image = sample_image()
        np.testing.assert_array_equal(processing.negative(image), 255 - image)

    def test_double_negative_restores_image(self):
        image = sample_image()
        np.testing.assert_array_equal(
            processing.negative(processing.negative(image)), image
        )


class RotateTests(unittest.TestCase):
    def test_zero_and_full_turn_keep_image(self):
        image = sample_image()
        for angle in (0, 360, -360):
            np.testing.assert_array_equal(
                processing.rotate(image, angle), image
            )

    def test_right_angles_are_exact(self):
        image = sample_image()
        np.testing.assert_array_equal(
            processing.rotate(image, 90), np.rot90(image, 1)
        )
        np.testing.assert_array_equal(
            processing.rotate(image, 180), np.rot90(image, 2)
        )
        np.testing.assert_array_equal(
            processing.rotate(image, -90), np.rot90(image, 3)
        )
        np.testing.assert_array_equal(
            processing.rotate(image, 270), np.rot90(image, 3)
        )

    def test_positive_angle_is_counterclockwise(self):
        image = np.zeros((101, 101, 3), dtype=np.uint8)
        image[45:56, 85:96] = 255  # метка справа от центра
        rotated = processing.rotate(image, 90)
        ys, xs = np.nonzero(rotated[:, :, 0])
        self.assertLess(ys.mean(), 30)  # метка оказалась сверху
        self.assertAlmostEqual(xs.mean(), 50, delta=1)

    def test_arbitrary_angle_expands_canvas(self):
        image = np.full((100, 200, 3), 200, dtype=np.uint8)
        rotated = processing.rotate(image, 45)
        height, width = rotated.shape[:2]
        expected = int(np.ceil((200 + 100) / np.sqrt(2)))
        self.assertAlmostEqual(width, expected, delta=1)
        self.assertAlmostEqual(height, expected, delta=1)
        # центр остаётся закрашенным, углы холста — чёрными
        self.assertEqual(int(rotated[height // 2, width // 2, 0]), 200)
        self.assertEqual(int(rotated[0, 0, 0]), 0)

    def test_arbitrary_angle_keeps_content(self):
        image = np.zeros((120, 120, 3), dtype=np.uint8)
        image[40:80, 40:80] = (10, 200, 90)
        rotated = processing.rotate(image, 30)
        # площадь закрашенной области сохраняется с точностью до интерполяции
        area = int((rotated[:, :, 1] > 100).sum())
        self.assertAlmostEqual(area, 40 * 40, delta=80)

    def test_fractional_angle(self):
        rotated = processing.rotate(sample_image(), 12.5)
        self.assertEqual(rotated.dtype, np.uint8)
        self.assertEqual(rotated.shape[2], 3)

    def test_non_finite_angle_is_rejected(self):
        for angle in (float("nan"), float("inf")):
            with self.assertRaises(processing.ProcessingError):
                processing.rotate(sample_image(), angle)

    def test_too_large_result_is_rejected(self):
        image = np.zeros((90, 90, 3), dtype=np.uint8)
        with mock.patch.object(processing, "MAX_SIDE", 100):
            with self.assertRaises(processing.ProcessingError):
                processing.rotate(image, 45)
            # поворот на прямой угол не увеличивает холст и остаётся допустимым
            self.assertEqual(processing.rotate(image, 90).shape, image.shape)

    def test_source_image_is_not_modified(self):
        image = sample_image()
        copy = image.copy()
        processing.rotate(image, 33)
        processing.rotate(image, 90)
        np.testing.assert_array_equal(image, copy)


class DrawCircleTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((200, 300, 3), 255, dtype=np.uint8)

    def assert_red(self, pixel):
        self.assertEqual(tuple(int(v) for v in pixel), processing.RED_BGR)

    def test_outline_is_red_and_center_untouched(self):
        result = processing.draw_circle(self.image, 150, 100, 50)
        self.assert_red(result[100, 200])  # точка на окружности справа
        self.assert_red(result[50, 150])  # точка на окружности сверху
        self.assertEqual(
            tuple(int(v) for v in result[100, 150]), (255, 255, 255)
        )

    def test_filled_circle_covers_center(self):
        result = processing.draw_circle(self.image, 150, 100, 50, filled=True)
        self.assert_red(result[100, 150])
        self.assert_red(result[100, 180])
        self.assertEqual(
            tuple(int(v) for v in result[100, 260]), (255, 255, 255)
        )

    def test_filled_area_matches_radius(self):
        result = processing.draw_circle(self.image, 150, 100, 40, filled=True)
        red = int(np.all(result == processing.RED_BGR, axis=2).sum())
        # площадь закрашенной области лежит между кругами радиусов 39 и 41
        self.assertGreater(red, np.pi * 39**2)
        self.assertLess(red, np.pi * 41**2)

    def test_circle_partly_outside_image(self):
        result = processing.draw_circle(self.image, 0, 0, 120, filled=True)
        self.assert_red(result[0, 0])
        self.assertEqual(result.shape, self.image.shape)

    def test_source_image_is_not_modified(self):
        copy = self.image.copy()
        processing.draw_circle(self.image, 150, 100, 50, filled=True)
        np.testing.assert_array_equal(self.image, copy)

    def test_non_positive_radius_is_rejected(self):
        for radius in (0, -5):
            with self.assertRaises(processing.ProcessingError):
                processing.draw_circle(self.image, 10, 10, radius)


if __name__ == "__main__":
    unittest.main()
