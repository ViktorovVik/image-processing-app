"""Тесты загрузки, сохранения и подготовки изображений к показу."""

import os
import tempfile
import unittest

import cv2
import numpy as np

from imageapp import image_io

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def gradient_image(height=40, width=60):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)
    image[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    image[:, :, 2] = 128
    return image


class LoadSaveTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.directory = self._directory.name

    def tearDown(self):
        self._directory.cleanup()

    def path(self, name):
        return os.path.join(self.directory, name)

    def test_png_round_trip_is_lossless(self):
        image = gradient_image()
        path = image_io.save_image(self.path("picture.png"), image)
        np.testing.assert_array_equal(image_io.load_image(path), image)

    def test_jpg_round_trip_keeps_size(self):
        image = gradient_image()
        for name in ("picture.jpg", "picture.jpeg", "PICTURE.JPG"):
            path = image_io.save_image(self.path(name), image)
            loaded = image_io.load_image(path)
            self.assertEqual(loaded.shape, image.shape)
            self.assertLess(
                float(np.abs(loaded.astype(int) - image.astype(int)).mean()), 6
            )

    def test_cyrillic_path(self):
        folder = self.path("Мои изображения")
        os.mkdir(folder)
        image = gradient_image()
        path = image_io.save_image(os.path.join(folder, "тест №7.png"), image)
        np.testing.assert_array_equal(image_io.load_image(path), image)

    def test_default_extension_is_png(self):
        path = image_io.save_image(self.path("result"), gradient_image())
        self.assertTrue(path.endswith("result.png"))
        self.assertTrue(os.path.isfile(path))

    def test_grayscale_and_alpha_are_converted_to_bgr(self):
        gray = np.full((10, 12), 77, dtype=np.uint8)
        rgba = np.full((10, 12, 4), 200, dtype=np.uint8)
        for name, data in (("gray.png", gray), ("rgba.png", rgba)):
            ok, buffer = cv2.imencode(".png", data)
            self.assertTrue(ok)
            buffer.tofile(self.path(name))
            loaded = image_io.load_image(self.path(name))
            self.assertEqual(loaded.shape, (10, 12, 3))
            self.assertEqual(loaded.dtype, np.uint8)

    def test_unsupported_extension(self):
        for name in (
            "picture.bmp",
            "picture.gif",
            "document.txt",
            "noextension",
        ):
            with open(self.path(name), "wb") as file:
                file.write(b"data")
            with self.subTest(name=name):
                with self.assertRaises(image_io.ImageFileError):
                    image_io.load_image(self.path(name))

    def test_missing_file(self):
        with self.assertRaises(image_io.ImageFileError):
            image_io.load_image(self.path("missing.png"))

    def test_directory_instead_of_file(self):
        os.mkdir(self.path("folder.png"))
        with self.assertRaises(image_io.ImageFileError):
            image_io.load_image(self.path("folder.png"))

    def test_empty_file(self):
        open(self.path("empty.png"), "wb").close()
        with self.assertRaises(image_io.ImageFileError):
            image_io.load_image(self.path("empty.png"))

    def test_corrupted_file(self):
        with open(self.path("broken.jpg"), "wb") as file:
            file.write(b"this is not an image at all")
        with self.assertRaises(image_io.ImageFileError):
            image_io.load_image(self.path("broken.jpg"))

    def test_truncated_png(self):
        ok, buffer = cv2.imencode(".png", gradient_image())
        self.assertTrue(ok)
        with open(self.path("cut.png"), "wb") as file:
            file.write(buffer.tobytes()[:20])
        with self.assertRaises(image_io.ImageFileError):
            image_io.load_image(self.path("cut.png"))

    def test_save_with_unsupported_extension(self):
        with self.assertRaises(image_io.ImageFileError):
            image_io.save_image(self.path("picture.bmp"), gradient_image())

    def test_save_to_missing_directory(self):
        with self.assertRaises(image_io.ImageFileError):
            image_io.save_image(
                self.path(os.path.join("нет", "picture.png")), gradient_image()
            )


class PreviewTests(unittest.TestCase):
    def test_fit_size_never_enlarges(self):
        self.assertEqual(image_io.fit_size(100, 50, 800, 600), (100, 50, 1.0))

    def test_fit_size_keeps_aspect_ratio(self):
        width, height, scale = image_io.fit_size(4000, 2000, 800, 600)
        self.assertEqual((width, height), (800, 400))
        self.assertAlmostEqual(scale, 0.2)
        width, height, scale = image_io.fit_size(1000, 3000, 800, 600)
        self.assertEqual((width, height), (200, 600))

    def test_fit_size_with_tiny_area(self):
        width, height, _scale = image_io.fit_size(4000, 10, 0, -5)
        self.assertGreaterEqual(width, 1)
        self.assertGreaterEqual(height, 1)

    def test_encode_preview_returns_scaled_png(self):
        image = gradient_image(400, 600)
        data, scale = image_io.encode_preview(image, 300, 300)
        self.assertTrue(data.startswith(PNG_SIGNATURE))
        self.assertAlmostEqual(scale, 0.5)
        decoded = cv2.imdecode(
            np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        self.assertEqual(decoded.shape, (200, 300, 3))

    def test_encode_preview_small_image_is_unchanged(self):
        image = gradient_image()
        data, scale = image_io.encode_preview(image, 1000, 1000)
        self.assertEqual(scale, 1.0)
        decoded = cv2.imdecode(
            np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        np.testing.assert_array_equal(decoded, image)


if __name__ == "__main__":
    unittest.main()
