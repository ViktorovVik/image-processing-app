"""Тесты проверки пользовательского ввода."""

import unittest

from imageapp.validation import (
    InputError,
    parse_angle,
    parse_circle,
    parse_int,
)


class ParseAngleTests(unittest.TestCase):
    def test_valid_values(self):
        self.assertEqual(parse_angle("45"), 45.0)
        self.assertEqual(parse_angle(" -90 "), -90.0)
        self.assertEqual(parse_angle("12.5"), 12.5)
        self.assertEqual(parse_angle("12,5"), 12.5)
        self.assertEqual(parse_angle("+360"), 360.0)
        self.assertEqual(parse_angle("0"), 0.0)

    def test_invalid_values(self):
        for text in (
            "",
            "   ",
            "abc",
            "45 градусов",
            "1.2.3",
            "--5",
            "nan",
            "inf",
            "361",
            "-360.5",
            "1e9",
            "45°",
        ):
            with self.subTest(text=text):
                with self.assertRaises(InputError):
                    parse_angle(text)


class ParseIntTests(unittest.TestCase):
    def test_valid_values(self):
        self.assertEqual(parse_int("10", "X", 0, 100), 10)
        self.assertEqual(parse_int(" 0 ", "X", 0, 100), 0)
        self.assertEqual(parse_int("+7", "X", 0, 100), 7)
        self.assertEqual(parse_int("100", "X", 0, 100), 100)

    def test_invalid_values(self):
        for text in (
            "",
            " ",
            "abc",
            "1.5",
            "1,5",
            "10px",
            "1 0",
            "-1",
            "101",
            "99999999999999999999",
            "0x10",
            "١٢",
        ):
            with self.subTest(text=text):
                with self.assertRaises(InputError):
                    parse_int(text, "X", 0, 100)

    def test_message_names_the_field(self):
        with self.assertRaises(InputError) as context:
            parse_int("abc", "Радиус", 1, 10)
        self.assertIn("Радиус", str(context.exception))


class ParseCircleTests(unittest.TestCase):
    def test_valid_circle(self):
        self.assertEqual(
            parse_circle("100", "50", "30", 640, 480), (100, 50, 30)
        )
        self.assertEqual(parse_circle("0", "0", "1", 640, 480), (0, 0, 1))
        self.assertEqual(
            parse_circle("639", "479", "640", 640, 480), (639, 479, 640)
        )

    def test_center_outside_image(self):
        for x, y in (("640", "10"), ("10", "480"), ("-1", "10"), ("10", "-1")):
            with self.subTest(x=x, y=y):
                with self.assertRaises(InputError):
                    parse_circle(x, y, "10", 640, 480)

    def test_invalid_radius(self):
        for radius in ("0", "-3", "641", "", "большой", "2.5"):
            with self.subTest(radius=radius):
                with self.assertRaises(InputError):
                    parse_circle("10", "10", radius, 640, 480)


if __name__ == "__main__":
    unittest.main()
