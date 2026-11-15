import unittest

from PIL import Image, ImageDraw

from engine.dessert_measure import measure_dessert


def parfait(middle):
    image = Image.new("RGB", (240, 380), (45, 47, 57))
    draw = ImageDraw.Draw(image)
    draw.polygon([(61, 164), (173, 164), (141, 337), (98, 337)], fill=(214, 220, 218))
    for low, high, color in [
        (0, 0.20, (199, 107, 67)),
        (0.20, 0.30, (231, 222, 201)),
        (0.30, 0.54, middle),
        (0.54, 0.73, (244, 211, 196)),
        (0.73, 1, (210, 183, 143)),
    ]:

        def corners(t):
            y = 330 - 160 * t
            half = 18 + 34 * t
            return (119 - half, y), (119 + half, y)

        (a, b), (c, d) = corners(low), corners(high)
        draw.polygon([a, b, d, c], fill=color)
    draw.ellipse((57, 147, 175, 169), fill=(236, 193, 124))
    draw.polygon(
        [(69, 149), (80, 112), (92, 89), (111, 65), (127, 77), (143, 99), (157, 145)],
        fill=(236, 217, 190),
    )
    for x, y, rx, ry, color in [
        (72, 143, 16, 20, (203, 33, 35)),
        (152, 141, 14, 17, (199, 22, 23)),
        (118, 148, 11, 11, (39, 42, 57)),
    ]:
        draw.ellipse((x - rx, y - ry, x + rx, y + ry), fill=color)
    return image


class TestDessertMeasure(unittest.TestCase):
    def test_observed_layer_palette_and_geometry(self):
        blue = measure_dessert(parfait((70, 117, 197)))
        green = measure_dessert(parfait((130, 181, 80)))
        for m in [blue, green]:
            self.assertIsNotNone(m)
            self.assertGreater(m["foot"][1], m["rim"][1])
            self.assertGreater(m["rim"][2], m["foot"][2])
            self.assertEqual({b["kind"] for b in m["berries"]}, {"red", "dark"})
            self.assertEqual(m["layers"][0]["bottom"], 0)
            self.assertEqual(m["layers"][-1]["top"], 1)
        self.assertTrue(
            any(b[2] > b[0] + 50 for b in [q["rgb"] for q in blue["layers"]])
        )
        self.assertTrue(
            any(b[1] > b[2] + 45 for b in [q["rgb"] for q in green["layers"]])
        )

    def test_rejects_unrelated_inputs(self):
        blank = Image.new("RGB", (300, 350), "ivory")
        self.assertIsNone(measure_dessert(blank))
        draw = ImageDraw.Draw(blank)
        draw.rectangle((70, 110, 230, 141), fill="tomato")
        draw.ellipse((80, 131, 106, 157), fill="black")
        draw.ellipse((191, 131, 217, 157), fill="black")
        self.assertIsNone(measure_dessert(blank))

    def test_scale_and_framing(self):
        subject = parfait((128, 187, 78))
        full = Image.new("RGB", (400, 440), (45, 47, 57))
        full.paste(subject, (60, 22))
        m = measure_dessert(full.resize((500, 550)))
        self.assertIsNotNone(m)
        self.assertAlmostEqual(m["rim"][0], (119 + 60) / 400, delta=0.035)
        self.assertEqual(m["image_size"], [500, 550])

    def test_ambiguous_side_by_side(self):
        subject = parfait((128, 187, 78))
        image = Image.new("RGB", (520, 380), (45, 47, 57))
        image.paste(subject, (0, 0))
        image.paste(subject, (270, 0))
        self.assertIsNone(measure_dessert(image))


if __name__ == "__main__":
    unittest.main()
