import unittest

from grader.agreement import cohen_kappa, fleiss_kappa, krippendorff_alpha, percent_agreement, weighted_kappa

# Krippendorff's own worked example: 4 coders, 12 units, values 1-5, None = missing.
# Published results: nominal alpha = 0.743, interval alpha = 0.849.
KRIPPENDORFF_EXAMPLE = [
    [1, 1, None, 1],
    [2, 2, 3, 2],
    [3, 3, 3, 3],
    [3, 3, 3, 3],
    [2, 2, 2, 2],
    [1, 2, 3, 4],
    [4, 4, 4, 4],
    [1, 1, 2, 1],
    [2, 2, 2, 2],
    [None, 5, 5, 5],
    [None, None, 1, 1],
    [None, 3, None, None],
]

# Fleiss (1971) as reproduced in most references: 10 items, 14 raters, 5 categories. kappa = 0.210.
FLEISS_EXAMPLE = [
    [0, 0, 0, 0, 14],
    [0, 2, 6, 4, 2],
    [0, 0, 3, 5, 6],
    [0, 3, 9, 2, 0],
    [2, 2, 8, 1, 1],
    [7, 7, 0, 0, 0],
    [3, 2, 6, 3, 0],
    [2, 5, 3, 2, 2],
    [6, 5, 2, 1, 0],
    [0, 2, 2, 3, 7],
]


class KrippendorffTests(unittest.TestCase):
    def test_published_nominal(self):
        self.assertAlmostEqual(krippendorff_alpha(KRIPPENDORFF_EXAMPLE, "nominal"), 0.743, places=3)

    def test_published_interval(self):
        self.assertAlmostEqual(krippendorff_alpha(KRIPPENDORFF_EXAMPLE, "interval"), 0.849, places=3)

    def test_perfect_agreement(self):
        self.assertEqual(krippendorff_alpha([[1, 1, 1], [2, 2, None], [5, 5, 5]]), 1.0)

    def test_units_with_one_value_are_ignored(self):
        with_singleton = KRIPPENDORFF_EXAMPLE + [[4, None, None, None]]
        self.assertAlmostEqual(
            krippendorff_alpha(with_singleton, "nominal"), krippendorff_alpha(KRIPPENDORFF_EXAMPLE, "nominal")
        )

    def test_systematic_disagreement_is_negative(self):
        self.assertLess(krippendorff_alpha([[1, 5], [5, 1], [1, 5], [5, 1]]), 0)


class KappaTests(unittest.TestCase):
    def test_cohen_textbook(self):
        # 50 items: yes/yes 20, yes/no 5, no/yes 10, no/no 15 -> po 0.70, pe 0.50, kappa 0.40.
        a = ["y"] * 25 + ["n"] * 25
        b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
        self.assertAlmostEqual(percent_agreement(a, b), 0.70)
        self.assertAlmostEqual(cohen_kappa(a, b), 0.40)

    def test_weighted_equals_cohen_for_two_categories(self):
        a = [1, 1, 2, 2, 1, 2, 1, 1]
        b = [1, 2, 2, 2, 1, 1, 1, 2]
        self.assertAlmostEqual(weighted_kappa(a, b, [1, 2]), cohen_kappa(a, b))

    def test_weighted_credits_near_misses(self):
        a = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        near = [2, 3, 4, 5, 4, 1, 2, 3, 4, 5]  # off by one where wrong
        self.assertGreater(weighted_kappa(a, near, [1, 2, 3, 4, 5]), cohen_kappa(a, near))

    def test_weighted_perfect(self):
        self.assertAlmostEqual(weighted_kappa([1, 3, 5, 2], [1, 3, 5, 2], [1, 2, 3, 4, 5]), 1.0)

    def test_fleiss_published(self):
        self.assertAlmostEqual(fleiss_kappa(FLEISS_EXAMPLE), 0.210, places=3)

    def test_fleiss_rejects_ragged_input(self):
        with self.assertRaises(ValueError):
            fleiss_kappa([[2, 1], [1, 1]])


if __name__ == "__main__":
    unittest.main()
