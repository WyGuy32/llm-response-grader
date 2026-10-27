import unittest

from grader.ranking import Comparison, bootstrap_win_rate_ci, bradley_terry, elo_scale, win_probability, win_rates


def games(winner: str, loser: str, n: int) -> list[Comparison]:
    return [Comparison(winner, loser, 1.0)] * n


class BradleyTerryTests(unittest.TestCase):
    def test_recovers_known_odds(self):
        # x beats y 3 times out of 4. With no prior, the fit should return exactly 3:1 odds.
        comps = games("x", "y", 30) + games("y", "x", 10)
        s = bradley_terry(comps, prior=0.0)
        self.assertAlmostEqual(win_probability(s, "x", "y"), 0.75, places=6)

    def test_transitive_ranking_through_common_opponent(self):
        # a and c never meet, but a beats b and b beats c.
        comps = games("a", "b", 8) + games("b", "a", 2) + games("b", "c", 8) + games("c", "b", 2)
        s = bradley_terry(comps)
        self.assertGreater(s["a"], s["b"])
        self.assertGreater(s["b"], s["c"])
        self.assertGreater(win_probability(s, "a", "c"), 0.8)

    def test_prior_keeps_undefeated_finite(self):
        s = bradley_terry(games("a", "b", 5))
        self.assertTrue(all(0 < v < float("inf") for v in s.values()))

    def test_ties_are_half_wins(self):
        s = bradley_terry([Comparison("a", "b", 0.5)] * 6, prior=0.0)
        self.assertAlmostEqual(win_probability(s, "a", "b"), 0.5)

    def test_elo_scale(self):
        e = elo_scale({"a": 10.0, "b": 1.0})
        self.assertAlmostEqual(e["a"] - e["b"], 400.0)


class WinRateTests(unittest.TestCase):
    def test_win_rates(self):
        r = win_rates(games("a", "b", 3) + [Comparison("a", "b", 0.5)])
        self.assertAlmostEqual(r["a"], 3.5 / 4)
        self.assertAlmostEqual(r["b"], 0.5 / 4)

    def test_bootstrap_interval_contains_point_estimate(self):
        groups = [games("a", "b", 1) for _ in range(7)] + [games("b", "a", 1) for _ in range(3)]
        ci = bootstrap_win_rate_ci(groups, samples=500)
        lo, hi = ci["a"]
        self.assertLessEqual(lo, 0.7)
        self.assertGreaterEqual(hi, 0.7)
        self.assertLess(hi - lo, 0.7)

    def test_bootstrap_is_reproducible(self):
        groups = [games("a", "b", 1), games("b", "a", 1), games("a", "b", 2)]
        self.assertEqual(bootstrap_win_rate_ci(groups, seed=3), bootstrap_win_rate_ci(groups, seed=3))


if __name__ == "__main__":
    unittest.main()
