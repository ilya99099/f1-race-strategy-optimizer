import unittest
import numpy as np
import pandas as pd
from run_analysis import filtering, design, fit_linear


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            "compound": ["SOFT", "SOFT", "HARD"],
            "tyre_age": [3, 6, 1], "lap_number": [1, 4, 8],
            "lap_in_stint": [1, 4, 2], "stint_number": [1, 1, 2],
            "lap_duration": [100., 102., 99.]})
        self.beta = np.array([96., 97., .1, .05, .03])

    def test_zero_process_noise_is_linear(self):
        predictions, nll = filtering(self.data, self.beta, 0., .04)
        expected = design(self.data) @ self.beta
        np.testing.assert_allclose(predictions[:, 0], expected)
        expected_nll = .5*np.sum(np.log(2*np.pi*.04)
                                + (self.data.lap_duration-expected)**2/.04)
        self.assertAlmostEqual(nll, expected_nll)

    def test_missing_laps_and_reset(self):
        predictions, _ = filtering(self.data, self.beta, .1, .04, update_until=0)
        np.testing.assert_allclose(predictions[:, 1], [.04, .34, .14])

    def test_current_and_future_values_cannot_change_prediction(self):
        first, _ = filtering(self.data, self.beta, .1, .04)
        changed = self.data.copy()
        changed.loc[1:, "lap_duration"] += 100
        second, _ = filtering(changed, self.beta, .1, .04)
        np.testing.assert_allclose(first[:2], second[:2])

    def test_fixed_forecast_ignores_test_values(self):
        first, _ = filtering(self.data, self.beta, .1, .04, update_until=1)
        changed = self.data.copy()
        changed.loc[1:, "lap_duration"] += 100
        second, _ = filtering(changed, self.beta, .1, .04, update_until=1)
        np.testing.assert_allclose(first, second)

    def test_rank_deficient_training_is_rejected(self):
        with self.assertRaises(ValueError):
            fit_linear(self.data)


if __name__ == "__main__":
    unittest.main()
