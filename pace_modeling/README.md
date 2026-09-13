# Pace model experiment

Copy this `pace_modeling` folder into your project root, beside `pace_laps.csv`.
Run from the project root:

```bash
.venv/bin/python -m pip install -r pace_modeling/requirements.txt
.venv/bin/python pace_modeling/run_analysis.py --input pace_laps.csv
.venv/bin/python -m unittest discover -s pace_modeling -p 'test_*.py'
```

To reproduce the included results without the full source dataset:

```bash
.venv/bin/python pace_modeling/run_analysis.py --input pace_modeling/results/selected_laps.csv
```

## Models

Both models use the mean `base[compound] + degradation[compound] * tyre_age
- race_lap_gain * (lap_number - 1)`.
The five mean parameters are estimated jointly, including race-lap gain.
The linear model uses ordinary least squares.

The Gaussian state-space model adds a latent residual random walk within each
stint. Its variance grows by `q * elapsed_laps`; measurement noise has variance
`r`. At each new stint, the residual resets to zero at the start of that stint.
Unobserved laps since that start contribute process variance. Used-tyre age is
included in the mean; there is no additional uncertainty for prior tyre usage.
This reset assumption is deliberately simple and should be revisited with more data.
Parameters are fitted with maximum likelihood, using a Kalman filter and three
optimization initializations. Variances have numerical bounds [1e-10, 100] s^2.
Mean coefficients are unconstrained. Convergence does not prove global optimality.
A very small q means the estimated model has effectively reduced to regression.

This is a Gaussian, maximum-likelihood adaptation inspired by
[Cappello and Hoegh (2025)](https://arxiv.org/html/2512.00640v1),
not a reproduction of their Bayesian Stan/MCMC analysis.
Optimization uses [SciPy minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).

## Data and validation

Session 9472, driver 4, valid pace laps, minimum observed gap >= 2 seconds.
`tyre_age` is the age at the start of the current lap. In this source CSV,
`tyre_age_at_start` is the age at the start of the STINT. Do not substitute it.
The gap filter is retrospective: results are conditional on the selected laps,
and do not describe performance on every race lap or guarantee clean air.

Default split: train through race lap 40, evaluate selected laps after 40.
All parameter estimation for validation uses only the training prefix.
The cutoff is a case-study choice, not a tuned or independent final benchmark.
Five joint mean parameters cannot be identified from only the first two Norris
stints: within each compound, tyre age and race lap are collinear. Repeated HARD
stints permit a full-rank fit, but uncertainty remains large. This experiment
therefore evaluates late third-stint HARD predictions only, not new-stint or
SOFT generalization and not a pre-race strategy model.

- `linear`: fixed parameters; no use of test observations.
- `state_fixed`: forecast the entire test suffix from the cutoff state.
- `state_online`: same frozen parameters; predict a lap, then update the state
  using that lap's observed time for subsequent predictions.
- `baseline_fixed`: mean training time within the same stint.
- `baseline_online`: last observed selected lap within the same stint.

Fixed and online forecasts use different information sets; compare each with
its corresponding baseline. Intervals include process and measurement noise,
but omit parameter uncertainty. Full-data fits are exported separately and
never used to calculate validation errors.

`race_lap_gain` is an empirical time trend that can absorb fuel, track evolution,
driving choices and other changes. It is not a measured fuel effect. More complex
models do not resolve this confounding automatically. There are only 39 selected
laps and one race; do not claim a universally better model from this experiment.

## Output

`results/parameters.json`: training and full-data coefficients, noise variances,
likelihood and MAEs. `validation_predictions.csv`: individual held-out predictions.
`selected_laps.csv`: reproducible selected input. `validation.png`: comparison plot.

C++ integration is a separate step. A stateful online correction cannot be
transferred to hypothetical strategies simply by copying five coefficients.

## Included run (12 September 2026)

28 training observations and 11 held-out observations (race laps 41-51).

| Forecast | MAE (seconds) |
| --- | ---: |
| Joint linear, fixed | 0.1944 |
| State-space, fixed | 0.1944 |
| State-space, online | 0.1944 |
| Training stint mean, fixed baseline | 0.3800 |
| Previous observed lap, online baseline | 0.1193 |

Training state-space q was approximately 1.28e-9 s^2 per lap, making the
latent correction negligible. The more complex model did not improve this
validation result. The full-data fit has a larger q (0.00348); this uses the
test suffix too, so it cannot replace the training fit in the comparison.
The jointly estimated race-lap gain is 0.07664 on training and 0.07091 for the
full-data linear fit. These values must not be interpreted as measured fuel gain.
Five unit tests passed (linear limit, missing laps/reset, prediction causality,
fixed forecast independence from test values, and rank-deficient training).
