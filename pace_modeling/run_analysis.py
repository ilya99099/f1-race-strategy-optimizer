"""Joint linear and Gaussian state-space pace models; retrospective case study."""
from pathlib import Path
import argparse
import io
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def load(path):
    text = Path(path).read_text(encoding="utf-8")
    text = "\n".join(line for line in text.splitlines()
                     if not line.strip().startswith("```"))
    data = pd.read_csv(io.StringIO(text))
    required = ["session_key", "driver_number", "lap_number", "stint_number",
                "tyre_age", "lap_in_stint", "lap_duration", "interval_min"]
    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    valid = data.is_valid_for_pace.astype(str).str.strip().str.lower().eq("true")
    data["compound"] = data.compound.astype(str).str.strip().str.upper()
    data = data.loc[valid & data.session_key.eq(9472) & data.driver_number.eq(4)
                    & data.interval_min.ge(2) & data.compound.isin(["SOFT", "HARD"])]
    data = data.dropna(subset=required).sort_values("lap_number").reset_index(drop=True)
    if data.lap_number.duplicated().any() or (data.tyre_age < 0).any():
        raise ValueError("Duplicate laps or negative tyre ages")
    return data


def design(data):
    soft = data.compound.eq("SOFT").to_numpy(dtype=float)
    age = data.tyre_age.to_numpy(dtype=float)
    return np.column_stack([soft, 1-soft, soft*age, (1-soft)*age,
                            -(data.lap_number.to_numpy(dtype=float)-1)])


def fit_linear(data):
    x = design(data)
    if np.linalg.matrix_rank(x) != 5:
        raise ValueError("Five parameters are not identifiable in this training prefix")
    return np.linalg.lstsq(x, data.lap_duration.to_numpy(), rcond=None)[0]


def filtering(data, beta, q, r, update_until=np.inf):
    """Residual random walk reset at each stint. q,r are variances in s^2.

    Conditional intervals exclude parameter uncertainty. Missing laps increase
    process variance. Initial used-tyre age belongs in the mean, not elapsed time.
    """
    means = design(data) @ beta
    residual, variance, previous_lap, previous_stint = 0., 0., None, None
    output, nll = [], 0.
    for i, row in enumerate(data.itertuples(index=False)):
        if row.stint_number != previous_stint:
            residual = 0.
            variance = q * (row.lap_in_stint - 1)
        else:
            variance += q * (row.lap_number - previous_lap)
        prediction = means[i] + residual
        predictive_variance = variance + r
        output.append((prediction, predictive_variance))
        if row.lap_number <= update_until:
            error = row.lap_duration - prediction
            nll += .5*(np.log(2*np.pi*predictive_variance)
                       + error**2/predictive_variance)
            gain = variance/predictive_variance
            residual += gain*error
            variance *= 1-gain
        previous_lap, previous_stint = row.lap_number, row.stint_number
    return np.asarray(output), float(nll)


def fit_state(data):
    beta = fit_linear(data)
    # Numerical variance bounds: SD from 0.00001 to 10 seconds.
    bounds = [(None, None)]*5 + [(np.log(1e-10), np.log(100))]*2
    candidates = []
    for q in [1e-8, .001, .03]:
        initial = np.r_[beta, np.log(q), np.log(.02)]
        result = minimize(lambda p: filtering(data, p[:5], *np.exp(p[5:]))[1],
                          initial, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter":1500, "ftol":1e-11})
        if result.success and np.isfinite(result.fun):
            candidates.append(result)
    if not candidates:
        raise RuntimeError("State-space likelihood optimization did not converge")
    best = min(candidates, key=lambda result: result.fun)
    return best.x[:5], *np.exp(best.x[5:]), float(best.fun)


def parameters(beta):
    return dict(zip(["soft_base_s", "hard_base_s", "soft_degradation_s_per_lap",
                     "hard_degradation_s_per_lap", "race_lap_gain_s_per_lap"],
                    map(float, beta)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent/"results")
    parser.add_argument("--cutoff", type=int, default=40)
    args = parser.parse_args()
    data = load(args.input)
    train = data.loc[data.lap_number <= args.cutoff]
    test = data.loc[data.lap_number > args.cutoff]
    if len(test) == 0:
        raise ValueError("No test laps after cutoff")
    print(f"Selected laps: {len(data)}; train: {len(train)}; test: {len(test)}")
    # Both models are fitted ONCE on the prefix; no test-label hyperparameter tuning.
    linear = fit_linear(train)
    beta, q, r, nll = fit_state(train)
    fixed = filtering(data, beta, q, r, args.cutoff)[0]
    online = filtering(data, beta, q, r)[0]  # Predict first, then assimilate that lap.
    rows = test.copy()
    idx = rows.index.to_numpy()
    rows["linear_prediction_s"] = design(test) @ linear
    rows["state_fixed_prediction_s"] = fixed[idx, 0]
    rows["state_online_prediction_s"] = online[idx, 0]
    rows["state_online_lower_95_s"] = online[idx, 0] - 1.96*np.sqrt(online[idx, 1])
    rows["state_online_upper_95_s"] = online[idx, 0] + 1.96*np.sqrt(online[idx, 1])
    means = train.groupby("stint_number").lap_duration.mean()
    rows["baseline_fixed_prediction_s"] = rows.stint_number.map(means)
    last_values = train.groupby("stint_number").lap_duration.last().to_dict()
    online_baseline = []
    for row in test.itertuples():
        online_baseline.append(last_values.get(row.stint_number, np.nan))
        last_values[row.stint_number] = row.lap_duration
    rows["baseline_online_prediction_s"] = online_baseline
    scores = {}
    for col in rows.columns:
        if col.endswith("prediction_s"):
            error = (rows[col]-rows.lap_duration).dropna()
            scores[col] = {"n":len(error), "mae_s":float(error.abs().mean())}
            print(f"{col}: n={len(error)}, MAE={error.abs().mean():.4f} s")
    full_beta, full_q, full_r, full_nll = fit_state(data)
    report = {"cutoff":args.cutoff, "train_n":len(train), "test_n":len(test),
              "training_linear":parameters(linear),
              "training_state":{**parameters(beta), "q":q, "r":r, "nll":nll},
              "full_data_linear":parameters(fit_linear(data)),
              "full_data_state":{**parameters(full_beta), "q":full_q,
                                 "r":full_r, "nll":full_nll}, "scores":scores}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"parameters.json").write_text(json.dumps(report, indent=2)+"\n")
    rows.to_csv(args.output/"validation_predictions.csv", index=False)
    data.to_csv(args.output/"selected_laps.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(rows.lap_number, rows.lap_duration, "ko-", label="Observed")
    for col in ["linear_prediction_s", "state_fixed_prediction_s", "state_online_prediction_s"]:
        ax.plot(rows.lap_number, rows[col], ".--", label=col.replace("_prediction_s", ""))
    ax.set(xlabel="Race lap", ylabel="Lap time (s)", title=f"Norris: held-out selected laps; training through lap {args.cutoff}")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(args.output/"validation.png", dpi=160)
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
