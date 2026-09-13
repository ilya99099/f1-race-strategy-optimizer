from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"


def load_result(filename):
    data = pd.read_csv(RESULTS_DIR / filename)

    expected_laps = np.arange(1, 58)

    if not np.array_equal(data["lap"].to_numpy(), expected_laps):
        raise ValueError(f"{filename}: expected consecutive laps 1–57")

    numeric = data[
        ["driving_time_s", "pit_loss_s", "cumulative_time_s"]
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        raise ValueError(f"{filename}: non-finite lap times")

    calculated_total = (
            data["driving_time_s"] + data["pit_loss_s"]
    ).cumsum()

    if not np.allclose(
            calculated_total,
            data["cumulative_time_s"],
            rtol=0,
            atol=1e-4,
    ):
        raise ValueError(f"{filename}: inconsistent cumulative times")

    return data


reference = load_result("reference_laps.csv")
optimized = load_result("optimized_laps.csv")

# Positive means the optimized strategy has used less time.
advantage = (
        reference["cumulative_time_s"]
        - optimized["cumulative_time_s"]
)

fig, (pace_ax, gap_ax) = plt.subplots(
    2, 1,
    figsize=(11, 8),
    sharex=True,
    constrained_layout=True,
)

for data, label, color in [
    (reference, "Reference", "tab:blue"),
    (optimized, "Optimized", "tab:orange"),
]:
    pace_ax.plot(
        data["lap"],
        data["driving_time_s"],
        label=label,
        color=color,
    )

    # The simulator charges pit loss on the first lap of the new stint.
    # Therefore the preceding lap is the pit-after lap.
    pit_after_laps = (
            data.loc[data["pit_loss_s"] > 0, "lap"] - 1
    )

    for pit_lap in pit_after_laps:
        pace_ax.axvline(
            pit_lap + 0.5,
            color=color,
            linestyle=":",
            alpha=0.5,
            )

pace_ax.set_title("Predicted driving time — pit loss excluded")
pace_ax.set_ylabel("Driving time (s)")
pace_ax.legend()
pace_ax.grid(alpha=0.25)

# Add the race start: both strategies have accumulated zero time.
plot_laps = np.r_[0, reference["lap"].to_numpy()]
plot_advantage = np.r_[0.0, advantage.to_numpy()]

gap_ax.plot(plot_laps, plot_advantage, color="tab:green")
gap_ax.axhline(0, color="black", linewidth=0.8)

gap_ax.set_title(
    "Cumulative model advantage — pit loss included\n"
    "Positive = optimized strategy ahead"
)
gap_ax.set_xlabel("Completed race laps")
gap_ax.set_ylabel("Reference − optimized (s)")
gap_ax.grid(alpha=0.25)

final_advantage = float(advantage.iloc[-1])

fig.suptitle(
    "Reference vs optimized strategy: same tyre inventory\n"
    f"Final model advantage: {final_advantage:.3f} s"
)

destination = RESULTS_DIR / "strategy_comparison.png"
fig.savefig(destination, dpi=180)
plt.close(fig)

print(f"Reference model time: {reference['cumulative_time_s'].iloc[-1]:.3f} s")
print(f"Optimized model time: {optimized['cumulative_time_s'].iloc[-1]:.3f} s")
print(f"Model advantage: {final_advantage:.3f} s")
print(f"Saved: {destination}")