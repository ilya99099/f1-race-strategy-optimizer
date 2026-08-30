# F1 Race Strategy Optimizer

An explainable race-strategy analysis project that combines historical Formula 1 data,
an interpretable pace model, a deterministic C++20 simulator, and dynamic programming.

The first case study is the 2024 Bahrain Grand Prix. The model will be fitted on cleaned
laps from the field and used to compare Lando Norris's actual strategy with strategies
that minimise predicted full-race time under explicit assumptions.

## Current milestone

The repository currently contains only the first data-ingestion step: resolving an
OpenF1 race session from descriptive fields instead of hard-coding its numeric key.

From the repository root, run:

```bash
python3 python/fetch_session.py \
  --year 2024 \
  --country Bahrain \
  --session-name Race
```

The command writes the selected session metadata to:

```text
data/raw/bahrain_2024/session.json
```

Historical OpenF1 data is available without authentication. OpenF1 is an unofficial,
community-operated data source: <https://openf1.org/docs/>.

## Intended MVP

1. Download and validate session, driver, lap, stint, pit, and race-control data.
2. Remove laps affected by starts, pit transitions, neutralisations, and bad timing data.
3. Fit and validate an interpretable tyre/pace model.
4. Precompute stint costs in C++20.
5. Find the best legal one-, two-, and three-stop strategies using dynamic programming.
6. Cross-check the optimiser against brute force on small synthetic races.
7. Compare model-optimal strategies with the actual race strategy and report limitations.

The optimiser will produce the best strategy **inside the fitted model**. It will not
claim to recover the objectively best real-world strategy.
