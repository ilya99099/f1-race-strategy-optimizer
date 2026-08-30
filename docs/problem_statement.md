# Problem statement

## Goal

For one historical dry Formula 1 race, find the legal tyre and pit-stop strategy that
minimises the target driver's predicted full-race time under an explicit pace model.

The initial case study is:

- event: 2024 Bahrain Grand Prix;
- target driver: Lando Norris (car number 4);
- race distance: 57 laps;
- dry compounds: Soft, Medium, Hard.

## Strategy representation

A strategy is an ordered sequence of non-empty stints. Each stint contains:

- a tyre compound;
- its first race lap;
- its final race lap.

Adjacent stints imply one pit stop, so a strategy with `k` stints has `k - 1` stops.

## Objective

For strategy `S`, minimise:

```text
predicted_total_time(S)
  = sum(predicted_lap_time for every race lap)
  + number_of_pit_stops * estimated_pit_loss
```

Lap time is predicted from cleaned historical observations. Raw lap times cannot be
reused directly for hypothetical strategies because only the actually driven compound,
tyre age, and race state are observed.

## MVP constraints

- Complete all 57 laps exactly once.
- Use at least two different dry compounds.
- Use at most three pit stops (therefore at most four stints).
- Do not create zero-length stints.
- Charge pit loss once between every pair of adjacent stints.

## Initial modelling boundary

The deterministic MVP will not explicitly model traffic, overtaking, safety-car timing,
weather changes, damage, driver errors, or competitors' strategic reactions. Laps visibly
affected by starts, pit transitions, neutralisations, missing timing, or large anomalies
will be removed before model fitting where the data permits.

The output is a model-optimal retrospective strategy, not a causal claim that the real
team made the wrong decision.
