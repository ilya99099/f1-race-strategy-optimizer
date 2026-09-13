#include "race_model.hpp"

#include <stdexcept>

std::string_view compound_name(Compound compound) {
    switch (compound) {
        case Compound::Soft:
            return "SOFT";
        case Compound::Hard:
            return "HARD";
    }

    throw std::invalid_argument("Unknown tyre compound");
}

const TyreModel& tyre_model(
    const RaceConfig& config,
    Compound compound
) {
    switch (compound) {
        case Compound::Soft:
            return config.soft;
        case Compound::Hard:
            return config.hard;
    }

    throw std::invalid_argument("Unknown tyre compound");
}

double predict_lap_time(
    const RaceConfig& config,
    Compound compound,
    int tyre_age_at_start,
    int race_lap
) {
    const auto& model = tyre_model(config, compound);

    return model.base_lap_time_s
        + model.degradation_s_per_lap * tyre_age_at_start
        - config.fuel_gain_s_per_lap * (race_lap - 1);
}

SimulationResult simulate(
    const RaceConfig& config,
    const std::vector<Stint>& strategy
) {
    if (config.total_laps <= 0 || strategy.empty()) {
        throw std::invalid_argument("Race and strategy must be non-empty");
    }

    long long planned_laps = 0;

    for (const auto& stint : strategy) {
        if (stint.laps <= 0 || stint.initial_tyre_age < 0) {
            throw std::invalid_argument("Invalid stint length or tyre age");
        }

        planned_laps += stint.laps;
    }

    if (planned_laps != config.total_laps) {
        throw std::invalid_argument(
            "Strategy must cover exactly the full race"
        );
    }

    SimulationResult result;
    result.laps.reserve(config.total_laps);

    int race_lap = 1;
    int stint_number = 0;

    for (const auto& stint : strategy) {
        ++stint_number;

        for (int lap_in_stint = 0; lap_in_stint < stint.laps; ++lap_in_stint) {
            const int tyre_age =
                stint.initial_tyre_age + lap_in_stint;

            const double driving_time = predict_lap_time(
                config, stint.compound, tyre_age, race_lap
            );

            // Account for the stop at the start of each new stint.
            // This is a bookkeeping convention, not a pit-lap model.
            const bool starts_after_stop =
                stint_number > 1 && lap_in_stint == 0;

            const double pit_loss =
                starts_after_stop ? config.pit_loss_s : 0.0;

            result.total_time_s += driving_time + pit_loss;

            result.laps.push_back({
                race_lap,
                stint_number,
                stint.compound,
                tyre_age,
                driving_time,
                pit_loss,
                result.total_time_s
            });

            ++race_lap;
        }
    }

    return result;
}