#pragma once

#include "race_types.hpp"

#include <string_view>

std::string_view compound_name(Compound compound);

const TyreModel& tyre_model(
    const RaceConfig& config,
    Compound compound
);

double predict_lap_time(
    const RaceConfig& config,
    Compound compound,
    int tyre_age_at_start,
    int race_lap
);

SimulationResult simulate(
    const RaceConfig& config,
    const std::vector<Stint>& strategy
);