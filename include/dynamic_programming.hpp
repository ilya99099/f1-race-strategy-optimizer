#pragma once

#include "race_types.hpp"

StrategyResult optimize_dp(
    const RaceConfig& config,
    int stop_count
);

StrategyResult optimize_inventory_dp(
    const RaceConfig& config,
    const std::vector<TyreSet>& inventory,
    int stop_count
);