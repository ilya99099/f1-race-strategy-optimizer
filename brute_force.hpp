#pragma once

#include "race_types.hpp"

std::vector<StrategyResult> enumerate_strategies(
    const RaceConfig& config
);

void validate_inventory_strategy(
    const std::vector<Stint>& strategy,
    const std::vector<TyreSet>& inventory
);

std::vector<StrategyResult> enumerate_inventory_strategies(
    const RaceConfig& config,
    const std::vector<TyreSet>& inventory,
    int max_stops
);