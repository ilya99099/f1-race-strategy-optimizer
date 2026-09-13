#pragma once

#include <filesystem>

#include "race_types.hpp"

void write_simulation_csv(
    const std::filesystem::path& path,
    const SimulationResult& result
);