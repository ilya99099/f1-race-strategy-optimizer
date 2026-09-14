#pragma once

#include <filesystem>

#include "race_types.hpp"

RaceConfig load_pace_model(
    const std::filesystem::path& path,
    RaceConfig config
);