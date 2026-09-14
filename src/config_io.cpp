#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "config_io.hpp"


RaceConfig load_pace_model(
    const std::filesystem::path& path,
    RaceConfig config
) {
    std::ifstream input(path);

    if (!input) {
        throw std::runtime_error(
            "Cannot open pace model: " + path.string()
        );
    }

    std::unordered_map<std::string, double> values;
    std::string key;

    while (input >> key) {
        double value{};

        if (!(input >> value) || !std::isfinite(value)) {
            throw std::runtime_error(
                "Invalid value for pace parameter: " + key
            );
        }

        if (!values.emplace(key, value).second) {
            throw std::runtime_error(
                "Duplicate pace parameter: " + key
            );
        }
    }

    if (!input.eof()) {
        throw std::runtime_error("Failed to read pace model");
    }

    const auto required = [&values](const std::string& name) {
        const auto it = values.find(name);

        if (it == values.end()) {
            throw std::runtime_error(
                "Missing pace parameter: " + name
            );
        }

        return it->second;
    };

    config.soft.base_lap_time_s =
        required("soft_base_s");

    config.soft.degradation_s_per_lap =
        required("soft_degradation_s_per_lap");

    config.hard.base_lap_time_s =
        required("hard_base_s");

    config.hard.degradation_s_per_lap =
        required("hard_degradation_s_per_lap");

    config.fuel_gain_s_per_lap =
        required("race_lap_gain_s_per_lap");

    if (values.size() != 5) {
        throw std::runtime_error("Unknown pace parameter");
    }

    if (
        config.soft.base_lap_time_s <= 0.0 ||
        config.hard.base_lap_time_s <= 0.0
    ) {
        throw std::runtime_error("Base lap times must be positive");
    }

    return config;
}