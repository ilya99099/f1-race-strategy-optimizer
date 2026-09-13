#pragma once

#include <vector>

enum class Compound {
    Soft,
    Hard
};

struct TyreModel {
    double base_lap_time_s;
    double degradation_s_per_lap;
};

struct RaceConfig {
    int total_laps;
    double pit_loss_s;
    double fuel_gain_s_per_lap;

    TyreModel soft;
    TyreModel hard;
};

struct TyreSet {
    int id;
    Compound compound;
    int initial_age;
};

struct Stint {
    Compound compound;
    int laps{};
    int initial_tyre_age{};

    int tyre_set_id = -1;
};

struct LapResult {
    int race_lap;
    int stint_number;
    Compound compound;
    int tyre_age_at_start;

    double driving_time_s;
    double pit_loss_s;
    double cumulative_time_s;
};

struct SimulationResult {
    double total_time_s = 0.0;
    std::vector<LapResult> laps;
};

struct StrategyResult {
    std::vector<Stint> strategy;
    double total_time_s;
};