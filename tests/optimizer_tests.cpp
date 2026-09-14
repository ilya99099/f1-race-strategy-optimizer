#include "race_types.hpp"
#include "race_model.hpp"
#include "brute_force.hpp"
#include "dynamic_programming.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

void verify_optimizer(const RaceConfig& base_config) {
    constexpr double tolerance = 1e-6;
    int checks = 0;

    for (int total_laps : {5, 8, 12}) {
        for (double pit_loss : {0.0, 5.0, 22.0}) {
            for (double soft_degradation : {0.0, 0.06, 0.18}) {
                RaceConfig config = base_config;
                config.total_laps = total_laps;
                config.pit_loss_s = pit_loss;
                config.soft.degradation_s_per_lap = soft_degradation;

                const auto brute_results = enumerate_strategies(config);

                for (int stops : {1, 2}) {
                    const auto dp_result = optimize_dp(config, stops);

                    const auto brute_best = std::find_if(
                        brute_results.begin(),
                        brute_results.end(),
                        [stops](const StrategyResult& result) {
                            return result.strategy.size()
                                == static_cast<std::size_t>(stops + 1);
                        }
                    );

                    if (brute_best == brute_results.end()) {
                        throw std::runtime_error(
                            "Missing brute-force reference"
                        );
                    }

                    int used_compounds = 0;

                    for (const auto& stint : dp_result.strategy) {
                        used_compounds |=
                            stint.compound == Compound::Soft ? 1 : 2;

                        if (stint.initial_tyre_age != 0) {
                            throw std::runtime_error(
                                "Expected fresh tyres in every stint"
                            );
                        }
                    }

                    if (
                        used_compounds != 3 ||
                        dp_result.strategy.size()
                            != static_cast<std::size_t>(stops + 1)
                    ) {
                        throw std::runtime_error(
                            "DP returned an invalid strategy"
                        );
                    }

                    // simulate() also checks stint lengths and total laps.
                    const double simulated_time =
                        simulate(config, dp_result.strategy).total_time_s;

                    if (
                        !std::isfinite(dp_result.total_time_s) ||
                        !std::isfinite(simulated_time) ||
                        std::abs(
                            dp_result.total_time_s
                            - brute_best->total_time_s
                        ) > tolerance ||
                        std::abs(
                            dp_result.total_time_s - simulated_time
                        ) > tolerance
                    ) {
                        throw std::runtime_error(
                            "Optimizer mismatch: laps="
                            + std::to_string(total_laps)
                            + ", pit_loss=" + std::to_string(pit_loss)
                            + ", soft_degradation="
                            + std::to_string(soft_degradation)
                            + ", stops=" + std::to_string(stops)
                        );
                    }

                    ++checks;
                }
            }
        }
    }

    std::cout << "\nOptimizer checks passed: " << checks << '\n';
}

void verify_inventory_optimizer(const RaceConfig& config) {
    const std::vector<TyreSet> inventory{
            {0, Compound::Soft, 2},
            {1, Compound::Hard, 0},
            {2, Compound::Hard, 0}
    };

    const auto brute_results =
        enumerate_inventory_strategies(config, inventory, 2);

    constexpr double tolerance = 1e-6;

    for (int stops : {1, 2}) {
        const auto dp_result =
            optimize_inventory_dp(config, inventory, stops);

        const auto brute_best = std::find_if(
            brute_results.begin(),
            brute_results.end(),
            [stops](const StrategyResult& result) {
                return result.strategy.size()
                    == static_cast<std::size_t>(stops + 1);
            }
        );

        if (brute_best == brute_results.end()) {
            throw std::runtime_error("Missing inventory reference");
        }

        validate_inventory_strategy(dp_result.strategy, inventory);

        const double simulated_time =
            simulate(config, dp_result.strategy).total_time_s;

        if (
            dp_result.strategy.size()
                != static_cast<std::size_t>(stops + 1) ||
            !std::isfinite(dp_result.total_time_s) ||
            !std::isfinite(brute_best->total_time_s) ||
            !std::isfinite(simulated_time) ||
            std::abs(
                dp_result.total_time_s - brute_best->total_time_s
            ) > tolerance ||
            std::abs(
                dp_result.total_time_s - simulated_time
            ) > tolerance
        ) {
            throw std::runtime_error(
                "Inventory optimizer mismatch for "
                + std::to_string(stops) + " stop(s)"
            );
        }
    }

    std::cout << "Inventory optimizer checks passed: 2\n";
}

int main() {
    try {
        const RaceConfig config{
            .total_laps = 57,
            .pit_loss_s = 22.0,
            .fuel_gain_s_per_lap = 0.03,
            .soft = {
                .base_lap_time_s = 96.0,
                .degradation_s_per_lap = 0.12
            },
            .hard = {
                .base_lap_time_s = 96.6,
                .degradation_s_per_lap = 0.06
            }
        };

        verify_optimizer(config);
        verify_inventory_optimizer(config);

        std::cout << "All optimizer checks passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "TEST FAILURE: " << error.what() << '\n';
        return 1;
    }
}