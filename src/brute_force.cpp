#include "brute_force.hpp"
#include "race_model.hpp"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <utility>

std::vector<StrategyResult> enumerate_strategies(
    const RaceConfig& config
) {
    std::vector<StrategyResult> results;

    // For each set of stint lengths, enumerate compound assignments.
    auto evaluate_lengths = [&](const std::vector<int>& lengths) {
        const int stint_count = static_cast<int>(lengths.size());
        const int assignment_count = 1 << stint_count;

        for (int mask = 0; mask < assignment_count; ++mask) {
            // Bit 0 means SOFT, bit 1 means HARD.
            // Reject assignments using only one compound.
            if (mask == 0 || mask == assignment_count - 1) {
                continue;
            }

            std::vector<Stint> strategy;
            strategy.reserve(stint_count);

            for (int i = 0; i < stint_count; ++i) {
                const Compound compound = (mask & (1 << i))
                    ? Compound::Hard
                    : Compound::Soft;

                strategy.push_back({
                    compound,
                    lengths[i],
                    0
                });
            }

            const double total_time =
                simulate(config, strategy).total_time_s;

            results.push_back({
                std::move(strategy),
                total_time
            });
        }
    };

    // One stop: two non-empty stints.
    for (int first = 1; first < config.total_laps; ++first) {
        evaluate_lengths({
            first,
            config.total_laps - first
        });
    }

    // Two stops: three non-empty stints.
    for (int first = 1; first < config.total_laps - 1; ++first) {
        for (
            int second = 1;
            second < config.total_laps - first;
            ++second
        ) {
            evaluate_lengths({
                first,
                second,
                config.total_laps - first - second
            });
        }
    }

    std::stable_sort(
        results.begin(),
        results.end(),
        [](const StrategyResult& left, const StrategyResult& right) {
            return left.total_time_s < right.total_time_s;
        }
    );

    return results;
}

void validate_inventory_strategy(
    const std::vector<Stint>& strategy,
    const std::vector<TyreSet>& inventory
) {
    std::vector<bool> used(inventory.size(), false);
    int compound_mask = 0;

    for (const auto& stint : strategy) {
        const auto it = std::find_if(
            inventory.begin(),
            inventory.end(),
            [&](const TyreSet& tyre_set) {
                return tyre_set.id == stint.tyre_set_id;
            }
        );

        if (it == inventory.end()) {
            throw std::invalid_argument("Unknown tyre set ID");
        }

        const std::size_t index =
            static_cast<std::size_t>(it - inventory.begin());

        if (used[index]) {
            throw std::invalid_argument("Tyre set used more than once");
        }

        if (
            stint.compound != it->compound ||
            stint.initial_tyre_age != it->initial_age
        ) {
            throw std::invalid_argument(
                "Stint does not match the selected tyre set"
            );
        }

        used[index] = true;

        compound_mask |=
            stint.compound == Compound::Soft ? 1 : 2;
    }

    if (compound_mask != 3) {
        throw std::invalid_argument(
            "Strategy must use both SOFT and HARD"
        );
    }
}

std::vector<StrategyResult> enumerate_inventory_strategies(
    const RaceConfig& config,
    const std::vector<TyreSet>& inventory,
    int max_stops
) {
    if (config.total_laps <= 0 || max_stops < 1) {
        throw std::invalid_argument("Invalid race or stop limit");
    }

    for (std::size_t i = 0; i < inventory.size(); ++i) {
        if (inventory[i].id < 0 || inventory[i].initial_age < 0) {
            throw std::invalid_argument("Invalid tyre set");
        }

        // Also validate the compound.
        tyre_model(config, inventory[i].compound);

        for (std::size_t j = 0; j < i; ++j) {
            if (inventory[i].id == inventory[j].id) {
                throw std::invalid_argument("Duplicate tyre set ID");
            }
        }
    }

    std::vector<StrategyResult> results;
    std::vector<Stint> current;
    std::vector<bool> used(inventory.size(), false);

    const std::size_t max_stints =
        static_cast<std::size_t>(max_stops) + 1;

    std::function<void(int, int)> search =
        [&](int completed_laps, int compound_mask) {
            if (completed_laps == config.total_laps) {
                // At least one stop and both compounds are required.
                if (current.size() < 2 || compound_mask != 3) {
                    return;
                }

                validate_inventory_strategy(current, inventory);

                const double total_time =
                    simulate(config, current).total_time_s;

                results.push_back({current, total_time});
                return;
            }

            if (current.size() >= max_stints) {
                return;
            }

            const int remaining_laps =
                config.total_laps - completed_laps;

            for (std::size_t i = 0; i < inventory.size(); ++i) {
                if (used[i]) {
                    continue;
                }

                const auto& tyre_set = inventory[i];
                const int compound_bit =
                    tyre_set.compound == Compound::Soft ? 1 : 2;

                used[i] = true;

                for (int length = 1; length <= remaining_laps; ++length) {
                    current.push_back({
                        tyre_set.compound,
                        length,
                        tyre_set.initial_age,
                        tyre_set.id
                    });

                    search(
                        completed_laps + length,
                        compound_mask | compound_bit
                    );

                    current.pop_back();
                }

                used[i] = false;
            }
        };

    search(0, 0);

    std::stable_sort(
        results.begin(),
        results.end(),
        [](const StrategyResult& left, const StrategyResult& right) {
            return left.total_time_s < right.total_time_s;
        }
    );

    return results;
}