#include "dynamic_programming.hpp"
#include "race_model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "brute_force.hpp"

#include <bit>

struct DpParent {
    int previous_laps = -1;
    int previous_mask = -1;
    Compound compound = Compound::Soft;
    int stint_laps = 0;
};

StrategyResult optimize_dp(
    const RaceConfig& config,
    int stop_count
) {
    const int stint_count = stop_count + 1;
    const int total_laps = config.total_laps;
    const double infinity = std::numeric_limits<double>::infinity();

    if (stop_count < 0 || stint_count > total_laps) {
        throw std::invalid_argument("Invalid stop count");
    }

    const std::array<Compound, 2> compounds{
        Compound::Soft,
        Compound::Hard
    };

    // cost[start][length][compound]:
    // driving time of a fresh-tyre stint after 'start' completed laps.
    std::vector<std::vector<std::array<double, 2>>> cost(
        total_laps + 1,
        std::vector<std::array<double, 2>>(total_laps + 1)
    );

    for (int start = 0; start < total_laps; ++start) {
        for (int c = 0; c < 2; ++c) {
            double accumulated = 0.0;

            for (int length = 1; start + length <= total_laps; ++length) {
                accumulated += predict_lap_time(
                    config,
                    compounds[c],
                    length - 1,
                    start + length
                );

                cost[start][length][c] = accumulated;
            }
        }
    }

    using MaskTimes = std::array<double, 4>;
    using MaskParents = std::array<DpParent, 4>;

    const MaskTimes unreachable{
        infinity, infinity, infinity, infinity
    };

    std::vector<std::vector<MaskTimes>> dp(
        stint_count + 1,
        std::vector<MaskTimes>(total_laps + 1, unreachable)
    );

    std::vector<std::vector<MaskParents>> parent(
        stint_count + 1,
        std::vector<MaskParents>(total_laps + 1)
    );

    dp[0][0][0] = 0.0;

    for (int k = 0; k < stint_count; ++k) {
        for (int completed = 0; completed <= total_laps; ++completed) {
            for (int mask = 0; mask < 4; ++mask) {
                if (!std::isfinite(dp[k][completed][mask])) {
                    continue;
                }

                // Reserve at least one lap for each later stint.
                const int later_stints = stint_count - k - 1;
                const int max_length =
                    total_laps - completed - later_stints;

                for (int length = 1; length <= max_length; ++length) {
                    for (int c = 0; c < 2; ++c) {
                        const int next_laps = completed + length;
                        const int next_mask = mask | (1 << c);

                        const double pit_loss =
                            k == 0 ? 0.0 : config.pit_loss_s;

                        const double candidate =
                            dp[k][completed][mask]
                            + cost[completed][length][c]
                            + pit_loss;

                        if (candidate < dp[k + 1][next_laps][next_mask]) {
                            dp[k + 1][next_laps][next_mask] = candidate;

                            parent[k + 1][next_laps][next_mask] = {
                                completed,
                                mask,
                                compounds[c],
                                length
                            };
                        }
                    }
                }
            }
        }
    }

    const double best_time = dp[stint_count][total_laps][3];

    if (!std::isfinite(best_time)) {
        throw std::runtime_error("No feasible strategy found by DP");
    }

    // Recover the selected stints from the final state.
    std::vector<Stint> strategy;
    int completed = total_laps;
    int mask = 3;

    for (int k = stint_count; k > 0; --k) {
        const auto& previous = parent[k][completed][mask];

        strategy.push_back({
            previous.compound,
            previous.stint_laps,
            0
        });

        completed = previous.previous_laps;
        mask = previous.previous_mask;
    }

    std::reverse(strategy.begin(), strategy.end());

    return {std::move(strategy), best_time};
}

struct InventoryParent {
    int previous_laps = -1;
    int tyre_set_index = -1;
};

StrategyResult optimize_inventory_dp(
    const RaceConfig& config,
    const std::vector<TyreSet>& inventory,
    int stop_count
) {
    // Keep the exponential state space small in this first version.
    if (inventory.empty() || inventory.size() > 10) {
        throw std::invalid_argument(
            "Inventory DP supports between 1 and 10 tyre sets"
        );
    }

    const int set_count = static_cast<int>(inventory.size());

    if (
        config.total_laps <= 0 ||
        stop_count < 1 ||
        stop_count >= set_count ||
        stop_count >= config.total_laps
    ) {
        throw std::invalid_argument("Invalid race or stop count");
    }

    for (int i = 0; i < set_count; ++i) {
        if (inventory[i].id < 0 || inventory[i].initial_age < 0) {
            throw std::invalid_argument("Invalid tyre set");
        }

        tyre_model(config, inventory[i].compound);

        for (int j = 0; j < i; ++j) {
            if (inventory[i].id == inventory[j].id) {
                throw std::invalid_argument("Duplicate tyre set ID");
            }
        }
    }

    const int target_stints = stop_count + 1;
    const int total_laps = config.total_laps;
    const unsigned mask_count = 1u << set_count;
    const double infinity = std::numeric_limits<double>::infinity();

    std::vector<std::vector<double>> dp(
        mask_count,
        std::vector<double>(total_laps + 1, infinity)
    );

    std::vector<std::vector<InventoryParent>> parent(
        mask_count,
        std::vector<InventoryParent>(total_laps + 1)
    );

    dp[0][0] = 0.0;

    for (unsigned mask = 0; mask < mask_count; ++mask) {
        const int used_count = std::popcount(mask);

        if (used_count >= target_stints) {
            continue;
        }

        for (int completed = 0; completed < total_laps; ++completed) {
            if (!std::isfinite(dp[mask][completed])) {
                continue;
            }

            const int later_stints =
                target_stints - used_count - 1;

            const int max_length =
                total_laps - completed - later_stints;

            for (int i = 0; i < set_count; ++i) {
                const unsigned bit = 1u << i;

                if (mask & bit) {
                    continue;
                }

                const auto& tyre_set = inventory[i];
                const unsigned next_mask = mask | bit;

                double stint_time = 0.0;

                for (int length = 1; length <= max_length; ++length) {
                    // Extend the stint by one lap.
                    stint_time += predict_lap_time(
                        config,
                        tyre_set.compound,
                        tyre_set.initial_age + length - 1,
                        completed + length
                    );

                    const double pit_loss =
                        used_count == 0 ? 0.0 : config.pit_loss_s;

                    const int next_laps = completed + length;

                    const double candidate =
                        dp[mask][completed] + stint_time + pit_loss;

                    if (candidate < dp[next_mask][next_laps]) {
                        dp[next_mask][next_laps] = candidate;

                        parent[next_mask][next_laps] = {
                            completed,
                            i
                        };
                    }
                }
            }
        }
    }

    double best_time = infinity;
    unsigned best_mask = 0;

    for (unsigned mask = 0; mask < mask_count; ++mask) {
        if (std::popcount(mask) != target_stints) {
            continue;
        }

        int compound_mask = 0;

        for (int i = 0; i < set_count; ++i) {
            if (mask & (1u << i)) {
                compound_mask |=
                    inventory[i].compound == Compound::Soft ? 1 : 2;
            }
        }

        if (compound_mask != 3) {
            continue;
        }

        if (dp[mask][total_laps] < best_time) {
            best_time = dp[mask][total_laps];
            best_mask = mask;
        }
    }

    if (!std::isfinite(best_time)) {
        throw std::runtime_error("No feasible inventory DP strategy");
    }

    std::vector<Stint> strategy;
    unsigned mask = best_mask;
    int completed = total_laps;

    while (mask != 0) {
        const auto previous = parent[mask][completed];

        if (previous.tyre_set_index < 0) {
            throw std::runtime_error("Missing DP parent");
        }

        const auto& tyre_set = inventory[previous.tyre_set_index];

        strategy.push_back({
            tyre_set.compound,
            completed - previous.previous_laps,
            tyre_set.initial_age,
            tyre_set.id
        });

        mask ^= 1u << previous.tyre_set_index;
        completed = previous.previous_laps;
    }

    std::reverse(strategy.begin(), strategy.end());
    validate_inventory_strategy(strategy, inventory);

    return {std::move(strategy), best_time};
}
