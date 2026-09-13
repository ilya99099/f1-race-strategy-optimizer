#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>
#include <array>
#include <algorithm>  // std::find_if, std::min
#include <cmath>      // std::abs

#include "race_types.hpp"
#include "race_model.hpp"
#include "brute_force.hpp"
#include "dynamic_programming.hpp"
#include "config_io.hpp"
#include "results_io.hpp"

void print_strategy(const std::vector<Stint>& strategy) {
    int completed_laps = 0;

    for (std::size_t i = 0; i < strategy.size(); ++i) {
        if (i > 0) {
            std::cout << " -> ";
        }

        const auto& stint = strategy[i];

        std::cout << compound_name(stint.compound)
                  << '(' << stint.laps << " laps";

        if (stint.tyre_set_id >= 0) {
            std::cout << ", set=" << stint.tyre_set_id
                      << ", initial_age=" << stint.initial_tyre_age;
        }

        std::cout << ')';

        completed_laps += stint.laps;

        if (i + 1 < strategy.size()) {
            std::cout << " [pit after lap " << completed_laps << ']';
        }
    }
}

int main(int argc, char* argv[]) {
    // Synthetic parameters for checking the simulator.
     RaceConfig config{
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

    if (argc != 2) {
        std::cerr
            << "Usage: " << argv[0]
            << " <pace_model.txt>\n";
        return 1;
    }

    try {
        config = load_pace_model(argv[1], config);
    } catch (const std::exception& error) {
        std::cerr << "CONFIG ERROR: " << error.what() << '\n';
        return 1;
    }

    std::cout << "Loaded pace model: " << argv[1] << '\n';

    const std::vector<Stint> reference_strategy{
        {Compound::Soft, 13, 3, 0},
        {Compound::Hard, 20, 0, 1},
        {Compound::Hard, 24, 0, 2}
    };

    const double reference_time =
        simulate(config, reference_strategy).total_time_s;

    const auto results = enumerate_strategies(config);

    if (results.empty()) {
        throw std::runtime_error("No feasible strategies found");
    }

    std::cout << std::fixed << std::setprecision(3);

    std::cout << "Strategies evaluated: " << results.size() << '\n';
    std::cout << "Reference time: " << reference_time << " s\n";

    // Print the best strategy for each stop count.
    for (int stops : {1, 2}) {
        const auto best = std::find_if(
            results.begin(),
            results.end(),
            [stops](const StrategyResult& result) {
                return result.strategy.size()
                    == static_cast<std::size_t>(stops + 1);
            }
        );

        if (best != results.end()) {
            std::cout << "\nBest with " << stops << " stop(s): ";
            print_strategy(best->strategy);
            std::cout << "\nTime: " << best->total_time_s << " s\n";
        }
    }

    const std::size_t top_count =
        std::min<std::size_t>(5, results.size());

    std::cout << "\nTop strategies:\n";

    for (std::size_t i = 0; i < top_count; ++i) {
        const auto& candidate = results[i];

        std::cout << i + 1 << ". ";
        print_strategy(candidate.strategy);

        std::cout
            << "\n   Time: " << candidate.total_time_s << " s"
            << " | Gap to best: "
            << candidate.total_time_s - results.front().total_time_s
            << " s\n";
    }

    // Sets observed in the reference race.
    // This is not the driver's verified complete pre-race inventory.
    const std::vector<TyreSet> inventory{
        {0, Compound::Soft, 3},
        {1, Compound::Hard, 0},
        {2, Compound::Hard, 0}
    };

    const auto inventory_results =
        enumerate_inventory_strategies(config, inventory, 2);


    if (inventory_results.empty()) {
        throw std::runtime_error("No feasible inventory strategies");
    }

    const auto& inventory_best = inventory_results.front();

    std::cout
        << "\n=== Reference comparison: same tyre inventory ===\n"
        << "Reference strategy: ";

    print_strategy(reference_strategy);

    std::cout
        << "\nReference model time: " << reference_time << " s\n"
        << "Best model time: " << inventory_best.total_time_s << " s\n"
        << "Model improvement over reference: "
        << reference_time - inventory_best.total_time_s
        << " s\n";

    if (inventory_results.empty()) {
        throw std::runtime_error("No feasible inventory strategies");
    }

    std::cout << "\n=== Limited tyre inventory ===\n";
    std::cout << "Strategies evaluated: "
              << inventory_results.size() << '\n';

    for (int stops : {1, 2}) {
        const auto best = std::find_if(
            inventory_results.begin(),
            inventory_results.end(),
            [stops](const StrategyResult& result) {
                return result.strategy.size()
                    == static_cast<std::size_t>(stops + 1);
            }
        );

        if (best == inventory_results.end()) {
            std::cout << "\nNo feasible strategy with "
                      << stops << " stop(s)\n";
            continue;
        }

        std::cout << "\nBest with " << stops << " stop(s): ";
        print_strategy(best->strategy);
        std::cout << "\nTime: " << best->total_time_s << " s\n";
    }

    std::cout << "\n=== Pit-loss sensitivity ===\n";
    std::cout
        << "pit_loss_s,one_stop_s,two_stops_s,"
        << "two_stop_advantage_s,preferred\n";

    for (double pit_loss : {15.0, 20.0, 22.0, 23.1, 25.0, 30.0}) {
        RaceConfig scenario = config;
        scenario.pit_loss_s = pit_loss;

        const auto one_stop =
            optimize_inventory_dp(scenario, inventory, 1);

        const auto two_stops =
            optimize_inventory_dp(scenario, inventory, 2);

        // Positive means the two-stop strategy is faster.
        const double advantage =
            one_stop.total_time_s - two_stops.total_time_s;

        const char* preferred;

        if (std::abs(advantage) <= 1e-6) {
            preferred = "TIE";
        } else if (advantage > 0.0) {
            preferred = "TWO_STOPS";
        } else {
            preferred = "ONE_STOP";
        }

        std::cout
            << pit_loss << ','
            << one_stop.total_time_s << ','
            << two_stops.total_time_s << ','
            << advantage << ','
            << preferred << '\n';
    }
    try {
        const auto reference_simulation =
            simulate(config, reference_strategy);

        const auto optimized_simulation =
            simulate(config, inventory_best.strategy);

        write_simulation_csv(
            "results/reference_laps.csv",
            reference_simulation
        );

        write_simulation_csv(
            "results/optimized_laps.csv",
            optimized_simulation
        );

        std::cout
            << "\nSaved: results/reference_laps.csv\n"
            << "Saved: results/optimized_laps.csv\n";
    } catch (const std::exception& error) {
        std::cerr << "EXPORT ERROR: " << error.what() << '\n';
        return 1;
    }

    return 0;


}