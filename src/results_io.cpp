#include "results_io.hpp"
#include "race_model.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

void write_simulation_csv(
    const std::filesystem::path& path,
    const SimulationResult& result
) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::ofstream output(path);

    if (!output) {
        throw std::runtime_error(
            "Cannot create results file: " + path.string()
        );
    }

    output
        << "lap,stint,compound,tyre_age_at_start,"
        << "driving_time_s,pit_loss_s,cumulative_time_s\n";

    output << std::fixed << std::setprecision(6);

    for (const auto& lap : result.laps) {
        output
            << lap.race_lap << ','
            << lap.stint_number << ','
            << compound_name(lap.compound) << ','
            << lap.tyre_age_at_start << ','
            << lap.driving_time_s << ','
            << lap.pit_loss_s << ','
            << lap.cumulative_time_s << '\n';
    }

    output.close();

    if (!output) {
        throw std::runtime_error(
            "Failed to write results file: " + path.string()
        );
    }
}