cmake_minimum_required(VERSION 4.0)
project(f1_race_strategy_optimizer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(race_core
        race_model.cpp
        brute_force.cpp
        dynamic_programming.cpp
        config_io.cpp
        results_io.cpp
)

target_include_directories(race_core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(f1_race_strategy_optimizer
        main.cpp
)

target_link_libraries(f1_race_strategy_optimizer PRIVATE
        race_core
)

add_executable(optimizer_tests
        optimizer_tests.cpp
)

target_link_libraries(optimizer_tests PRIVATE
        race_core
)

enable_testing()

add_test(
        NAME optimizer_checks
        COMMAND optimizer_tests
)