cmake_minimum_required(VERSION 4.0)
project(f1_race_strategy_optimizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(race_core
        src/race_model.cpp
        src/brute_force.cpp
        src/dynamic_programming.cpp
        src/config_io.cpp
        src/results_io.cpp
)

target_include_directories(race_core PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

add_executable(f1_race_strategy_optimizer
        app/main.cpp
)

target_link_libraries(f1_race_strategy_optimizer PRIVATE
        race_core
)

add_executable(optimizer_tests
        tests/optimizer_tests.cpp
)

target_link_libraries(optimizer_tests PRIVATE
        race_core
)

enable_testing()

add_test(
        NAME optimizer_checks
        COMMAND optimizer_tests
)