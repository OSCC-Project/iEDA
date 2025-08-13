include_directories(${HOME_PLATFORM}/data_manager)

add_library(pnp-optimizer
    PdnOptimizer.cpp
    SimulatedAnnealing.cpp
)

target_link_libraries(pnp-optimizer
    PUBLIC
    pnp-evaluator
    pnp-data
)