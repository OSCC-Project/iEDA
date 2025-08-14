include_directories(${HOME_PLATFORM}/data_manager)

add_library(pnp-optimizer
    PdnOptimizer.cpp
    SimulatedAnnealing.cpp
)

target_link_libraries(pnp-optimizer
    PUBLIC
    pnp-evaluator
    pnp-data
    pnp-config
    idm
)

target_include_directories(pnp-optimizer
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)