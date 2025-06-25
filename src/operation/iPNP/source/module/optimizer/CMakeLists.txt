include_directories(${HOME_PLATFORM}/data_manager)

add_library(optimizer
    PdnOptimizer.cpp
    SimulatedAnnealing.cpp
)

target_link_libraries(optimizer
    PUBLIC
    evaluator
    data_manager
)