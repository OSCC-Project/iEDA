add_library(optimizer
    PdnOptimizer.cpp
    SimulatedAnnealing.cpp
    
)

target_link_libraries(optimizer
    PUBLIC
    evaluator
)