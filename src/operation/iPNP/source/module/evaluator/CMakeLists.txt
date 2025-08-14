add_library(pnp-evaluator 
CongestionEval.cpp 
IREval.cpp 
FastPlacer.cpp 
DRCEval.cpp
)

target_link_libraries(pnp-evaluator PUBLIC power ista-engine pnp-data ipl-api pnp-config)

target_include_directories(pnp-evaluator
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)