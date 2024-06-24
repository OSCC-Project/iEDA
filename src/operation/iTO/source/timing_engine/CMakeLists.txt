add_library(ito_timing_engine
  timing_engine.cpp
  timing_engine_builder.cpp
  timing_engine_calculate.cpp
  timing_engine_inst.cpp
  timing_engine_util.cpp
)

target_link_libraries(ito_timing_engine
    PUBLIC
    ista-engine
    ito_config
    ito_eval
    ito_placer
    ito_utility
)

target_include_directories(ito_timing_engine
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

