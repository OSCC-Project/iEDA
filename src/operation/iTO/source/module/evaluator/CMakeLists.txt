add_library(ito_eval
EstimateParasitics.cpp
RoutingTree.cpp
)

target_link_libraries(ito_eval
    PUBLIC
    ito_timing_engine
    ito_data_manager
    ito_config
)

target_include_directories(ito_eval
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_OPERATION}/iTO/source/data
)


