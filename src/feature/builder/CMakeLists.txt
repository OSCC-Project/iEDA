add_library(feature_builder
    feature_builder.cpp
    feature_builder_tool.cpp
    route_builder.cpp

    feature_eval_wirelength.cpp
    feature_eval_congestion.cpp
    feature_eval_density.cpp
    feature_eval_timing.cc

    feature_eval_union.cpp
)

target_include_directories(feature_builder 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(feature_builder 
    PRIVATE 
        feature_db
        idb
        flow_config
        idm
        irt_interface
        ipl-api
        icts_api
        ito_api
        # ista-engine
        # idrc_pro_api
        ieda_report_evaluator
        eval_congestion_api
        eval_density_api
        eval_wirelength_api
        eval_timing_api
        eval_union_api
)