add_library(feature_builder
    feature_builder.cpp
    feature_builder_tool.cpp
    feature_timing_eval.cc
    route_builder.cpp
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
        eval_pro_congestion_api
        eval_pro_density_api
        eval_pro_wirelength_api
        eval_pro_timing_api
)