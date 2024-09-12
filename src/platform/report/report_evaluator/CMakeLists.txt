add_library(ieda_report_evaluator
  report_evaluator.cpp
)

target_link_libraries(ieda_report_evaluator 
    PUBLIC
        report_basic
        tool_manager 
        def_service
        log
        eval_pro_wirelength_api
        eval_pro_congestion_api
        eval_pro_density_api
)

target_include_directories(ieda_report_evaluator 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)
