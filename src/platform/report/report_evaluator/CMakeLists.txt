add_library(ieda_report_evaluator
  report_evaluator.cpp
)

target_link_libraries(ieda_report_evaluator 
    PUBLIC
        report_basic
        tool_manager 
        def_service
        log
        eval_wirelength_api
        eval_congestion_api
        eval_density_api
)

target_include_directories(ieda_report_evaluator 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)
