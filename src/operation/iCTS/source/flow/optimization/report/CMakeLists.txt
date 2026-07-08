add_library(
  icts_source_flow_optimization_report
  ${ICTS_FLOW_OPTIMIZATION_REPORT}/OptimizationReport.cc
)

target_link_libraries(
  icts_source_flow_optimization_report
  PUBLIC
  icts_source_flow_optimization_model
  PRIVATE
  icts_source_database_design
  icts_source_utils_logger
  log
)

target_include_directories(
  icts_source_flow_optimization_report
  PUBLIC
  ${ICTS_FLOW}
)
