add_library(
  icts_source_flow_evaluation_qor
  ${ICTS_FLOW_EVALUATION_QOR}/QorEvaluation.cc
  ${ICTS_FLOW_EVALUATION_QOR}/QorEvaluationMetrics.cc
)

target_link_libraries(
  icts_source_flow_evaluation_qor
  PUBLIC
  icts_source_database_qor
  icts_source_database_design
  PRIVATE
  icts_source_database_config
  icts_source_database_io
  icts_source_module_routing
  icts_source_module_timing
  icts_source_utils_logger
  log
)

target_include_directories(
  icts_source_flow_evaluation_qor
  PUBLIC
  ${ICTS_FLOW}
)
