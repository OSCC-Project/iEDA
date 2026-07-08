add_library(
  icts_source_flow_optimization_preparation
  ${ICTS_FLOW_OPTIMIZATION_PREPARATION}/OptimizationPreparation.cc
)

target_link_libraries(
  icts_source_flow_optimization_preparation
  PUBLIC
  icts_source_database_adapter_fast_sta
  icts_source_flow_optimization_model
  PRIVATE
  icts_source_database_config
  icts_source_database_design
  icts_source_flow_optimization_policy
  icts_source_flow_optimization_report
  icts_source_module_routing
  icts_source_utils_logger
  log
)

target_include_directories(
  icts_source_flow_optimization_preparation
  PUBLIC
  ${ICTS_FLOW}
)
