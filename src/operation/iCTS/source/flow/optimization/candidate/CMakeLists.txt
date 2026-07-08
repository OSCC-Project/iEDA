add_library(
  icts_source_flow_optimization_candidate
  ${ICTS_FLOW_OPTIMIZATION_CANDIDATE}/OptimizationCandidates.cc
  ${ICTS_FLOW_OPTIMIZATION_CANDIDATE}/OptimizationScalableCandidates.cc
)

target_link_libraries(
  icts_source_flow_optimization_candidate
  PUBLIC
  icts_source_database_adapter_fast_sta
  icts_source_flow_optimization_model
  PRIVATE
  icts_source_flow_optimization_policy
  icts_source_flow_optimization_preparation
  icts_source_flow_optimization_state
  log
)

target_include_directories(
  icts_source_flow_optimization_candidate
  PUBLIC
  ${ICTS_FLOW}
)
