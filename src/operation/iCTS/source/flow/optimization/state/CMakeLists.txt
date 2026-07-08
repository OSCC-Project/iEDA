add_library(
  icts_source_flow_optimization_state
  ${ICTS_FLOW_OPTIMIZATION_STATE}/OptimizationState.cc
)

target_link_libraries(
  icts_source_flow_optimization_state
  PUBLIC
  icts_source_database_adapter_fast_sta
  icts_source_flow_optimization_model
)

target_include_directories(
  icts_source_flow_optimization_state
  PUBLIC
  ${ICTS_FLOW}
)
