add_library(
  icts_source_flow_optimization_policy
  ${ICTS_FLOW_OPTIMIZATION_POLICY}/OptimizationPolicy.cc
)

target_include_directories(
  icts_source_flow_optimization_policy
  PUBLIC
  ${ICTS_FLOW}
)
