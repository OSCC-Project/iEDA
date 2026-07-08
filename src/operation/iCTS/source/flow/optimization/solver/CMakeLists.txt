add_library(
  icts_source_flow_optimization_solver
  ${ICTS_FLOW_OPTIMIZATION_SOLVER}/OptimizationSolver.cc
)

target_link_libraries(
  icts_source_flow_optimization_solver
  PUBLIC
  icts_source_database_adapter_fast_sta
  icts_source_flow_optimization_model
  PRIVATE
  icts_source_flow_optimization_candidate
  icts_source_flow_optimization_policy
  icts_source_flow_optimization_report
  icts_source_flow_optimization_state
  log
)

target_include_directories(
  icts_source_flow_optimization_solver
  PUBLIC
  ${ICTS_FLOW}
)
