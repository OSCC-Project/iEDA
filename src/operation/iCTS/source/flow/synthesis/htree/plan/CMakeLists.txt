add_library(
  icts_source_flow_synthesis_htree_plan
  ${ICTS_FLOW_SYNTHESIS_HTREE_PLAN}/Plan.cc
)

target_link_libraries(
  icts_source_flow_synthesis_htree_plan
  PRIVATE
  icts_source_database_spatial
  icts_source_module_characterization
  icts_source_utils_geometry
  log
)

add_library(
  icts_source_flow_synthesis_htree_depth_plan
  ${ICTS_FLOW_SYNTHESIS_HTREE_PLAN}/DepthPlan.cc
  ${ICTS_FLOW_SYNTHESIS_HTREE_PLAN}/DepthPlanReport.cc
)

target_link_libraries(
  icts_source_flow_synthesis_htree_depth_plan
  PUBLIC
  icts_source_flow_synthesis_htree_compensation
  icts_source_flow_synthesis_htree_region
  icts_source_flow_synthesis_htree_topology_pruning
  PRIVATE
  icts_source_flow_synthesis_htree_plan
  icts_source_utils_logger
)

target_include_directories(
  icts_source_flow_synthesis_htree_plan
  PUBLIC
  ${ICTS_SOURCE}
  ${ICTS_FLOW}
)

target_include_directories(
  icts_source_flow_synthesis_htree_depth_plan
  PUBLIC
  ${ICTS_SOURCE}
  ${ICTS_FLOW}
)
