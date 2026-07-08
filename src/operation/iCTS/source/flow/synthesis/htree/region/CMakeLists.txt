add_library(icts_source_flow_synthesis_htree_region ${ICTS_FLOW_SYNTHESIS_HTREE_REGION}/SinkLoadRegion.cc)

target_link_libraries(
  icts_source_flow_synthesis_htree_region
  PRIVATE
  icts_source_flow_synthesis_htree_segment_pruning
  icts_source_flow_synthesis_htree_constraint
  icts_source_database_routing
  icts_source_database_spatial
  icts_source_module_topology_clustering
  icts_source_module_topology_config
  icts_source_module_topology_fast_clustering
  icts_source_utils_logger
  log
)

target_include_directories(
  icts_source_flow_synthesis_htree_region
  PUBLIC
  ${ICTS_SOURCE}
  ${ICTS_FLOW}
)
