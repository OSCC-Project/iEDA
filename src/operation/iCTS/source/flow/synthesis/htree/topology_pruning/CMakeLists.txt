add_library(
  icts_source_flow_synthesis_htree_topology_pruning
  ${ICTS_FLOW_SYNTHESIS_HTREE_TOPOLOGY_PRUNING}/TopologyPruning.cc
  ${ICTS_FLOW_SYNTHESIS_HTREE_TOPOLOGY_PRUNING}/TopologyPruningGlobalSelection.cc
)

target_link_libraries(
  icts_source_flow_synthesis_htree_topology_pruning
  PUBLIC
  icts_source_flow_synthesis_htree_constraint
  icts_source_flow_synthesis_htree_segment_pruning
  PRIVATE
  icts_source_flow_synthesis_htree_compensation
  icts_source_flow_synthesis_htree_region
  icts_source_utils_logger
  log
)

target_include_directories(
  icts_source_flow_synthesis_htree_topology_pruning
  PUBLIC
  ${ICTS_SOURCE}
  ${ICTS_FLOW}
)
