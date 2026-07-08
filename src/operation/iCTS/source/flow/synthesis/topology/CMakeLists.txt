set(ICTS_FLOW_SYNTHESIS_TOPOLOGY_BUFFER ${ICTS_FLOW_SYNTHESIS_TOPOLOGY}/buffer)
set(ICTS_FLOW_SYNTHESIS_TOPOLOGY_SINK ${ICTS_FLOW_SYNTHESIS_TOPOLOGY}/sink)
set(ICTS_FLOW_SYNTHESIS_TOPOLOGY_TRUNK ${ICTS_FLOW_SYNTHESIS_TOPOLOGY}/trunk)

add_library(
  icts_source_flow_synthesis_topology_source_trunk_stage
  ${ICTS_FLOW_SYNTHESIS_TOPOLOGY}/SourceTrunkStage.cc
)

target_include_directories(
  icts_source_flow_synthesis_topology_source_trunk_stage
  PUBLIC
  ${ICTS_FLOW}
)

add_subdirectory(${ICTS_FLOW_SYNTHESIS_TOPOLOGY_BUFFER})
add_subdirectory(${ICTS_FLOW_SYNTHESIS_TOPOLOGY_SINK})
add_subdirectory(${ICTS_FLOW_SYNTHESIS_TOPOLOGY_TRUNK})

add_library(icts_source_flow_synthesis_topology ${ICTS_FLOW_SYNTHESIS_TOPOLOGY}/Topology.cc)

target_link_libraries(
  icts_source_flow_synthesis_topology
  PRIVATE
  icts_source_flow_synthesis_topology_sink
  icts_source_flow_synthesis_topology_trunk
  icts_source_database_config
  icts_source_database_io
  icts_source_database_spatial
  icts_source_flow_synthesis_realization
  icts_source_flow_synthesis_topology_source_trunk_stage
  icts_source_flow_synthesis_trace
  icts_source_flow_synthesis_trace_layout
  icts_source_utils_geometry
  icts_source_utils_logger
  log
)

target_link_libraries(
  icts_source_flow_synthesis_topology
  PUBLIC
  icts_source_flow_synthesis_htree
  icts_source_module_topology_clustering
)

target_include_directories(
  icts_source_flow_synthesis_topology
  PUBLIC
  ${ICTS_SOURCE}
  ${ICTS_FLOW}
)
