add_library(icts_source_flow_synthesis_trace_distance ${ICTS_FLOW_SYNTHESIS_TRACE_DISTANCE}/TopologyDistanceReport.cc)

target_link_libraries(
  icts_source_flow_synthesis_trace_distance
  PUBLIC
  icts_source_module_characterization_headers
  PRIVATE
  icts_source_flow_synthesis_topology_buffer
  icts_source_module_topology_clustering
  icts_source_database_design
  icts_source_utils_logger
  log
)

target_include_directories(
  icts_source_flow_synthesis_trace_distance
  PUBLIC
  ${ICTS_FLOW}
)
