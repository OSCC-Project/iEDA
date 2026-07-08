add_library(icts_source_flow_synthesis_topology_buffer ${ICTS_FLOW_SYNTHESIS_TOPOLOGY_BUFFER}/BufferInsertion.cc)

target_link_libraries(
  icts_source_flow_synthesis_topology_buffer
  PUBLIC
  icts_source_database_design
  icts_source_module_characterization_headers
  PRIVATE
  icts_source_database_spatial
  icts_source_module_topology_clustering
  icts_source_utils_logger
  log
)

target_include_directories(
  icts_source_flow_synthesis_topology_buffer
  PUBLIC
  ${ICTS_FLOW}
)
