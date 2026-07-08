add_library(icts_source_flow_synthesis_trace_topology_build ${ICTS_FLOW_SYNTHESIS_TRACE_TOPOLOGY_BUILD}/TopologyBuildTrace.cc)

target_link_libraries(
  icts_source_flow_synthesis_trace_topology_build
  PUBLIC
  icts_source_module_characterization_headers
  PRIVATE
  icts_source_module_topology_clustering
  icts_source_database_design
)

target_include_directories(
  icts_source_flow_synthesis_trace_topology_build
  PUBLIC
  ${ICTS_FLOW}
)
