add_library(
  icts_source_flow_synthesis_htree_compensation
  ${ICTS_FLOW_SYNTHESIS_HTREE_COMPENSATION}/RootDriverCompensation.cc
  ${ICTS_FLOW_SYNTHESIS_HTREE_COMPENSATION}/RootDriverCompensationLoad.cc
)

target_link_libraries(
  icts_source_flow_synthesis_htree_compensation
  PUBLIC
  icts_source_database_characterization
  PRIVATE
  icts_source_database_io
  icts_source_database_design
  icts_source_database_routing
  icts_source_flow_synthesis_htree_segment_pruning
  icts_source_module_routing
  icts_source_utils_logger
  log
)

target_include_directories(
  icts_source_flow_synthesis_htree_compensation
  PUBLIC
  ${ICTS_SOURCE}
  ${ICTS_FLOW}
)
