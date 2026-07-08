add_library(
  icts_source_flow_synthesis_htree_characterization_wirelength
  ${ICTS_FLOW_SYNTHESIS_HTREE_CHARACTERIZATION_WIRELENGTH}/WirelengthGrid.cc
)

target_link_libraries(
  icts_source_flow_synthesis_htree_characterization_wirelength
  PUBLIC
  icts_source_module_characterization
  PRIVATE
  icts_source_database_config
  icts_source_database_spatial
  icts_source_utils_geometry
  icts_source_utils_logger
  log
)

target_include_directories(
  icts_source_flow_synthesis_htree_characterization_wirelength
  PUBLIC
  ${ICTS_SOURCE}
  ${ICTS_FLOW}
)
