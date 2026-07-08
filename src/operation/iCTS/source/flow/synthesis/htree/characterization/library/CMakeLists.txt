add_library(
  icts_source_flow_synthesis_htree_characterization_library
  ${ICTS_FLOW_SYNTHESIS_HTREE_CHARACTERIZATION_LIBRARY}/CharacterizationLibrary.cc
)

target_link_libraries(
  icts_source_flow_synthesis_htree_characterization_library
  PUBLIC
  icts_source_database_routing
  icts_source_module_characterization
  PRIVATE
  icts_source_database_config
  icts_source_database_io
  icts_source_utils_logger
  log
)

target_include_directories(
  icts_source_flow_synthesis_htree_characterization_library
  PUBLIC
  ${ICTS_SOURCE}
  ${ICTS_FLOW}
)
