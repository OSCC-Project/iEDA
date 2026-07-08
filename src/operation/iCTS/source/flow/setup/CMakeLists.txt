set(ICTS_FLOW_SETUP_CLOCK_DATA ${ICTS_FLOW_SETUP}/clock_data)

add_subdirectory(${ICTS_FLOW_SETUP_CLOCK_DATA})

add_library(icts_source_flow_setup ${ICTS_FLOW_SETUP}/Setup.cc)

target_link_libraries(
  icts_source_flow_setup
  PRIVATE
  idm
  icts_source_database_config
  icts_source_database_design
  icts_source_database_io
  icts_source_flow_setup_clock_data
  icts_source_utils_logger
  log
)

target_include_directories(
  icts_source_flow_setup
  PUBLIC
  ${ICTS_FLOW}
)
