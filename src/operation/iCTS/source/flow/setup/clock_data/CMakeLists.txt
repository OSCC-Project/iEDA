add_library(icts_source_flow_setup_clock_data ${ICTS_FLOW_SETUP_CLOCK_DATA}/ClockDataRead.cc)

target_link_libraries(
  icts_source_flow_setup_clock_data
  PRIVATE
  icts_source_database_design
  icts_source_database_io
  icts_source_database_adapter_sdc
  icts_source_database_config
  icts_source_utils_logger
  log
)

target_include_directories(
  icts_source_flow_setup_clock_data
  PUBLIC
  ${ICTS_FLOW}
)
