set(ICTS_FLOW_REPORT_EXPORT ${ICTS_FLOW_REPORT}/export)
set(ICTS_FLOW_REPORT_OVERVIEW ${ICTS_FLOW_REPORT}/overview)
set(ICTS_FLOW_REPORT_QOR ${ICTS_FLOW_REPORT}/qor)
set(ICTS_FLOW_REPORT_VISUALIZATION ${ICTS_FLOW_REPORT}/visualization)

add_subdirectory(${ICTS_FLOW_REPORT_EXPORT})
add_subdirectory(${ICTS_FLOW_REPORT_OVERVIEW})
add_subdirectory(${ICTS_FLOW_REPORT_QOR})
add_subdirectory(${ICTS_FLOW_REPORT_VISUALIZATION})

add_library(
  icts_source_flow_report
  ${ICTS_FLOW_REPORT}/Report.cc
)

target_link_libraries(
  icts_source_flow_report
  PRIVATE
  icts_source_database_config
  icts_source_database_design
  icts_source_database_io
  icts_source_flow_evaluation
  icts_source_flow_report_export
  icts_source_flow_report_overview
  icts_source_flow_report_qor
  icts_source_flow_report_visualization
  icts_source_utils_logger
  log
)

target_include_directories(
  icts_source_flow_report
  PUBLIC
  ${ICTS_FLOW}
)
