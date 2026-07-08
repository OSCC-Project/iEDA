add_library(icts_source_flow_report_overview ${ICTS_FLOW_REPORT_OVERVIEW}/Overview.cc)

target_link_libraries(
  icts_source_flow_report_overview
  PRIVATE
  icts_source_flow_report_export
  icts_source_utils_logger
)

target_include_directories(
  icts_source_flow_report_overview
  PUBLIC
  ${ICTS_FLOW}
)
