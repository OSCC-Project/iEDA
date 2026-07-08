add_library(
  icts_source_flow_report_qor
  ${ICTS_FLOW_REPORT_QOR}/QorReport.cc
  ${ICTS_FLOW_REPORT_QOR}/QorFiles.cc
)

target_link_libraries(
  icts_source_flow_report_qor
  PRIVATE
  icts_source_database_qor
  icts_source_flow_evaluation_qor
  icts_source_utils_logger
  log
)

target_include_directories(
  icts_source_flow_report_qor
  PUBLIC
  ${ICTS_FLOW}
)
