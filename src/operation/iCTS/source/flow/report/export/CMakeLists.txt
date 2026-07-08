add_library(icts_source_flow_report_export ${ICTS_FLOW_REPORT_EXPORT}/ReportExport.cc)

target_link_libraries(
  icts_source_flow_report_export
  PRIVATE
  icts_source_database_config
)

target_include_directories(
  icts_source_flow_report_export
  PUBLIC
  ${ICTS_FLOW}
)
