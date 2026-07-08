add_library(icts_source_flow_report_visualization_gds_writer ${ICTS_FLOW_REPORT_VISUALIZATION_GDS_WRITER}/GdsStream.cc)

target_link_libraries(
  icts_source_flow_report_visualization_gds_writer
  PUBLIC
  icts_source_database_spatial
  PRIVATE
  log
)

target_include_directories(
  icts_source_flow_report_visualization_gds_writer
  PUBLIC
  ${ICTS_FLOW}
)
