add_library(icts_source_flow_report_visualization_drawing ${ICTS_FLOW_REPORT_VISUALIZATION_DRAWING}/Drawing.cc)

target_link_libraries(
  icts_source_flow_report_visualization_drawing
  PUBLIC
  icts_source_database_design
  icts_source_database_spatial
  PRIVATE
  icts_source_database_io
  icts_source_module_routing
  log
)

target_include_directories(
  icts_source_flow_report_visualization_drawing
  PUBLIC
  ${ICTS_FLOW}
)
