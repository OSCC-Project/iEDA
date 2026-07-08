set(ICTS_FLOW_REPORT_VISUALIZATION_GDS_LAYER ${ICTS_FLOW_REPORT_VISUALIZATION_GDS}/layer)
set(ICTS_FLOW_REPORT_VISUALIZATION_GDS_WRITER ${ICTS_FLOW_REPORT_VISUALIZATION_GDS}/writer)

add_subdirectory(${ICTS_FLOW_REPORT_VISUALIZATION_GDS_WRITER})
add_subdirectory(${ICTS_FLOW_REPORT_VISUALIZATION_GDS_LAYER})

add_library(icts_source_flow_report_visualization_gds ${ICTS_FLOW_REPORT_VISUALIZATION_GDS}/GdsVisualization.cc)

target_link_libraries(
  icts_source_flow_report_visualization_gds
  PRIVATE
  icts_source_database_config
  icts_source_database_design
  icts_source_flow_report_visualization_drawing
  icts_source_flow_report_visualization_gds_layer
  icts_source_flow_report_visualization_gds_writer
  icts_source_utils_logger
  log
)

target_include_directories(
  icts_source_flow_report_visualization_gds
  PUBLIC
  ${ICTS_FLOW}
)
