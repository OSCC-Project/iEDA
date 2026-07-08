add_library(icts_source_flow_report_visualization_gds_layer ${ICTS_FLOW_REPORT_VISUALIZATION_GDS_LAYER}/LayerPolicy.cc)

target_link_libraries(
  icts_source_flow_report_visualization_gds_layer
  PUBLIC
  icts_source_flow_report_visualization_gds_writer
  PRIVATE
  icts_source_flow_report_visualization_drawing
  log
)

target_include_directories(
  icts_source_flow_report_visualization_gds_layer
  PUBLIC
  ${ICTS_FLOW}
)
