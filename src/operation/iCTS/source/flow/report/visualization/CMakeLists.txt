set(ICTS_FLOW_REPORT_VISUALIZATION_DRAWING ${ICTS_FLOW_REPORT_VISUALIZATION}/drawing)
set(ICTS_FLOW_REPORT_VISUALIZATION_GDS ${ICTS_FLOW_REPORT_VISUALIZATION}/gds)
set(ICTS_FLOW_REPORT_VISUALIZATION_SVG ${ICTS_FLOW_REPORT_VISUALIZATION}/svg)

add_subdirectory(${ICTS_FLOW_REPORT_VISUALIZATION_DRAWING})
add_subdirectory(${ICTS_FLOW_REPORT_VISUALIZATION_GDS})
add_subdirectory(${ICTS_FLOW_REPORT_VISUALIZATION_SVG})

add_library(icts_source_flow_report_visualization ${ICTS_FLOW_REPORT_VISUALIZATION}/Visualization.cc)

target_link_libraries(
  icts_source_flow_report_visualization
  PRIVATE
  icts_source_database_config
  icts_source_database_design
  icts_source_database_io
  icts_source_flow_report_visualization_gds
  icts_source_flow_report_visualization_svg
  icts_source_utils_logger
  log
)

target_include_directories(
  icts_source_flow_report_visualization
  PUBLIC
  ${ICTS_FLOW}
)
