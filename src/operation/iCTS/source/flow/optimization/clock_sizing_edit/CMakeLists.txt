add_library(
  icts_source_flow_optimization_clock_sizing_edit
  ${ICTS_FLOW_OPTIMIZATION_CLOCK_SIZING_EDIT}/ClockSizingAcceptedEdit.cc
)

target_link_libraries(
  icts_source_flow_optimization_clock_sizing_edit
  PUBLIC
  icts_source_flow_optimization_model
  PRIVATE
  icts_source_database_design
  log
)

target_include_directories(
  icts_source_flow_optimization_clock_sizing_edit
  PUBLIC
  ${ICTS_FLOW}
)
