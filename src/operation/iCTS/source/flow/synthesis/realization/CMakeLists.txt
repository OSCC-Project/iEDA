add_library(
  icts_source_flow_synthesis_realization
  OBJECT
  ${ICTS_FLOW_SYNTHESIS_REALIZATION}/ClockTreeRealization.cc
)

target_link_libraries(
  icts_source_flow_synthesis_realization
  PRIVATE
  icts_source_utils_logger
  log
)

target_include_directories(
  icts_source_flow_synthesis_realization
  PUBLIC
  ${ICTS_FLOW}
  PRIVATE
  ${ICTS_DATABASE}
)

if(DEBUG_ICTS_SOURCE_FLOW_SYNTHESIS_REALIZATION)
  if(MSVC)
    target_compile_options(icts_source_flow_synthesis_realization PRIVATE /Zi /Od /Ob0)
    target_link_options(icts_source_flow_synthesis_realization PRIVATE /DEBUG)
  else()
    target_compile_options(icts_source_flow_synthesis_realization PRIVATE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
