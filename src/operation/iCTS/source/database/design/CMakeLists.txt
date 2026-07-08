add_library(
  icts_source_database_design
  ${ICTS_DATABASE_DESIGN}/ClockDAG.cc
  ${ICTS_DATABASE_DESIGN}/ClockNetwork.cc
  ${ICTS_DATABASE_DESIGN}/ClockLayout.cc
  ${ICTS_DATABASE_DESIGN}/Design.cc
)

target_link_libraries(
  icts_source_database_design
  PUBLIC
  icts_source_database_spatial
  PRIVATE
  icts_source_database_config
  icts_source_utils_logger
  log
)

target_include_directories(
  icts_source_database_design
  PUBLIC
  ${ICTS_DATABASE_DESIGN}
  ${ICTS_DATABASE}
  PRIVATE
  ${ICTS_SOURCE}
)

if(DEBUG_ICTS_SOURCE_DATABASE_DESIGN)
  if(MSVC)
    target_compile_options(icts_source_database_design PRIVATE /Zi /Od /Ob0)
    target_link_options(icts_source_database_design PRIVATE /DEBUG)
  else()
    target_compile_options(icts_source_database_design PRIVATE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
