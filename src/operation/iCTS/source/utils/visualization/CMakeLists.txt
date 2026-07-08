add_library(
  icts_source_utils_visualization
  STATIC
  ${ICTS_VISUALIZATION}/cluster/ClusterSvgWriter.cc
  ${ICTS_VISUALIZATION}/topology/TopologySvgWriter.cc
)

target_link_libraries(
  icts_source_utils_visualization
  PUBLIC
  icts_source_database_design
  icts_source_database_spatial
)

target_include_directories(
  icts_source_utils_visualization
  PUBLIC
  ${ICTS_VISUALIZATION}
  ${ICTS_UTILS}
)

if(DEBUG_ICTS_SOURCE_UTILS_VISUALIZATION)
  if(MSVC)
    target_compile_options(icts_source_utils_visualization PRIVATE /Zi /Od /Ob0)
    target_link_options(icts_source_utils_visualization PRIVATE /DEBUG)
  else()
    target_compile_options(icts_source_utils_visualization PRIVATE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
