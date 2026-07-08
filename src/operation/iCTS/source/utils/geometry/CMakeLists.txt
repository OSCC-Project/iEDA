add_library(icts_source_utils_geometry INTERFACE)

target_link_libraries(icts_source_utils_geometry INTERFACE icts_source_database_spatial)

target_include_directories(
  icts_source_utils_geometry
  INTERFACE
  ${ICTS_GEOMETRY}
  ${ICTS_UTILS}
)

if(DEBUG_ICTS_SOURCE_UTILS_GEOMETRY)
  if(MSVC)
    target_compile_options(icts_source_utils_geometry INTERFACE /Zi /Od /Ob0)
  else()
    target_compile_options(icts_source_utils_geometry INTERFACE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
