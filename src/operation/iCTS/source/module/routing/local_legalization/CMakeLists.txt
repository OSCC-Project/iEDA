add_library(icts_source_module_routing_local_legalization ${ICTS_MODULE_ROUTING}/local_legalization/LocalLegalization.cc)

target_link_libraries(
  icts_source_module_routing_local_legalization
  PUBLIC
  icts_source_database_spatial
  icts_source_utils_geometry
  PRIVATE
  log
)

target_include_directories(
  icts_source_module_routing_local_legalization
  PUBLIC
  ${ICTS_MODULE_ROUTING}/local_legalization
)

if(DEBUG_ICTS_SOURCE_MODULE_ROUTING_LOCAL_LEGALIZATION)
  if(MSVC)
    target_compile_options(icts_source_module_routing_local_legalization PRIVATE /Zi /Od /Ob0)
    target_link_options(icts_source_module_routing_local_legalization PRIVATE /DEBUG)
  else()
    target_compile_options(icts_source_module_routing_local_legalization PRIVATE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
