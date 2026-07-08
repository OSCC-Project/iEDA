add_library(icts_source_module_routing_flute ${ICTS_MODULE_ROUTING}/flute/FLUTERouter.cc)

target_link_libraries(
  icts_source_module_routing_flute
  PUBLIC
  icts_source_database_routing
  PRIVATE
  icts_source_module_routing_helper
  icts_source_utils_geometry
  log
  flute
  salt
)

target_include_directories(
  icts_source_module_routing_flute
  PUBLIC
  ${ICTS_MODULE_ROUTING}/flute
)

if(DEBUG_ICTS_SOURCE_MODULE_ROUTING_FLUTE)
  if(MSVC)
    target_compile_options(icts_source_module_routing_flute PRIVATE /Zi /Od /Ob0)
    target_link_options(icts_source_module_routing_flute PRIVATE /DEBUG)
  else()
    target_compile_options(icts_source_module_routing_flute PRIVATE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
