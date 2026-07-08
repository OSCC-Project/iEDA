add_library(
  icts_source_module_routing
  ${ICTS_MODULE_ROUTING}/router/Router.cc
)

target_link_libraries(
  icts_source_module_routing
  PUBLIC
  icts_source_database_routing
  icts_source_database_timing
  icts_source_module_routing_local_legalization
  PRIVATE
  icts_source_database_design
  icts_source_module_routing_flute
  icts_source_module_routing_bst
  icts_source_module_routing_cbs
  icts_source_module_routing_salt
  icts_source_module_routing_helper
  icts_source_utils_geometry
  log
  lemon
)

target_include_directories(
  icts_source_module_routing
  PUBLIC
  ${ICTS_MODULE_ROUTING}
  ${ICTS_SOURCE}
  ${ICTS_DATABASE}
)

if(DEBUG_ICTS_SOURCE_MODULE_ROUTING)
  if(MSVC)
    target_compile_options(icts_source_module_routing PRIVATE /Zi /Od /Ob0)
    target_link_options(icts_source_module_routing PRIVATE /DEBUG)
  else()
    target_compile_options(icts_source_module_routing PRIVATE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
