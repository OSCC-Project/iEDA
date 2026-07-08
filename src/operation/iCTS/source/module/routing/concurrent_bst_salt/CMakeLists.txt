add_library(icts_source_module_routing_cbs ${ICTS_MODULE_ROUTING}/concurrent_bst_salt/CBSRouter.cc)

target_link_libraries(
  icts_source_module_routing_cbs
  PUBLIC
  icts_source_database_routing
  PRIVATE
  icts_source_module_routing_bst
  icts_source_utils_geometry
  log
  salt
)

target_include_directories(
  icts_source_module_routing_cbs
  PUBLIC
  ${ICTS_MODULE_ROUTING}/concurrent_bst_salt
)

if(DEBUG_ICTS_SOURCE_MODULE_ROUTING_CBS)
  if(MSVC)
    target_compile_options(icts_source_module_routing_cbs PRIVATE /Zi /Od /Ob0)
    target_link_options(icts_source_module_routing_cbs PRIVATE /DEBUG)
  else()
    target_compile_options(icts_source_module_routing_cbs PRIVATE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
