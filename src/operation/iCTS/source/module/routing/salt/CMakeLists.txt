add_library(icts_source_module_routing_salt ${ICTS_MODULE_ROUTING}/salt/SALTRouter.cc)

target_link_libraries(
  icts_source_module_routing_salt
  PUBLIC
  icts_source_database_routing
  PRIVATE
  icts_source_module_routing_helper
  icts_source_utils_geometry
  log
  salt
)

target_include_directories(
  icts_source_module_routing_salt
  PUBLIC
  ${ICTS_MODULE_ROUTING}/salt
)

if(DEBUG_ICTS_SOURCE_MODULE_ROUTING_SALT)
  if(MSVC)
    target_compile_options(icts_source_module_routing_salt PRIVATE /Zi /Od /Ob0)
    target_link_options(icts_source_module_routing_salt PRIVATE /DEBUG)
  else()
    target_compile_options(icts_source_module_routing_salt PRIVATE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
