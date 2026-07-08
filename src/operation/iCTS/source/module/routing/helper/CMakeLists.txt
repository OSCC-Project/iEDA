add_library(
  icts_source_module_routing_helper
  ${ICTS_MODULE_ROUTING}/helper/PinLocationHelper.cc
  ${ICTS_MODULE_ROUTING}/helper/SaltPinBuilder.cc
)

target_link_libraries(
  icts_source_module_routing_helper
  PUBLIC
  icts_source_database_routing
  icts_source_database_spatial
  PRIVATE
  icts_source_database_design
  salt
)

target_include_directories(
  icts_source_module_routing_helper
  PUBLIC
  ${ICTS_MODULE_ROUTING}/helper
)

if(DEBUG_ICTS_SOURCE_MODULE_ROUTING_HELPER)
  if(MSVC)
    target_compile_options(icts_source_module_routing_helper PRIVATE /Zi /Od /Ob0)
    target_link_options(icts_source_module_routing_helper PRIVATE /DEBUG)
  else()
    target_compile_options(icts_source_module_routing_helper PRIVATE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
