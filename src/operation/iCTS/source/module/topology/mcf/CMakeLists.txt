add_library(icts_source_module_topology_mcf INTERFACE)

target_link_libraries(
  icts_source_module_topology_mcf
  INTERFACE
  lemon
)

set_target_properties(icts_source_module_topology_mcf PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(
  icts_source_module_topology_mcf
  INTERFACE
  ${ICTS_MODULE_TOPOLOGY}/mcf
)

if(DEBUG_ICTS_SOURCE_MODULE_TOPOLOGY_MCF)
  if(MSVC)
    target_compile_options(icts_source_module_topology_mcf INTERFACE /Zi /Od /Ob0)
  else()
    target_compile_options(icts_source_module_topology_mcf INTERFACE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
