add_library(icts_source_database_routing INTERFACE)

target_link_libraries(icts_source_database_routing INTERFACE icts_source_database_spatial)

target_include_directories(icts_source_database_routing INTERFACE ${ICTS_DATABASE_ROUTING} ${ICTS_DATABASE})

if(DEBUG_ICTS_SOURCE_DATABASE_ROUTING)
  if(MSVC)
    target_compile_options(icts_source_database_routing INTERFACE /Zi /Od /Ob0)
  else()
    target_compile_options(icts_source_database_routing INTERFACE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
