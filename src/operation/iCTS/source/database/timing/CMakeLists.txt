add_library(icts_source_database_timing INTERFACE)

target_link_libraries(icts_source_database_timing INTERFACE icts_source_database_spatial)

target_include_directories(icts_source_database_timing INTERFACE ${ICTS_DATABASE_TIMING} ${ICTS_DATABASE})

if(DEBUG_ICTS_SOURCE_DATABASE_TIMING)
  if(MSVC)
    target_compile_options(icts_source_database_timing INTERFACE /Zi /Od /Ob0)
  else()
    target_compile_options(icts_source_database_timing INTERFACE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
