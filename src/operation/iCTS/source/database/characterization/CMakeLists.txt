add_library(icts_source_database_characterization INTERFACE)

target_link_libraries(
  icts_source_database_characterization
  INTERFACE
)

target_include_directories(icts_source_database_characterization INTERFACE ${ICTS_DATABASE_CHARACTERIZATION} ${ICTS_DATABASE})

if(DEBUG_ICTS_SOURCE_DATABASE_CHARACTERIZATION)
  if(MSVC)
    target_compile_options(icts_source_database_characterization INTERFACE /Zi /Od /Ob0)
  else()
    target_compile_options(icts_source_database_characterization INTERFACE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
