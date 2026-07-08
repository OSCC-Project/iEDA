add_library(icts_source_database_spatial INTERFACE)

target_include_directories(icts_source_database_spatial INTERFACE ${ICTS_DATABASE_SPATIAL} ${ICTS_DATABASE})

if(DEBUG_ICTS_SOURCE_DATABASE_SPATIAL)
  if(MSVC)
    target_compile_options(icts_source_database_spatial INTERFACE /Zi /Od /Ob0)
  else()
    target_compile_options(icts_source_database_spatial INTERFACE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
