set(ICTS_DATABASE_ADAPTER_FAST_STA ${ICTS_DATABASE}/adapter/fast_sta)
set(ICTS_DATABASE_ADAPTER_SDC ${ICTS_DATABASE}/adapter/sdc)

add_subdirectory(${ICTS_DATABASE_ADAPTER_FAST_STA})
add_subdirectory(${ICTS_DATABASE_ADAPTER_SDC})

add_library(icts_source_database_adapter INTERFACE)

target_link_libraries(
  icts_source_database_adapter
  INTERFACE
  icts_source_database_adapter_fast_sta
  icts_source_database_adapter_sdc
)

target_include_directories(icts_source_database_adapter INTERFACE ${ICTS_DATABASE}/adapter)

if(DEBUG_ICTS_SOURCE_DATABASE_ADAPTER)
  if(MSVC)
    target_compile_options(icts_source_database_adapter INTERFACE /Zi /Od /Ob0)
  else()
    target_compile_options(icts_source_database_adapter INTERFACE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
