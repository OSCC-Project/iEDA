add_library(icts_source_module_timing ${ICTS_MODULE_TIMING}/TimingEngine.cc)

target_link_libraries(
  icts_source_module_timing
  PUBLIC
  icts_source_database_timing
  PRIVATE
  log
)

target_include_directories(
  icts_source_module_timing
  PUBLIC
  ${ICTS_MODULE_TIMING}
)

if(DEBUG_ICTS_SOURCE_MODULE_TIMING)
  if(MSVC)
    target_compile_options(icts_source_module_timing PRIVATE /Zi /Od /Ob0)
    target_link_options(icts_source_module_timing PRIVATE /DEBUG)
  else()
    target_compile_options(icts_source_module_timing PRIVATE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
