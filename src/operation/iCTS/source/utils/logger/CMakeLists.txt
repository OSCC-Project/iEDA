add_library(
  icts_source_utils_logger
  ${ICTS_LOGGER}/Schema.cc
  ${ICTS_LOGGER}/SchemaScope.cc
)

target_include_directories(
  icts_source_utils_logger
  PUBLIC
  ${ICTS_UTILS}
  ${ICTS_LOGGER}
)

target_link_libraries(
  icts_source_utils_logger
  PRIVATE
  log
  usage
)

if(DEBUG_ICTS_SOURCE_UTILS_LOGGER)
  if(MSVC)
    target_compile_options(icts_source_utils_logger PRIVATE /Zi /Od /Ob0)
    target_link_options(icts_source_utils_logger PRIVATE /DEBUG)
  else()
    target_compile_options(icts_source_utils_logger PRIVATE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
