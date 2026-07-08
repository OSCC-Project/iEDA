add_library(icts_source_database_config ${ICTS_DATABASE_CONFIG}/Config.cc)

target_link_libraries(
  icts_source_database_config
  PUBLIC
  icts_source_utils_logger
  PRIVATE
  log
)

target_include_directories(
  icts_source_database_config
  BEFORE
  PUBLIC
  ${ICTS_DATABASE_CONFIG}
  ${ICTS_DATABASE}
  PRIVATE
  ${HOME_THIRDPARTY}/json
)

if(DEBUG_ICTS_SOURCE_DATABASE_CONFIG)
  if(MSVC)
    target_compile_options(icts_source_database_config PRIVATE /Zi /Od /Ob0)
    target_link_options(icts_source_database_config PRIVATE /DEBUG)
  else()
    target_compile_options(icts_source_database_config PRIVATE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
