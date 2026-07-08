add_library(
  icts_source_database_io
  ${ICTS_DATABASE_IO}/Wrapper.cc
  ${ICTS_DATABASE_IO}/WrapperLiberty.cc
  ${ICTS_DATABASE_IO}/WrapperRc.cc
  ${ICTS_DATABASE_IO}/WrapperClockReader.cc
  ${ICTS_DATABASE_IO}/WrapperClockWriter.cc
)

target_include_directories(
  icts_source_database_io
  PUBLIC
  ${ICTS_DATABASE_IO}
  ${ICTS_DATABASE}
  PRIVATE
  ${HOME_DATABASE}/data/design
  ${HOME_DATABASE}/data/design/db_design
  ${HOME_DATABASE}/data/design/db_layout
  ${HOME_DATABASE}/manager/builder
  ${HOME_UTILITY}/stdBase/include
)

target_link_libraries(
  icts_source_database_io
  PUBLIC
  icts_source_database_spatial
  icts_source_database_routing
  PRIVATE
  idm
  icts_source_database_adapter_sdc
  icts_source_database_design
  icts_source_utils_logger
  liberty
  log
)

if(DEBUG_ICTS_SOURCE_DATABASE_IO)
  if(MSVC)
    target_compile_options(icts_source_database_io PRIVATE /Zi /Od /Ob0)
    target_link_options(icts_source_database_io PRIVATE /DEBUG)
  else()
    target_compile_options(icts_source_database_io PRIVATE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
