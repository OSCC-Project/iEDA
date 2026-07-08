set(ICTS_DATABASE_ADAPTER ${ICTS_DATABASE}/adapter)
set(ICTS_DATABASE_CONFIG ${ICTS_DATABASE}/config)
set(ICTS_DATABASE_DESIGN ${ICTS_DATABASE}/design)
set(ICTS_DATABASE_IO ${ICTS_DATABASE}/io)
set(ICTS_DATABASE_SPATIAL ${ICTS_DATABASE}/spatial)
set(ICTS_DATABASE_ROUTING ${ICTS_DATABASE}/routing)
set(ICTS_DATABASE_TIMING ${ICTS_DATABASE}/timing)
set(ICTS_DATABASE_CHARACTERIZATION ${ICTS_DATABASE}/characterization)
set(ICTS_DATABASE_QOR ${ICTS_DATABASE}/qor)

# build
add_subdirectory(${ICTS_DATABASE_ADAPTER})
add_subdirectory(${ICTS_DATABASE_CONFIG})
add_subdirectory(${ICTS_DATABASE_DESIGN})
add_subdirectory(${ICTS_DATABASE_IO})
add_subdirectory(${ICTS_DATABASE_SPATIAL})
add_subdirectory(${ICTS_DATABASE_ROUTING})
add_subdirectory(${ICTS_DATABASE_TIMING})
add_subdirectory(${ICTS_DATABASE_CHARACTERIZATION})
add_subdirectory(${ICTS_DATABASE_QOR})

add_library(icts_source_database INTERFACE)

target_link_libraries(
  icts_source_database
  INTERFACE
  icts_source_database_adapter
  icts_source_database_config
  icts_source_database_design
  icts_source_database_io
  icts_source_database_spatial
  icts_source_database_routing
  icts_source_database_timing
  icts_source_database_characterization
  icts_source_database_qor
)

target_include_directories(
  icts_source_database
  INTERFACE
  ${ICTS_DATABASE}
  ${ICTS_DATABASE_ADAPTER}
  ${ICTS_DATABASE_CONFIG}
  ${ICTS_DATABASE_DESIGN}
  ${ICTS_DATABASE_IO}
  ${ICTS_DATABASE_SPATIAL}
  ${ICTS_DATABASE_ROUTING}
  ${ICTS_DATABASE_TIMING}
  ${ICTS_DATABASE_CHARACTERIZATION}
  ${ICTS_DATABASE_QOR}
)

if(DEBUG_ICTS_SOURCE_DATABASE)
  if(MSVC)
    target_compile_options(icts_source_database INTERFACE /Zi /Od /Ob0)
  else()
    target_compile_options(icts_source_database INTERFACE -g3 -O0 -fno-inline -fno-omit-frame-pointer)
  endif()
endif()
