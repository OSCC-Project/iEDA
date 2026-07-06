set(IRCX_DATABASE_DESIGN_SRC
  RCXData.cc
  TopoPool.cc
)

add_library(ircx_database_design ${IRCX_DATABASE_DESIGN_SRC})

target_link_libraries(ircx_database_design
  PUBLIC
    ircx_cap_table
    ircx_headers
    ircx_mapping
  PRIVATE
    itf_data
    log
)

target_include_directories(ircx_database_design
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
