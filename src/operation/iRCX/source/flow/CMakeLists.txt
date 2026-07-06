set(IRCX_FLOW_SRC
  setup/Setup.cc
)

add_library(ircx_flow ${IRCX_FLOW_SRC})

target_link_libraries(ircx_flow
  PUBLIC
    ircx_cap_table
    ircx_config
    ircx_database_design
    ircx_headers
    ircx_mapping
  PRIVATE
    idm
    ircx_database_io
    itf_builder
    log
)

target_include_directories(ircx_flow
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
