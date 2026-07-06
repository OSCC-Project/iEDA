add_library(ircx_flow_extraction Extraction.cc)

target_link_libraries(ircx_flow_extraction
  PUBLIC
    ircx_calculate
    ircx_database_design
    ircx_environment
    ircx_headers
    ircx_process
    ircx_topology
  PRIVATE
    log
)

target_include_directories(ircx_flow_extraction
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
