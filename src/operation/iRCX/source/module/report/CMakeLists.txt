set(IRCX_REPORT_SRC
  SpefDumper.cc
  SpefDumperDNet.cc
  SpefDumperGeometry.cc
  SpefDumperHeader.cc
)

add_library(ircx_report ${IRCX_REPORT_SRC})

target_link_libraries(ircx_report
  PUBLIC
    ircx_database_design
    ircx_headers
  PRIVATE
    OpenMP::OpenMP_CXX
    ircx_config
    ircx_topology
    itf_data
    log
    usage
)

target_include_directories(ircx_report
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
