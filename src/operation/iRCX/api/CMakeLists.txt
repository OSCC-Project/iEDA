set(IRCX_API_SRC
  RCXAPI.cc
)

add_library(ircx_api ${IRCX_API_SRC})

target_link_libraries(ircx_api
  PUBLIC
    ircx_tool_compare_spef
    ircx_tool_dump_net_shape
    ircx_tool_plot_spef
  PRIVATE
    ircx_config
    ircx_database_design
    ircx_flow
    ircx_flow_extraction
    ircx_flow_report
    log
    OpenMP::OpenMP_CXX
)

target_include_directories(ircx_api
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
