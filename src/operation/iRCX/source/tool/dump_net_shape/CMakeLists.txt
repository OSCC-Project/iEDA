add_library(ircx_tool_dump_net_shape
  DumpNetShapeTool.cc
)

target_link_libraries(ircx_tool_dump_net_shape
  PUBLIC
    ircx_headers
  PRIVATE
    log
    OpenMP::OpenMP_CXX
)

target_include_directories(ircx_tool_dump_net_shape
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
