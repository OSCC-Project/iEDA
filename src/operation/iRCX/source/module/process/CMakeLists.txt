add_library(ircx_process ProcessVariation.cc)

target_link_libraries(ircx_process
  PUBLIC
    ircx_database_design
    ircx_headers
  PRIVATE
    itf_data
    OpenMP::OpenMP_CXX
    usage
)

target_include_directories(ircx_process
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
