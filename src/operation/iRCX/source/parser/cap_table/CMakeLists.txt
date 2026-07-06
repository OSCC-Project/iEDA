add_library(ircx_cap_table CapTable.cpp)

target_link_libraries(ircx_cap_table
  PRIVATE
    log
)

target_include_directories(ircx_cap_table
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
