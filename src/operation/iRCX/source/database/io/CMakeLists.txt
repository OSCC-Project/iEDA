set(IRCX_DATABASE_IO_SRC
  IdbAdapter.cc
)

add_library(ircx_database_io ${IRCX_DATABASE_IO_SRC})

target_link_libraries(ircx_database_io
  PUBLIC
    IdbBuilder
    ircx_database_design
    ircx_headers
  PRIVATE
    def_service
    idb
    lef_service
    log
)

target_include_directories(ircx_database_io
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
