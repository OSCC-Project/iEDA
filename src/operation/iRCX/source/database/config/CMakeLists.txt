set(IRCX_CONFIG_SRC
  RCXConfig.cc
)

add_library(ircx_config ${IRCX_CONFIG_SRC})

target_link_libraries(ircx_config
  PUBLIC
    ircx_headers
  PRIVATE
    log
)

target_include_directories(ircx_config
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
