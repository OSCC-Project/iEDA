add_library(ircx_mapping MappingBuilder.cpp)

target_link_libraries(ircx_mapping
  PRIVATE
    log
)

target_include_directories(ircx_mapping
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
