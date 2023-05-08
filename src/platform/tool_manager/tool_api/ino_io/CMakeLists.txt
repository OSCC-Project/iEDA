add_library(tool_api_ino
    ino_io.cpp
)

target_include_directories(tool_api_ino
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(tool_api_ino
    PRIVATE
    ino_source
)