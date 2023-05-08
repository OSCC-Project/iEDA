add_library(tool_api_irt
    irt_io.cpp
)

target_include_directories(tool_api_irt
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(tool_api_irt
    PRIVATE
)