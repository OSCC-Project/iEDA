add_library(tool_api_ieval
    ieval_io.cpp
)

target_include_directories(tool_api_ieval
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(tool_api_ieval
    PRIVATE
    idm
    # eval_data
    # eval_wirelength
    # eval_timing
)