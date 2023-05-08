add_library(file_manager_cts
    file_cts.cpp
)

target_link_libraries(file_manager_cts file_manager_base)

target_include_directories(file_manager_cts
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)