add_library(file_manager_base
    file_manager.cpp
)

target_include_directories(file_manager_base
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)