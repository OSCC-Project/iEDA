add_library(file_manager_placement
    file_placement.cpp
)

target_link_libraries(file_manager_placement file_manager_base)

target_include_directories(file_manager_placement
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
