add_subdirectory(data_manager)
add_subdirectory(graph)
add_subdirectory(layout)

add_library(large_model
large_model.cpp
)

target_link_libraries(large_model
    PUBLIC
    ilm_dm
)

target_include_directories(large_model
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)