add_subdirectory(data_manager)
add_subdirectory(graph)
add_subdirectory(layout)
add_subdirectory(patch)
add_subdirectory(feature)
add_subdirectory(utility)

add_library(large_model
large_model.cpp
)

target_link_libraries(large_model
    PUBLIC
    log
    idm
    ilm_dm
    ilm_feature
    memory
)

target_include_directories(large_model
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)