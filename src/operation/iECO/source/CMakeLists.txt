add_subdirectory(data)
add_subdirectory(data_manager)
add_subdirectory(eco_via)

add_library(ieco_src
ieco.cpp
)

target_link_libraries(ieco_src
    PUBLIC
    ieco_dm
    ieco_via
)

target_include_directories(ieco_src
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
