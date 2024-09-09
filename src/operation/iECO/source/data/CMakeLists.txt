add_library(ieco_data
    ieco_data.cpp
    ieco_data_via.cpp
)

target_link_libraries(ieco_data
    PUBLIC
    idb
    solver_geometry
    solver_geometry_boost
)

target_include_directories(ieco_data
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
