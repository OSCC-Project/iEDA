add_subdirectory(geometry_boost)

add_library(solver_geometry 
    engine_geometry.cpp
    engine_geometry_creator.cpp
)

target_link_libraries(solver_geometry
    PUBLIC
    solver_geometry_boost   
)

target_include_directories(solver_geometry
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/geometry_boost
)
set(CMAKE_BUILD_TYPE "release")
# set(CMAKE_BUILD_TYPE "debug")