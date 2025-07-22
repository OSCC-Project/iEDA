add_library(solver_geometry_boost 
    geometry_boost.cpp
)

target_link_libraries(solver_geometry_boost
    PUBLIC
        
)

target_include_directories(solver_geometry_boost
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_DATABASE}/interaction
)
