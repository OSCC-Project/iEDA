add_library(ieco_via
ieco_via.cpp
ieco_via_init.cpp
ieco_via_repair.cpp
)

target_link_libraries(ieco_via
    PUBLIC
    idm
    idb
    geometry_db
    solver_geometry
    solver_geometry_boost
    ieco_data
    ieco_dm

)

target_include_directories(ieco_via
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
