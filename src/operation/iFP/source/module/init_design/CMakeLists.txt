
add_library(ifp_init
    init_design.cpp
)

target_link_libraries(ifp_init 
    PUBLIC
    idm
)

target_include_directories(ifp_init 
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
