add_library(ifp_tapcell
    tapcell.cpp
)

target_link_libraries(ifp_tapcell 
    PUBLIC
    idm
)

target_include_directories(ifp_tapcell 
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)