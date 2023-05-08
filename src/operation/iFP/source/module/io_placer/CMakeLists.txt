
add_library(ifp_io_placer
    io_placer.cpp
)

target_link_libraries(ifp_io_placer 
    PUBLIC
    idm
)

target_include_directories(ifp_io_placer 
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_OPERATION}/iFP/source/data
)
