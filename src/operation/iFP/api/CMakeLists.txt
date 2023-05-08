add_library(ifp_api
    ifp_api.cpp

)

target_include_directories(ifp_api
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_OPERATION}/iFP/source/config
    ${HOME_OPERATION}/iFP/source/data
    ${HOME_OPERATION}/iFP/source/module/init_design
    ${HOME_OPERATION}/iFP/source/module/io_placer
    ${HOME_OPERATION}/iFP/source/module/tap_cell
    
)

target_link_libraries(ifp_api
    PRIVATE
        ifp_init
        ifp_io_placer
        ifp_tapcell
)