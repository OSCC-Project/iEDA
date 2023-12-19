set(HOME_ISTA ${HOME_OPERATION}/iSTA)

add_library(eval_ista_libs INTERFACE)

target_link_libraries(eval_ista_libs
    INTERFACE
        liberty
        graph
        verilog-parser
        sta
        delay
        ista-engine
        sdc
        sdc-cmd
        utility
        tcl
        str
        time
        pthread
        stdc++fs
        fort
)

target_include_directories(eval_ista_libs
    INTERFACE
    ${HOME_UTILITY}/stdBase
    ${HOME_UTILITY}/stdBase/include
    ${HOME_UTILITY}
    ${HOME_OPERATION}/iSTA
    ${HOME_OPERATION}/iSTA/source
    ${HOME_OPERATION}/iSTA/source/module
    ${HOME_OPERATION}/iSTA/source/module/include
)

target_include_directories(eval_ista_libs 
    SYSTEM INTERFACE
    ${HOME_OPERATION}/iSTA/source/third-party
    ${HOME_THIRDPARTY}
)

target_link_directories(eval_ista_libs
    INTERFACE
        ${CMAKE_BINARY_DIR}/lib
)

find_package(Eigen3 REQUIRED)
target_include_directories(eval_ista_libs
    SYSTEM INTERFACE 
    ${EIGEN3_INCLUDE_DIR}
)
message(STATUS "Eigen3 ${EIGEN3_INCLUDE_DIR}")

target_link_directories(eval_ista_libs
    INTERFACE
        ${EIGEN3_LIBRARY}
)
