add_library(solver_lg_method
    abacus/AbacusCluster.cc
    abacus/Abacus.cc
    customization/LGCustomization.cc
)

target_link_libraries(solver_lg_method 
    PRIVATE
    usage
    ipl-module-logger
    ipl-bridge
    ipl-configurator
)

target_include_directories(solver_lg_method 
    PUBLIC 
        ${HOME_OPERATION}/iPL/source
        ${HOME_OPERATION}/iPL/source/module/legalizer/config
        ${HOME_OPERATION}/iPL/source/module/legalizer/database

        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/abacus
        ${CMAKE_CURRENT_SOURCE_DIR}/customization
        
)