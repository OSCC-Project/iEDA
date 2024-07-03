add_subdirectory(utility)
add_subdirectory(config)
add_subdirectory(data_manager)
add_subdirectory(timing_engine)
add_subdirectory(module)

add_library(ito_source
    iTO.cpp
)

target_link_libraries(ito_source
    PUBLIC
        ista-engine
        ito_config
        ito_fix_drv
        ito_fix_hold
        ito_fix_setup
        ito_data_manager
        
)

target_include_directories(ito_source
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${HOME_OPERATION}/iTO/source/data
)