add_library(ito_data_manager
data_manager.cpp
data_manager_calculate.cpp
)

target_link_libraries(ito_data_manager
PUBLIC
        ito_config
        idm
        idb
        ito_utility
)

target_include_directories(ito_data_manager 
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${HOME_OPERATION}/iTO/source/data
        ${HOME_OPERATION}/iTO/source/utility
)
