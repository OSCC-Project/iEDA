add_library(ito_placer
Placer.cpp
)

target_link_libraries(ito_placer
    PUBLIC
    ito_utility
    ito_config
    ito_eval
    ito_data_manager
    idb
    idm
)

target_include_directories(ito_placer
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_OPERATION}/iTO/source/data
)


