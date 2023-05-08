add_library(idrc_config
    DrcConfigurator.cpp

)

target_include_directories(idrc_config
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}

    ${HOME_OPERATION}/iDRC/source/util
)

target_link_libraries(idrc_config
    PRIVATE

)