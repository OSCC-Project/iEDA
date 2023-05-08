add_subdirectory(config)
add_subdirectory(data)
add_subdirectory(module)
add_subdirectory(util)

add_library(idrc_src
    DRC.cpp
    DRCInit.cpp
)

target_include_directories(idrc_src
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}

    ${HOME_OPERATION}/iDRC/source/util
    # ${HOME_OPERATION}/iDRC/data/basic

)

target_link_libraries(idrc_src
    PRIVATE
    idrc_config
    idrc_db

    idrc_multi_pattern
    idrc_region_query
    idrc_shape_check
    idrc_spacing_check
    idrc_spot_parser

)
