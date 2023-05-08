add_library(idrc_api
    DrcAPI.cpp

)

target_include_directories(idrc_api
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_OPERATION}/iDRC/source
    ${HOME_OPERATION}/iDRC/source/data/basic
    
    
)

target_link_libraries(idrc_api
    PUBLIC
        idrc_db
    PRIVATE
        idrc_config
        idrc_src
        idrc_api_external_lib
        idrc_multi_pattern
        idrc_region_query
        idrc_shape_check
        idrc_spacing_check
        idrc_spot_parser
)