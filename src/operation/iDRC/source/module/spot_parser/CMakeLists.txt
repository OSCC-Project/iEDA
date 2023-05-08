add_library(idrc_spot_parser
    SpotParser.cpp

)

target_include_directories(idrc_spot_parser
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    
)

target_link_libraries(idrc_spot_parser
    PRIVATE
    idrc_db
    idrc_config
    idrc_shape_check
    idrc_spacing_check
    idrc_region_query

)