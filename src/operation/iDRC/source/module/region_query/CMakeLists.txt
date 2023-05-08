add_library(idrc_region_query
    RegionQuery.cpp

)

target_include_directories(idrc_region_query
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_OPERATION}/iDRC/source/util
    ${HOME_OPERATION}/iDRC/source
    ${HOME_OPERATION}/iDRC/source/module/spacing_check/corner_fill
    ${HOME_OPERATION}/iDRC/source/module/spacing_check/wire/eol



)

target_link_libraries(idrc_region_query
    PRIVATE
    idrc_db
    idrc_config
    idrc_spacing_check
)
