add_library(idrc_shape_check
    area/enclosed/EnclosedAreaCheck.cpp
    area/wire/RoutingAreaCheck.cpp
    enclosure/EnclosureCheck.cpp
    width/wire/RoutingWidthCheck.cpp
    minstep/MinStepCheck.cpp

)

target_include_directories(idrc_shape_check
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/area
    ${CMAKE_CURRENT_SOURCE_DIR}/area/enclosed
    ${CMAKE_CURRENT_SOURCE_DIR}/area/wire
    ${CMAKE_CURRENT_SOURCE_DIR}/enclosure
    ${CMAKE_CURRENT_SOURCE_DIR}/minstep
    ${CMAKE_CURRENT_SOURCE_DIR}/width
    ${CMAKE_CURRENT_SOURCE_DIR}/width/wire

    ${HOME_OPERATION}/iDRC/source/util
    ${HOME_OPERATION}/iDRC/source/data
    ${HOME_OPERATION}/iDRC/source/data/basic
    ${HOME_OPERATION}/iDRC/source/data/rule
)

target_link_libraries(idrc_shape_check
    PRIVATE

    idrc_db
    idrc_config
    idrc_region_query
    idrc_spacing_check

)