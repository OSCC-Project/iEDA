add_library(idrc_spacing_check
    ${CMAKE_CURRENT_SOURCE_DIR}/cut/common/CutSpacingCheck.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cut/eol/CutEolSpacingCheck.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wire/common/RoutingSpacingCheck.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wire/eol/EOLSpacingCheck.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wire/notch/NotchSpacingCheck.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wire/jog/JogSpacingCheck.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/corner_fill/CornerFillSpacingCheck.cpp


)

target_include_directories(idrc_spacing_check
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cut/common
    ${CMAKE_CURRENT_SOURCE_DIR}/cut/eol
    ${CMAKE_CURRENT_SOURCE_DIR}/wire
    ${CMAKE_CURRENT_SOURCE_DIR}/wire/common
    ${CMAKE_CURRENT_SOURCE_DIR}/wire/eol
    ${CMAKE_CURRENT_SOURCE_DIR}/wire/notch
    ${CMAKE_CURRENT_SOURCE_DIR}/wire/jog
    ${CMAKE_CURRENT_SOURCE_DIR}/corner_fill



    ${HOME_OPERATION}/iDRC/source/data/basic
    ${HOME_OPERATION}/iDRC/source/data/rule

    ${HOME_OPERATION}/iDRC/source/util
)

target_link_libraries(idrc_spacing_check
    PRIVATE
    idrc_db
    idrc_region_query

)
