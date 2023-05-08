add_library(idrc_db
    DrcIDBWrapper.cpp
    IDRWrapper.cpp

    basic/DrcNet.cpp
    basic/DrcConflictGraph.cpp
    basic/Tech.cpp
    basic/DrcLayer.cpp
)

target_include_directories(idrc_db
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/rule
    ${CMAKE_CURRENT_SOURCE_DIR}/basic
    ${HOME_OPERATION}/iDRC/source/util
    ${HOME_OPERATION}/iDRC/source/config


)

target_link_libraries(idrc_db
    PRIVATE
    idrc_config
    idrc_region_query
)