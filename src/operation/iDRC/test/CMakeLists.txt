SET(CMAKE_BUILD_TYPE "Debug")

ADD_EXECUTABLE(run_drc ${HOME_OPERATION}/iDRC/test/test_edge_enclosure.cpp)

target_include_directories(run_drc
    PUBLIC
    ${HOME_OPERATION}/iDRC/source
    ${HOME_OPERATION}/iDRC/api
    ${HOME_OPERATION}/iDRC/source/module/shape_check/area/wire
    ${HOME_OPERATION}/iDRC/source/module/spacing_check/cut/eol

    ${HOME_OPERATION}/iDRC/source/data/basic
    ${HOME_OPERATION}/iDRC/source/data/rule
    ${HOME_OPERATION}/iDRC/source/config
    ${HOME_OPERATION}/iDRC/source/module/region_query



    

    ${HOME_OPERATION}/iDRC/source/util
    # ${HOME_OPERATION}/iDRC/data/basic

)

target_link_libraries(run_drc
    PRIVATE
    idrc_src
    idrc_api
)

ADD_EXECUTABLE(test_eol ${HOME_OPERATION}/iDRC/test/test_eol.cpp)

target_include_directories(test_eol
    PUBLIC
    ${HOME_OPERATION}/iDRC/source
    ${HOME_OPERATION}/iDRC/api
    ${HOME_OPERATION}/iDRC/source/data
    ${HOME_OPERATION}/iDRC/source/data/basic
    ${HOME_OPERATION}/iDRC/source/config
    ${HOME_OPERATION}/iDRC/source/data/rule
    ${HOME_OPERATION}/iDRC/source/module/region_query
    ${HOME_OPERATION}/iDRC/source/module/spacing_check/wire/eol
    ${HOME_OPERATION}/iDRC/source/module/spacing_check/cut/eol
    ${HOME_OPERATION}/iDRC/source/module/spot_parser
    

    ${HOME_OPERATION}/iDRC/source/util
    # ${HOME_OPERATION}/iDRC/data/basic

)

target_link_libraries(test_eol
    PRIVATE
    idrc_src
    idrc_api
)



