if(DEBUG_IRT_DRC_ENGINE)
    message(STATUS "RT: DEBUG_IRT_DRC_ENGINE")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: RELEASE_IRT_DRC_ENGINE")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(irt_drc_engine
    ## module drc_engine
    ${IRT_MODULE}/drc_engine/DRCEngine.cpp
)

target_link_libraries(irt_drc_engine
    PUBLIC
    irt_data_manager
    irt_module
    irt_toolkit
)

target_include_directories(irt_drc_engine
    PUBLIC
    ${IRT_MODULE}/drc_engine/de_data_manager
    ${IRT_MODULE}/drc_engine
)
