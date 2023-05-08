if(DEBUG_ITO_DATA)            
    message(STATUS "TO: DEBUG_ITO_DATA")       
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "TO: RELEASE_ITO_DATA")   
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(ito_data
    ${ITO_DATA_PATH}/Inst.cpp
    ${ITO_DATA_PATH}/Layout.cpp
    ${ITO_DATA_PATH}/Master.cpp
)

target_link_libraries(ito_data 
        ito_api
        ito_source_external_libs
        ito_utility
)

target_include_directories(ito_data 
    INTERFACE
        ${HOME_DATABASE}/data/design
        ${HOME_DATABASE}/data/design/db_design
        ${HOME_DATABASE}/data/design/db_layout
        ${ITO_DATA_PATH}
        ${ITO_UTILITY}
)
