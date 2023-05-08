if(DEBUG_ITO_UTILITY)            
    message(STATUS "TO: DEBUG_ITO_UTILITY")       
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "TO: RELEASE_ITO_UTILITY")   
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(ito_utility 
    ${ITO_UTILITY_PATH}/DesignCalculator.cpp
    ${ITO_UTILITY_PATH}/Utility.cpp
    ${ITO_UTILITY_PATH}/GDSwriter.cpp
)

target_link_libraries(ito_utility 
    PRIVATE
        ito_api
        ito_source_external_libs
)

target_include_directories(ito_utility 
    PRIVATE
        ${ITO_UTILITY_PATH}
        ${ITO_PATH}/api
        ${HOME_DATABASE}/builder/builder
        ${ITO_DATA_PATH}
)
