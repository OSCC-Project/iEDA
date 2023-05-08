if(DEBUG_ITO_SOURCE)            
    message(STATUS "TO: DEBUG_ITO_SOURCE")       
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "TO: RELEASE_ITO_SOURCE")   
    set(CMAKE_BUILD_TYPE "Release")
endif()


# build
ADD_SUBDIRECTORY(${ITO_DATA_PATH})
ADD_SUBDIRECTORY(${ITO_IO_PATH})
ADD_SUBDIRECTORY(${ITO_MODULE_PATH})
ADD_SUBDIRECTORY(${ITO_UTILITY_PATH})

set(ITO_SOURCE_HEADER
    ${ITO_SOURCE_PATH}
    # config
    ${ITO_SOURCE_PATH}/config

    # database
    ${ITO_SOURCE_PATH}/data

    # io
    ${ITO_SOURCE_PATH}/io

    # module
    ${ITO_SOURCE_PATH}/module
    ${ITO_SOURCE_PATH}/module/evaluator
    ${ITO_SOURCE_PATH}/module/fix_drv
    ${ITO_SOURCE_PATH}/module/fix_hold
    ${ITO_SOURCE_PATH}/module/fix_setup
    ${ITO_SOURCE_PATH}/module/placer
    
    # tcl
    # ${ITO_SOURCE_PATH}/tcl

    # utility
    ${ITO_SOURCE_PATH}/utility
)

add_library(ito_source
    ${ITO_SOURCE_PATH}/iTO.cpp
)

target_link_libraries(ito_source
    PUBLIC
        ito_io
        ito_data
        ito_module
        ito_utility
        ito_api_external_libs
)

target_include_directories(ito_source
    PUBLIC
        ${ITO_SOURCE_HEADER}
        ${ITO_API_PATH}
)