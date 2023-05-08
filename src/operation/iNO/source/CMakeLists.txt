if(DEBUG_INO_SOURCE)            
    message(STATUS "NO: DEBUG_INO_SOURCE")       
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "NO: RELEASE_INO_SOURCE")   
    set(CMAKE_BUILD_TYPE "Release")
endif()


# build
ADD_SUBDIRECTORY(${INO_IO_PATH})
ADD_SUBDIRECTORY(${INO_MODULE_PATH})

set(INO_SOURCE_HEADER
    ${INO_SOURCE_PATH}
    # config
    ${INO_SOURCE_PATH}/config

    # io
    ${INO_SOURCE_PATH}/io

    # module
    ${INO_SOURCE_PATH}/module
    ${INO_SOURCE_PATH}/module/fix_fanout
)

add_library(ino_source
    ${INO_SOURCE_PATH}/iNO.cpp
)

target_link_libraries(ino_source
    PUBLIC
        ino_io
        ino_module
        ino_api_external_libs
)

target_include_directories(ino_source
    PUBLIC
        ${INO_SOURCE_HEADER}
        ${INO_API_PATH}
)