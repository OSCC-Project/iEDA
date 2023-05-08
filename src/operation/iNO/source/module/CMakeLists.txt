if(DEBUG_INO_MODULE)            
    message(STATUS "NO: DEBUG_INO_MODULE")       
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "NO: RELEASE_INO_MODULE")   
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(ino_module 
    ${INO_MODULE_PATH}/fix_fanout/FixFanout.cpp 
)

target_link_libraries(ino_module 
    PUBLIC
        ino_api
        ino_source
        ino_source_external_libs
)

target_include_directories(ino_module 
    PUBLIC
        ${INO_MODULE_PATH}
        ${INO_PATH}/api

        # module
        ${INO_SOURCE_PATH}/module/fix_fanout
        ${INO_SOURCE_PATH}/io

        ${INO_IO_PATH}
        ${HOME_DATABASE}/data/design
)
