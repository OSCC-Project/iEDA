if(DEBUG_INO_IO)            
    message(STATUS "NO: DEBUG_INO_IO")       
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "NO: RELEASE_INO_IO")   
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(ino_io
    ${INO_IO_PATH}/DbInterface.cpp
    ${INO_IO_PATH}/JsonParser.cpp
    ${INO_IO_PATH}/Reporter.cpp
)

target_link_libraries(ino_io
    PRIVATE
        ino_api
        ino_source_external_libs
        ino_api_external_libs
        ino_source
        idm
        idb
)

target_include_directories(ino_io 
    PRIVATE
        ${INO_IO_PATH}
        ${INO_PATH}/api
        ${HOME_THIRDPARTY}/json
)
