if(DEBUG_INO_API)            
    message(STATUS "NO: DEBUG_INO_API")       
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "NO: RELEASE_INO_API")   
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(ino_api
    ${INO_API_PATH}/NoApi.cpp
)

target_link_libraries(ino_api
    PRIVATE
        ino_source
        ino_api_external_libs
)

target_include_directories(ino_api
    PUBLIC
        ${INO_API_PATH}
)
