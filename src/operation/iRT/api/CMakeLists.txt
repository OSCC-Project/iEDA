if(DEBUG_IRT_API)            
    message(STATUS "RT: DEBUG_IRT_API")       
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "RT: RELEASE_IRT_API")   
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(irt_api
    ${IRT_API}/RTAPI.cpp
)

target_link_libraries(irt_api
    PRIVATE
        irt_source
        irt_api_external_libs
)

target_include_directories(irt_api
    PUBLIC
        ${IRT_API}   
)
