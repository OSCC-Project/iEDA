if(DEBUG_ITO_API)            
    message(STATUS "TO: DEBUG_ITO_API")       
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "TO: RELEASE_ITO_API")   
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(ito_api
    ${ITO_API_PATH}/ToApi.cpp
)

target_link_libraries(ito_api
    PRIVATE
        ito_source
        ito_api_external_libs
)

target_include_directories(ito_api
    PUBLIC
        ${ITO_API_PATH}
)
