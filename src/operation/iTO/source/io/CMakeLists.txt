if(DEBUG_ITO_IO)            
    message(STATUS "TO: DEBUG_ITO_IO")       
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "TO: RELEASE_ITO_IO")   
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(ito_io
    ${ITO_IO_PATH}/DbInterface.cpp
    ${ITO_IO_PATH}/JsonParser.cpp
    ${ITO_IO_PATH}/Reporter.cpp
)

target_link_libraries(ito_io
    PRIVATE
        ito_api
        ito_source_external_libs
        ito_api_external_libs
)

target_include_directories(ito_io 
    PRIVATE
        ${ITO_IO_PATH}
        ${ITO_PATH}/api
        ${HOME_THIRDPARTY}/json
)
