if(DEBUG_ITO_CONFIG)
  message(STATUS "TO: DEBUG_ITO_CONFIG")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "TO: RELEASE_ITO_CONFIG")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(ito_config
JsonParser.cpp
ToConfig.cpp
)

target_link_libraries(ito_config
    PUBLIC
    idm
)

target_include_directories(ito_config 
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${HOME_UTILITY}/json
        ${HOME_THIRDPARTY}/json
)
