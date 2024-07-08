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
        ${HOME_PLATFORM}/data_manager
        ${HOME_DATABASE}/data/design
        ${HOME_DATABASE}/manager/builder
        ${HOME_DATABASE}/manager/builder/def_builder
        ${HOME_DATABASE}/manager/builder/lef_builder
        ${HOME_DATABASE}/manager/builder/gds_builder
        ${HOME_DATABASE}/manager/builder/json_builder
        ${HOME_DATABASE}/manager/builder/verilog_builder
        ${HOME_DATABASE}/manager/service/def_service
        ${HOME_DATABASE}/manager/service/lef_service
        ${HOME_DATABASE}/manager/parser/gdsii
        ${HOME_DATABASE}/manager/parser/json
        ${HOME_DATABASE}/manager/parser/
        ${HOME_THIRDPARTY}/lefdef/def/def/
        ${HOME_THIRDPARTY}/lefdef/lef
)
