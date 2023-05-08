if(DEBUG_ICTS_CONFIG)
  message(STATUS "CTS: DEBUG_ICTS_CONFIG")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_CONFIG")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(icts_config ${ICTS_DATA_MANAGER}/config/JsonParser.cc)

target_link_libraries(icts_config PUBLIC icts_data_manager)

target_include_directories(icts_config PUBLIC ${ICTS_DATA_MANAGER}/config)
