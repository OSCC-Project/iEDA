if(DEBUG_ICTS_BALANCER)
  message(STATUS "CTS: DEBUG_ICTS_BALANCER")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_BALANCER")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(icts_balancer ${ICTS_MODULE}/balancer/Balancer.cc)

target_link_libraries(icts_balancer PUBLIC icts_data_manager icts_module)

target_include_directories(icts_balancer PUBLIC ${ICTS_MODULE}/balancer)
