if(DEBUG_ICTS_MIN_COST_FLOW)
  message(STATUS "CTS: DEBUG_ICTS_MIN_COST_FLOW")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_MIN_COST_FLOW")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(icts_min_cost_flow INTERFACE)

target_link_libraries(icts_min_cost_flow INTERFACE lemon)