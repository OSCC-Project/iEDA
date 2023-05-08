if(DEBUG_ICTS_COST_CALCULATOR)
  message(STATUS "CTS: DEBUG_ICTS_COST_CALCULATOR")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_COST_CALCULATOR")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(icts_cost_calculator INTERFACE)

target_link_libraries(icts_cost_calculator INTERFACE icts_timing_calculator
                                                     icts_api)

target_include_directories(icts_cost_calculator
                           INTERFACE ${ICTS_SOLVER}/costCalculator)
