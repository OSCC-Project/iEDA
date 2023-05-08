if(DEBUG_ICTS_SLEW_AWARE)
  message(STATUS "CTS: DEBUG_ICTS_SLEW_AWARE")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_SLEW_AWARE")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(icts_slew_aware ${ICTS_SOLVER}/slewAware/SlewAware.cc)

target_link_libraries(
  icts_slew_aware PUBLIC icts_data_manager icts_clustering
                         icts_timing_calculator icts_cost_calculator icts_api)

target_include_directories(
  icts_slew_aware PUBLIC ${ICTS_SOLVER}/slewAware ${ICTS_SOLVER}/clustering
                         ${ICTS_DATABASE})
