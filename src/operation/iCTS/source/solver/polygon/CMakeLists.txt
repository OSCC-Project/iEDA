if(DEBUG_ICTS_POLYGON)
  message(STATUS "CTS: DEBUG_ICTS_POLYGON")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_POLYGON")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(icts_polygon INTERFACE)

target_link_libraries(icts_polygon INTERFACE icts_solver)

target_include_directories(icts_polygon INTERFACE ${ICTS_SOLVER}/polygon)
