if(DEBUG_ICTS_CLUSTERING)
  message(STATUS "CTS: DEBUG_ICTS_CLUSTERING")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_CLUSTERING")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(icts_clustering INTERFACE)

target_link_libraries(icts_clustering INTERFACE icts_polygon)

target_include_directories(icts_clustering INTERFACE ${ICTS_SOLVER}/clustering)
