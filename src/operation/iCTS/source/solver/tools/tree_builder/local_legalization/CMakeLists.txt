if(DEBUG_ICTS_LOCAL_LEGALIZATION)
  message(STATUS "CTS: DEBUG_ICTS_LOCAL_LEGALIZATION")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_LOCAL_LEGALIZATION")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(
  icts_local_legalization
  ${ICTS_SOLVER}/tools/tree_builder/local_legalization/LocalLegalization.cc)

target_link_libraries(icts_local_legalization PUBLIC icts_solver_database)

target_include_directories(icts_local_legalization
                           PUBLIC ${ICTS_SOLVER}/database)
