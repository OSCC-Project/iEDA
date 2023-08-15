if(DEBUG_ICTS_GOCA)
  message(STATUS "CTS: DEBUG_ICTS_GOCA")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_GOCA")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_subdirectory(${ICTS_SOLVER}/GOCA/database)
add_subdirectory(${ICTS_SOLVER}/GOCA/tools)

add_library(icts_goca ${ICTS_SOLVER}/GOCA/GOCA.cc)

target_link_libraries(icts_goca PUBLIC icts_database icts_api
                                       icts_goca_database icts_goca_tools)

target_include_directories(
  icts_goca PUBLIC ${ICTS_SOLVER}/GOCA ${ICTS_SOLVER}/GOCA/database
                   ${ICTS_SOLVER}/GOCA/tools)
