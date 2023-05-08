if(DEBUG_ICTS_DME)
  message(STATUS "CTS: DEBUG_ICTS_DME")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_DME")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(icts_dme INTERFACE)

target_link_libraries(icts_dme INTERFACE icts_polygon)

target_include_directories(
  icts_dme INTERFACE ${ICTS_SOLVER}/DME/DMEbase ${ICTS_SOLVER}/DME/ZST
                     ${ICTS_SOLVER}/DME/BST ${ICTS_SOLVER}/DME/UST)
