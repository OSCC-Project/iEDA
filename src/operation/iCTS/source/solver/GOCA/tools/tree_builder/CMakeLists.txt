add_subdirectory(${ICTS_SOLVER}/GOCA/tools/tree_builder/bound_skew_tree)
add_subdirectory(${ICTS_SOLVER}/GOCA/tools/tree_builder/local_legalization)

if(DEBUG_ICTS_GOCA_TREE_BUILDER)
  message(STATUS "CTS: DEBUG_ICTS_GOCA_TREE_BUILDER")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_GOCA_TREE_BUILDER")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(icts_goca_tree_builder
            ${ICTS_SOLVER}/GOCA/tools/tree_builder/TreeBuilder.cc)

target_link_libraries(
  icts_goca_tree_builder
  PUBLIC icts_data_manager icts_api icts_goca_database icts_bst
         icts_local_legalization icts_goca_timing_propagator)

target_include_directories(
  icts_goca_tree_builder
  PUBLIC ${ICTS_SOLVER}/GOCA/database
         ${ICTS_SOLVER}/GOCA/tools/tree_builder/bound_skew_tree
          ${ICTS_SOLVER}/GOCA/tools/tree_builder/local_legalization
         ${ICTS_SOLVER}/GOCA/tools/timing_propagator)
