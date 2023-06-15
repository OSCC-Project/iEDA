if(DEBUG_ICTS_GOCA_TREE_BUILDER)
  message(STATUS "CTS: DEBUG_ICTS_GOCA_TREE_BUILDER")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_GOCA_TREE_BUILDER")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(icts_goca_tree_builder
            ${ICTS_SOLVER}/GOCA/tools/treeBuilder/TreeBuilder.cc)

target_link_libraries(icts_goca_tree_builder PUBLIC icts_data_manager icts_api
                                                    icts_goca_timing_propagator)

target_include_directories(
  icts_goca_tree_builder PUBLIC ${ICTS_SOLVER}/GOCA/database
                                ${ICTS_SOLVER}/GOCA/tools/timingPropagator)
