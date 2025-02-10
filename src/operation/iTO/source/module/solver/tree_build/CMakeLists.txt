if(DEBUG_ITO_TREE_BUILD)
  message(STATUS "TO: DEBUG_ITO_TREE_BUILD")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "TO: RELEASE_ITO_TREE_BUILD")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_subdirectory(HV_tree)
add_subdirectory(rsmt_flute)
add_subdirectory(shallow_light_tree)


add_library(ito_tree_build
TreeBuild.cpp
)

target_link_libraries(ito_tree_build
    PUBLIC
    ito_utility
    ito_timing_engine
    ito_config
    # ito_eval
    ito_data_manager
    ito_fix_drv
    ito_placer
    ito_hv_tree
    ito_flute_tree
    ito_shallow_light_tree
)

target_include_directories(ito_tree_build
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_OPERATION}/iTO/source/data
)

