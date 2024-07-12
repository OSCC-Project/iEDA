
if(DEBUG_ITO_TREE_BUILD_SLT)
  message(STATUS "TO: DEBUG_ITO_TREE_BUILD_SLT")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "TO: RELEASE_ITO_TREE_BUILD_SLT")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(ito_shallow_light_tree
ShallowLightTree.cpp
)

target_link_libraries(ito_shallow_light_tree
    PUBLIC
    ito_data_manager
    salt
)

target_include_directories(ito_shallow_light_tree
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_OPERATION}/iTO/source/data

    ${PROJECT_SOURCE_DIR}/src/third_party/salt
)