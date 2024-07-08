
if(DEBUG_ITO_TREE_BUILD_FLUTE)
  message(STATUS "TO: DEBUG_ITO_TREE_BUILD_FLUTE")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "TO: RELEASE_ITO_TREE_BUILD_FLUTE")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(ito_flute_tree
FLUTETree.cpp
)

target_link_libraries(ito_flute_tree
    PUBLIC
    ito_data_manager
)

target_include_directories(ito_flute_tree
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_OPERATION}/iTO/source/data
)