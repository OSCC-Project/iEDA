if(DEBUG_ITO_EVALUATOR)
  message(STATUS "TO: DEBUG_ITO_EVALUATOR")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "TO: RELEASE_ITO_EVALUATOR")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(ito_eval
EstimateParasitics.cpp
# RoutingTree.cpp
)

target_link_libraries(ito_eval
    PUBLIC
    ito_timing_engine
    ito_data_manager
    ito_config
    ito_tree_build
    salt
)

target_include_directories(ito_eval
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_OPERATION}/iTO/source/data
    ${HOME_OPERATION}/iTO/source/module/solver
)


