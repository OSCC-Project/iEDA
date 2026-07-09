include_directories(${PROJECT_SOURCE_DIR}/src/third_party/abseil-cpp ${PROJECT_SOURCE_DIR}/src/database/manager/parser ${PROJECT_SOURCE_DIR}/src/operation/iSTA/source/module/include ${PROJECT_SOURCE_DIR}/src/operation/iSTA/source/module)
add_compile_options(-include ${CMAKE_CURRENT_SOURCE_DIR}/cts_lib_bridge.hh)
message(STATUS "############## CTS: START CMAKE ##############")

# Debug options
# source/database
option(DEBUG_ICTS_SOURCE_DATABASE_CONFIG "source database config" OFF)
option(DEBUG_ICTS_SOURCE_DATABASE_DESIGN "source database design" OFF)
option(DEBUG_ICTS_SOURCE_DATABASE_IO "source database io" OFF)
option(DEBUG_ICTS_SOURCE_DATABASE_SPATIAL "source database spatial" OFF)
option(DEBUG_ICTS_SOURCE_DATABASE "source database" OFF)

# source/utils
option(DEBUG_ICTS_SOURCE_UTILS_LOGGER "source utils logger" OFF)
option(DEBUG_ICTS_SOURCE_UTILS_GEOMETRY "source utils geometry" OFF)
option(DEBUG_ICTS_SOURCE_UTILS "source utils" OFF)

# source/flow
option(DEBUG_ICTS_SOURCE_FLOW_SYNTHESIS_REALIZATION "source flow synthesis realization" OFF)
option(DEBUG_ICTS_SOURCE_FLOW "source flow" OFF)

# source/module
option(DEBUG_ICTS_SOURCE_MODULE_TOPOLOGY_KMEANS "source module topology kmeans" OFF)
option(DEBUG_ICTS_SOURCE_MODULE_TOPOLOGY_MCF "source module topology mcf" OFF)
option(DEBUG_ICTS_SOURCE_MODULE_TOPOLOGY_CLUSTERING "source module topology clustering" OFF)
option(DEBUG_ICTS_SOURCE_MODULE_TOPOLOGY "source module topology" OFF)
option(DEBUG_ICTS_SOURCE_MODULE "source module" OFF)

# source
option(DEBUG_ICTS_SOURCE "source" OFF)

# api
option(DEBUG_ICTS_API "api" OFF)

# test
option(DEBUG_ICTS_TEST "test" OFF)

# Path setting
set(HOME_ICTS ${HOME_OPERATION}/iCTS)

set(ICTS_EXTERNAL_LIBS ${HOME_ICTS}/external_libs)
set(ICTS_API ${HOME_ICTS}/api)
set(ICTS_SOURCE ${HOME_ICTS}/source)
set(ICTS_TEST ${HOME_ICTS}/test)

add_subdirectory(${ICTS_EXTERNAL_LIBS})
add_subdirectory(${ICTS_SOURCE})
add_subdirectory(${ICTS_API})
add_subdirectory(${ICTS_TEST})

message(STATUS "############## CTS: END CMAKE ##############")
