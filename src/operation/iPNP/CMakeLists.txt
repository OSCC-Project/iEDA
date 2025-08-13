cmake_minimum_required(VERSION 3.0)
project(iPNP)
set (CMAKE_CXX_STANDARD 20)

include_directories(${HOME_DATABASE}/manager)
include_directories(${HOME_PLATFORM}/data_manager)
include_directories(${HOME_EVALUATION})
include_directories(${HOME_EVALUATION}/api)

set(IPNP_PROJECT_SOURCE_DIR ${HOME_OPERATION}/iPNP)

include_directories(${IPNP_PROJECT_SOURCE_DIR}/source)

include_directories(${IPNP_PROJECT_SOURCE_DIR}/api)
include_directories(${IPNP_PROJECT_SOURCE_DIR}/source/config)
include_directories(${IPNP_PROJECT_SOURCE_DIR}/source/include)
include_directories(${IPNP_PROJECT_SOURCE_DIR}/source/data_manager)
include_directories(${IPNP_PROJECT_SOURCE_DIR}/source/pnp-cmd)
include_directories(${IPNP_PROJECT_SOURCE_DIR}/source/module/evaluator)
include_directories(${IPNP_PROJECT_SOURCE_DIR}/source/module/optimizer)
include_directories(${IPNP_PROJECT_SOURCE_DIR}/source/module/router)
include_directories(${IPNP_PROJECT_SOURCE_DIR}/source/module/synthesis)
include_directories(${IPNP_PROJECT_SOURCE_DIR}/source/module)
include_directories(SYSTEM ${HOME_THIRDPARTY})

include(${HOME_CMAKE}/operation/idb.cmake)
include(${HOME_CMAKE}/operation/irt.cmake)
include(${HOME_CMAKE}/operation/iplf.cmake)

link_directories(${CMAKE_BINARY_DIR}/lib)

add_subdirectory(api)
# add_subdirectory(data)
add_subdirectory(source)
add_subdirectory(test)

add_executable(iPNP main.cpp)

target_link_libraries(iPNP pnp ipnp_api)

