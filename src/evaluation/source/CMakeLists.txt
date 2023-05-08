if(DEBUG_EVAL_SOURCE)
    message(STATUS "evaluator: DEBUG EVAL_SOURCE")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "evaluator: RELEASE EVAL_SOURCE")
    set(CMAKE_BUILD_TYPE "Release")
endif()

## set path
set(EVAL_CONFIG ${EVAL_SOURCE}/config)
set(EVAL_MODULE ${EVAL_SOURCE}/module)
set(EVAL_UTIL ${EVAL_SOURCE}/util)

## build
add_subdirectory(${EVAL_CONFIG})
add_subdirectory(${EVAL_MODULE})
add_subdirectory(${EVAL_UTIL})

add_library(eval_source INTERFACE)

target_link_libraries(eval_source
    INTERFACE
        eval_config
        eval_module
        eval_util
)
