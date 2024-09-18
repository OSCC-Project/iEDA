## set path 
set(EVAL_MODULE ${EVAL_SOURCE}/module)
set(EVAL_UTIL ${EVAL_SOURCE}/util)

## build
add_subdirectory(${EVAL_MODULE})
add_subdirectory(${EVAL_UTIL})

add_library(eval_source INTERFACE)

target_link_libraries(eval_source
    INTERFACE
        eval_module
        eval_util
)
