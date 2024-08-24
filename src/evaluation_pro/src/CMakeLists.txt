## set path 
set(EVAL_PRO_MODULE ${EVAL_PRO_SOURCE}/module)
set(EVAL_PRO_UTIL ${EVAL_PRO_SOURCE}/util)

## build
add_subdirectory(${EVAL_PRO_MODULE})
add_subdirectory(${EVAL_PRO_UTIL})

add_library(eval_pro_source INTERFACE)

target_link_libraries(eval_pro_source
    INTERFACE
        eval_pro_module
        eval_pro_util
)
