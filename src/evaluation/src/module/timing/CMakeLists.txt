add_library(eval_timing_eval
    timing_eval.cc
)

target_link_libraries(eval_timing_eval
    PRIVATE
    eval_util_init_ista
    eval_db
)

target_include_directories(eval_timing_eval
    PUBLIC
    ${EVAL_MODULE}/timing
    ${EVAL_DATA}
    ${EVAL_UTIL}
)
