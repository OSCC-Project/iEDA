add_library(eval_pro_timing_eval
    timing_eval.cc
)

target_link_libraries(eval_pro_timing_eval
    PRIVATE
    eval_pro_util_init_ista
)

target_include_directories(eval_pro_timing_eval
    PUBLIC
    ${EVAL_PRO_MODULE}/timing
    ${EVAL_PRO_DATA}
    ${EVAL_PRO_UTIL}
)
