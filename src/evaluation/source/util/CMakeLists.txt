add_library(eval_util
    ${EVAL_UTIL}/EvalLog.cpp
)

target_include_directories(eval_util
    PUBLIC
        ${EVAL_UTIL}
)
