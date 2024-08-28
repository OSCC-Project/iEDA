add_library(eval_pro_density_eval
    density_eval.cpp
)

target_include_directories(eval_pro_density_eval
    PUBLIC
        ${EVAL_PRO_MODULE}/density
        ${EVAL_PRO_DATA}
)