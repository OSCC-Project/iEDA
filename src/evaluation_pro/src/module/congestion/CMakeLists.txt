add_library(eval_pro_congestion_eval
    congestion_eval.cpp
)

target_link_libraries(eval_pro_congestion_eval 
    PRIVATE 
        eval_pro_util_init_egr     
        eval_pro_util_wirelength_lut
)

target_include_directories(eval_pro_congestion_eval
    PUBLIC
        ${EVAL_PRO_MODULE}/congestion
        ${EVAL_PRO_DATA}
)
