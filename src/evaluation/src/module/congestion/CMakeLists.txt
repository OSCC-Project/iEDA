add_library(eval_congestion_eval
    congestion_eval.cpp
)

target_link_libraries(eval_congestion_eval 
    PRIVATE 
        idm
        eval_util_init_egr     
        eval_util_wirelength_lut
        eval_util_general_ops 
        eval_util_init_idb
)

target_include_directories(eval_congestion_eval
    PUBLIC
        ${EVAL_MODULE}/congestion
        ${EVAL_DATA}
)
