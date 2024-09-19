add_library(eval_wirelength_eval
    wirelength_eval.cpp
)

target_link_libraries(eval_wirelength_eval 
    PRIVATE 
        eval_util_init_flute  
        eval_util_init_egr   
        eval_util_init_idb
)

target_include_directories(eval_wirelength_eval
    PUBLIC
        ${EVAL_MODULE}/wirelength
        ${EVAL_DATA}
)
