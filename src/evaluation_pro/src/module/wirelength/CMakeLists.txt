add_library(eval_pro_wirelength_eval
    wirelength_eval.cpp
)

target_link_libraries(eval_pro_wirelength_eval 
    PRIVATE 
        eval_pro_util_init_flute     
)

target_include_directories(eval_pro_wirelength_eval
    PUBLIC
        ${EVAL_PRO_MODULE}/wirelength
        ${EVAL_PRO_DATA}
)
