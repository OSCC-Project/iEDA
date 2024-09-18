add_library(eval_pro_density_eval
    density_eval.cpp
)

target_link_libraries(eval_pro_density_eval 
    PRIVATE 
        eval_pro_util_general_ops     
        eval_pro_util_init_idb        
)

target_include_directories(eval_pro_density_eval
    PUBLIC
        ${EVAL_PRO_MODULE}/density
        ${EVAL_PRO_DATA}
)