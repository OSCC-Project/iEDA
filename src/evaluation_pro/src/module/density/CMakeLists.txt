add_library(eval_density_eval
    density_eval.cpp
)

target_link_libraries(eval_density_eval 
    PRIVATE 
        eval_util_general_ops     
        eval_util_init_idb        
)

target_include_directories(eval_density_eval
    PUBLIC
        ${EVAL_MODULE}/density
        ${EVAL_DATA}
)