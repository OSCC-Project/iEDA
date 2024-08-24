# 拥塞评估API
add_library(eval_pro_congestion_api
    ${EVAL_PRO_API}/congestion_api.cpp
)

target_link_libraries(eval_pro_congestion_api 
    PRIVATE 
        eval_pro_db
        eval_pro_congestion_eval
)

target_include_directories(eval_pro_congestion_api 
    PUBLIC
        ${EVAL_PRO_API}
)

# 线长评估API
add_library(eval_pro_wirelength_api
    ${EVAL_PRO_API}/wirelength_api.cpp
)

target_link_libraries(eval_pro_wirelength_api 
    PRIVATE 
        eval_pro_db
        eval_pro_wirelength_eval
)

target_include_directories(eval_pro_wirelength_api 
    PUBLIC
        ${EVAL_PRO_API}
)
