# iRT-egr
add_library(eval_pro_util_init_egr
    ${EVAL_PRO_UTIL}/init_egr.cpp
)

target_link_libraries(eval_pro_util_init_egr 
    PRIVATE 
        irt_interface
)

target_include_directories(eval_pro_util_init_egr
    PUBLIC
        ${EVAL_PRO_UTIL}
)

# FLUTE
add_library(eval_pro_util_init_flute
    ${EVAL_PRO_UTIL}/init_flute.cpp
)

target_link_libraries(eval_pro_util_init_flute 
    PRIVATE 
        flute
)

target_include_directories(eval_pro_util_init_flute
    PUBLIC
        ${EVAL_PRO_UTIL}
        ${HOME_THIRDPARTY}/flute3
)



