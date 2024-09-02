# set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_BUILD_TYPE "Debug")

# general ops
add_library(eval_pro_util_general_ops
    ${EVAL_PRO_UTIL}/general_ops.cpp
)

target_include_directories(eval_pro_util_general_ops
    PUBLIC
        ${EVAL_PRO_UTIL}
)

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

# iSTA
add_library(eval_pro_util_init_ista
    ${EVAL_PRO_UTIL}/init_sta.cc
)

target_link_libraries(eval_pro_util_init_ista
    PRIVATE
        ieda_feature
        idm
        irt_interface
        ista-engine
        power
        salt
)

target_include_directories(eval_pro_util_init_ista
    PUBLIC
        ${EVAL_PRO_UTIL}
        ${PROJECT_SOURCE_DIR}/src/third_party/salt
)


# Wirelength LUT
add_library(eval_pro_util_wirelength_lut
    ${EVAL_PRO_UTIL}/wirelength_lut.cpp
)

target_include_directories(eval_pro_util_wirelength_lut
    PUBLIC
        ${EVAL_PRO_UTIL}
)


# iDB
add_library(eval_pro_util_init_idb
    ${EVAL_PRO_UTIL}/init_idb.cpp
)

target_link_libraries(eval_pro_util_init_idb
    PRIVATE
        idm
)

target_include_directories(eval_pro_util_init_idb
    PUBLIC
        ${EVAL_PRO_UTIL}
)
