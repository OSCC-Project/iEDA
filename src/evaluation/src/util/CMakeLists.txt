# set(CMAKE_BUILD_TYPE "Release")

# general ops
add_library(eval_util_general_ops
    ${EVAL_UTIL}/general_ops.cpp
)

target_link_libraries(eval_util_general_ops
    PRIVATE
        idm
)

target_include_directories(eval_util_general_ops
    PUBLIC
        ${EVAL_UTIL}
)

# iRT-egr
add_library(eval_util_init_egr
    ${EVAL_UTIL}/init_egr.cpp
)

target_link_libraries(eval_util_init_egr
    PRIVATE
        irt_interface
        idm
)

target_include_directories(eval_util_init_egr
    PUBLIC
        ${EVAL_UTIL}
)

# FLUTE
add_library(eval_util_init_flute
    ${EVAL_UTIL}/init_flute.cpp
)

target_link_libraries(eval_util_init_flute
    PRIVATE
        flute
)

target_include_directories(eval_util_init_flute
    PUBLIC
        ${EVAL_UTIL}
        ${HOME_THIRDPARTY}/flute3
)

# iSTA
add_library(eval_util_init_ista
    ${EVAL_UTIL}/init_sta.cc
)

target_link_libraries(eval_util_init_ista
    PRIVATE
        ieda_feature
        idm
        ivec_api
        irt_interface
        ista-engine
        power
        salt
        ivec_layout_db
)

target_include_directories(eval_util_init_ista
    PUBLIC
        ${EVAL_UTIL}
        ${PROJECT_SOURCE_DIR}/src/third_party/salt
)


# Wirelength LUT
add_library(eval_util_wirelength_lut
    ${EVAL_UTIL}/wirelength_lut.cpp
)

target_include_directories(eval_util_wirelength_lut
    PUBLIC
        ${EVAL_UTIL}
)


# iDB
add_library(eval_util_init_idb
    ${EVAL_UTIL}/init_idb.cpp
)

target_link_libraries(eval_util_init_idb
    PRIVATE
        idm
        eval_db
)

target_include_directories(eval_util_init_idb
    PUBLIC
        ${EVAL_UTIL}
)
