add_library(ivec_feature
    vec_feature.cpp
    vec_feature_drc.cpp
    vec_feature_timing.cpp
    vec_feature_statis.cpp
)

target_link_libraries(ivec_feature
    PUBLIC
    ivec_layout_db
    ivec_patch_db
    ivec_patch_dm
    idm
    idb
    log
    usage
    ivec_util
    tool_api_idrc
    eval_congestion_api
    eval_density_api
    eval_wirelength_api
)

set(EVALUTION_INCLUDE ${HOME_EVALUATION}/src/util)

target_include_directories(ivec_feature
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EVALUTION_INCLUDE}
    ${HOME_DATABASE}/interaction/RT_DRC
)