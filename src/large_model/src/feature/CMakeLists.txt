add_library(ilm_feature
    lm_feature.cpp
    lm_feature_drc.cpp
    lm_feature_timing.cpp
    lm_feature_statis.cpp
)

target_link_libraries(ilm_feature
    PUBLIC
    ilm_layout_db
    ilm_patch_db
    ilm_patch_dm
    idm
    idb
    log
    usage
    ilm_util
    tool_api_idrc
    eval_congestion_api
    eval_density_api
    eval_wirelength_api
)

set(EVALUTION_INCLUDE ${HOME_EVALUATION}/src/util)

target_include_directories(ilm_feature
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EVALUTION_INCLUDE}
    ${HOME_DATABASE}/interaction/RT_DRC
)