add_library(ilm_feature
lm_feature.cpp
lm_feature_drc.cpp
lm_feature_timing.cpp
lm_feature_statis.cpp
)

target_link_libraries(ilm_feature
    PUBLIC
    ilm_layout_db
    idm
    idb
    log
    usage
    ilm_util
    tool_api_idrc
)

target_include_directories(ilm_feature
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set(CMAKE_BUILD_TYPE "debug")