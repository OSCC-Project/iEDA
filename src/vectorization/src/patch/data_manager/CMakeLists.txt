add_library(ivec_patch_dm
    vec_patch_dm.cpp
    vec_patch_init.cpp
)

target_link_libraries(ivec_patch_dm
    PUBLIC
    idb
    idm
    geometry_db
    log
    usage
    ivec_db
    ivec_patch_db
    ivec_layout_db
    ivec_util
)

target_include_directories(ivec_patch_dm
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_UTILITY}/json
    ${OpenCV_INCLUDE_DIRS}
)

include_directories(SYSTEM ${Boost_INCLUDE_DIRS})