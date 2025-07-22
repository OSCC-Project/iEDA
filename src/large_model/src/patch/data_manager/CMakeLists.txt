add_library(ilm_patch_dm
    lm_patch_dm.cpp
    lm_patch_init.cpp
)

# find_package(OpenCV REQUIRED)

target_link_libraries(ilm_patch_dm
    PUBLIC
    idb
    idm
    geometry_db
    log
    usage
    ilm_db
    ilm_patch_db
    ilm_layout_db
    ilm_util
    # ${OpenCV_LIBS}
)

target_include_directories(ilm_patch_dm
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_UTILITY}/json
    ${OpenCV_INCLUDE_DIRS}
)

include_directories(SYSTEM ${Boost_INCLUDE_DIRS})