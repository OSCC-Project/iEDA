add_library(idrc_pro_dm 
    idrc_dm.cpp
)

target_link_libraries(idrc_pro_dm
    PUBLIC
    idm
    idb
    idrc_pro_data
)

target_include_directories(idrc_pro_dm
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_DATABASE}/interaction/RT_DRC
)

# set(CMAKE_BUILD_TYPE "debug")