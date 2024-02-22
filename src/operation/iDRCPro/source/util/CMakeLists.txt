file(GLOB_RECURSE DRC_UTIL "*.cpp")

add_library(idrc_pro_util ${DRC_UTIL})

target_link_libraries(idrc_pro_util
    PUBLIC
    idrc_pro_engine
)

target_include_directories(idrc_pro_util
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)