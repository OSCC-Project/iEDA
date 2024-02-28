file(GLOB_RECURSE DRC_DB "*.cpp")

add_library(idrc_pro_data ${DRC_DB})

target_link_libraries(idrc_pro_data
    PUBLIC
        
)

target_include_directories(idrc_pro_data
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_DATABASE}/interaction/RT_DRC
)