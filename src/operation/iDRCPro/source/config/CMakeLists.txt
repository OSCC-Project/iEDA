add_library(idrc_pro_config
    idrc_config.cpp
)

target_link_libraries(idrc_pro_config
    PUBLIC
    idrc_pro_data
        
)

target_include_directories(idrc_pro_config
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)