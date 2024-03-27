add_library(idrc_pro_region_query 
    idrc_region_query.cpp
)

target_link_libraries(idrc_pro_region_query
    PUBLIC
    idrc_pro_data
    idrc_pro_dm
    idm
    idb
)

target_include_directories(idrc_pro_region_query
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)