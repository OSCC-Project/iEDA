add_library(ipdn_db
    ipdn_basic.cpp
    
)

target_link_libraries(ipdn_db 
    PUBLIC
    idm
)

target_include_directories(ipdn_db 
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)