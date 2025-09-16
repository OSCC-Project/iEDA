
add_library(ipdn_via
    pdn_via.cpp
)

target_link_libraries(ipdn_via 
    PUBLIC
    idm
    idb
)

target_include_directories(ipdn_via 
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_OPERATION}/iPDN/source/data
)