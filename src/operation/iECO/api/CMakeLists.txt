add_library(ieco_api
    ieco_api.cpp
)

target_link_libraries(ieco_api
    PUBLIC
        idm
        idb
        ieco_src
)

target_include_directories(ieco_api
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
