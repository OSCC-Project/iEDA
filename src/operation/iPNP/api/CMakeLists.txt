add_library(ipnp_api 
iPNPApi.cpp
)

target_link_libraries(ipnp_api 
    PUBLIC
    pnp
)

target_include_directories(ipnp_api
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

