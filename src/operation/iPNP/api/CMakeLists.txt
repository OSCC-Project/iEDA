add_library(ipnp_api 
ipnp_api.cpp
)

target_link_libraries(ipnp_api 
    PUBLIC
    pnp
)

target_include_directories(ipnp_api
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

