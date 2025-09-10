include_directories(${HOME_PLATFORM}/data_manager)

add_library(pnp-synthesis
    NetworkSynthesis.cpp
    PowerRouter.cpp
    PowerVia.cpp
)

target_link_libraries(pnp-synthesis
    PUBLIC
    pnp-config
    idb
    idm
    pnp-data
    log
)

target_include_directories(pnp-synthesis
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
