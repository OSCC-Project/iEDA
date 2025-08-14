add_library(pnp-data
    PNPGridManager.cpp
    SingleTemplate.cpp
    TemplateLib.cpp
    PNPIdbWrapper.cpp
)

target_link_libraries(pnp-data
    PUBLIC
    idb
    pnp-synthesis
    pnp-config
    log
    idm
)

target_include_directories(pnp-data
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)