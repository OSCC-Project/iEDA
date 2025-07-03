add_library(pnp-cmd
    CmdRunPnp.cc
    CmdAddVIA1.cc
)

target_link_libraries(pnp-cmd
    PUBLIC
    pnp
)

target_include_directories(pnp-cmd
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
