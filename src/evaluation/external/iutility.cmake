add_library(eval_iutility_libs INTERFACE)
target_link_libraries(eval_iutility_libs
    INTERFACE
        usage
)

target_include_directories( eval_iutility_libs
    INTERFACE
        ${HOME_UTILITY}
)
