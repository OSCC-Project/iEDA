add_library(eval_third_party_libs INTERFACE)
target_include_directories(eval_third_party_libs
    SYSTEM INTERFACE
        ${HOME_THIRDPARTY}
        ${HOME_THIRDPARTY}/flute3
        ${HOME_THIRDPARTY}/magic_enum
)
