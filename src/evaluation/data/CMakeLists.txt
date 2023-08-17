add_library(eval_data
    ${EVAL_DATA}/CongBin.cpp
    ${EVAL_DATA}/TimingNet.cpp
    ${EVAL_DATA}/WLNet.cpp
)

target_link_libraries(eval_data
    PUBLIC
        eval_util
        eval_source_external_libs
        eval_api_external_libs
)

target_include_directories(eval_data
    PUBLIC
        ${EVAL_DATA}
)
