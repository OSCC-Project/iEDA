if(DEBUG_EVAL_CONFIG)
    message(STATUS "evaluator: DEBUG EVAL_CONFIG")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "evaluator: RELEASE EVAL_CONFIG")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(eval_config
    ${EVAL_CONFIG}/Config.cpp
)

target_link_libraries(eval_config
    PUBLIC
        eval_util
        eval_source_external_libs
)

target_include_directories(eval_config
    PUBLIC
        ${EVAL_CONFIG}
)
