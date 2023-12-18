if(DEBUG_EVAL_WRAPPER)
    message(STATUS "evaluator: DEBUG EVAL_WRAPPER")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "evaluator: RELEASE EVAL_WRAPPER")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(eval_wrapper
    ${EVAL_WRAPPER}/DBWrapper.cpp
)

target_link_libraries(eval_wrapper
    PUBLIC
        eval_config
        eval_data
)

target_include_directories(eval_wrapper
    PUBLIC
        ${EVAL_WRAPPER}
        ${EVAL_WRAPPER}/database
)
