if(DEBUG_EVAL_WIRELENGTH)
    message(STATUS "evaluator: DEBUG EVAL_WIRELENGTH")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "evaluator: RELEASE EVAL_WIRELENGTH")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(eval_wirelength
    ${EVAL_WIRELENGTH}/WirelengthEval.cpp
    ${EVAL_WIRELENGTH}/WL.cpp
)

target_link_libraries(eval_wirelength
    PUBLIC
        eval_data
        eval_wrapper
)

target_include_directories(eval_wirelength
    PUBLIC
        ${EVAL_WIRELENGTH}
)
