if(DEBUG_EVAL_CONGESTION)
    message(STATUS "evaluator: DEBUG EVAL_CONGESTION")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "evaluator: RELEASE EVAL_CONGESTION")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(eval_congestion
    ${EVAL_CONGESTION}/CongestionEval.cpp
)

target_link_libraries(eval_congestion
    PUBLIC
        eval_data
)

target_include_directories(eval_congestion
    PUBLIC
        ${EVAL_CONGESTION}
)
