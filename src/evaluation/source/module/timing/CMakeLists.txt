set(DEBUG_EVAL_TIMING ON)
if(DEBUG_EVAL_TIMING)
    message(STATUS "evaluator: DEBUG EVAL_TIMING")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "evaluator: RELEASE EVAL_TIMING")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(eval_timing
    ${EVAL_TIMING}/TimingEval.cpp
    ${EVAL_DATA}/TimingNet.cpp
)

target_link_libraries(eval_timing
    PUBLIC
        eval_data
        ista-engine 
)

target_include_directories(eval_timing
    PUBLIC
        ${EVAL_TIMING}
)
