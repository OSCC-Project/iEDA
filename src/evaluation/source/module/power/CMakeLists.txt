if(DEBUG_EVAL_POWER)
    message(STATUS "evaluator: DEBUG EVAL_POWER")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "evaluator: RELEASE EVAL_POWER")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(eval_power
    ${EVAL_POWER}/PowerEval.cpp
)

target_link_libraries(eval_power
    PUBLIC
        eval_data
)

target_include_directories(eval_power
    PUBLIC
        ${EVAL_POWER}
)
