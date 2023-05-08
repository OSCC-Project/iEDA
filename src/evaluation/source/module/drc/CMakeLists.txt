if(DEBUG_EVAL_DRC)
    message(STATUS "evaluator: DEBUG EVAL_DRC")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "evaluator: RELEASE EVAL_DRC")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(eval_drc
    ${EVAL_DRC}/DRCEval.cpp
)

target_link_libraries(eval_drc
    PUBLIC
        eval_data
)

target_include_directories(eval_drc
    PUBLIC
        ${EVAL_DRC}
)
