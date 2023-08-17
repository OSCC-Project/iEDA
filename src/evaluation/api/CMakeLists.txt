if(DEBUG_EVAL_API)
    message(STATUS "evaluator: DEBUG EVAL_API")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "evaluator: RELEASE EVAL_API")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(eval_api
    ${EVAL_API}/EvalAPI.cpp
)

target_link_libraries(eval_api
    PUBLIC
        eval_data
        eval_source
        eval_wirelength
        irt_api     
)

target_include_directories(eval_api
    PUBLIC
        ${EVAL_API}
)
