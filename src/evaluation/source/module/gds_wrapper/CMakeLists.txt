if(DEBUG_EVAL_GDS_WRAPPER)
    message(STATUS "evaluator: DEBUG EVAL_GDS_WRAPPER")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "evaluator: RELEASE EVAL_GDS_WRAPPER")
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_library(eval_gds_wrapper
    ${EVAL_GDS_WRAPPER}/GDSWrapper.cpp
)

target_link_libraries(eval_gds_wrapper
    PUBLIC
        eval_data
)

target_include_directories(eval_gds_wrapper
    PUBLIC
        ${EVAL_GDS_WRAPPER}
)
