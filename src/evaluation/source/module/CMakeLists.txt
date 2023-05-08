if(DEBUG_EVAL_MODULE)
    message(STATUS "evaluator: DEBUG EVAL_MODULE")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "evaluator: RELEASE EVAL_MODULE")
    set(CMAKE_BUILD_TYPE "Release")
endif()

## set path
set(EVAL_CONGESTION ${EVAL_MODULE}/congestion)
set(EVAL_DRC ${EVAL_MODULE}/drc)
set(EVAL_GDS_WRAPPER ${EVAL_MODULE}/gds_wrapper)
set(EVAL_POWER ${EVAL_MODULE}/power)
set(EVAL_TIMING ${EVAL_MODULE}/timing)
set(EVAL_WIRELENGTH ${EVAL_MODULE}/wirelength)
set(EVAL_WRAPPER ${EVAL_MODULE}/wrapper)

## build
add_subdirectory(${EVAL_CONGESTION})
add_subdirectory(${EVAL_DRC})
add_subdirectory(${EVAL_GDS_WRAPPER})
add_subdirectory(${EVAL_POWER})
add_subdirectory(${EVAL_TIMING})
add_subdirectory(${EVAL_WIRELENGTH})
add_subdirectory(${EVAL_WRAPPER})

add_library(eval_module
    ${EVAL_MODULE}/manager.cpp
)

target_link_libraries(eval_module
    PUBLIC
        eval_congestion
        eval_drc
        eval_gds_wrapper
        eval_power
        eval_timing
        eval_wirelength
        eval_wrapper
)

target_include_directories(eval_module
    PUBLIC
        ${EVAL_MODULE}
)
