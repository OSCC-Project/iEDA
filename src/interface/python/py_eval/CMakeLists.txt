aux_source_directory(. PY_EVAL_SRC)

add_library(py_eval ${PY_EVAL_SRC})

target_link_libraries(py_eval
    PUBLIC
        eval_wirelength_api
        eval_congestion_api
        eval_density_api
        eval_timing_api
        pybind11::pybind11

)
target_include_directories(py_eval
    PUBLIC
    ${EVAL_DATA}
    ${CMAKE_CURRENT_SOURCE_DIR}
)