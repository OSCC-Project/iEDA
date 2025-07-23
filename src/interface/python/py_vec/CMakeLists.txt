aux_source_directory(. PY_VEC_SRC)

add_library(py_vec ${PY_VEC_SRC})

target_link_libraries(py_vec
    PUBLIC
        ivec_api
        pybind11::pybind11

)
target_include_directories(py_vec
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)