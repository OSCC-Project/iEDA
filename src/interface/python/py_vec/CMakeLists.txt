set(CMAKE_BUILD_TYPE Debug)


aux_source_directory(. PY_VEC_SRC)

add_library(py_vec ${PY_VEC_SRC})

target_compile_options(py_vec PRIVATE
    $<$<CONFIG:Debug>:-g -O0>
)


target_link_libraries(py_vec
    PUBLIC
        ivec_api
        pybind11::pybind11

)
target_include_directories(py_vec
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)