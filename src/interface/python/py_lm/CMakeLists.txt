aux_source_directory(. PY_LM_SRC)

add_library(py_lm ${PY_LM_SRC})

target_link_libraries(py_lm
    PUBLIC
        ilm_api
        pybind11::pybind11

)
target_include_directories(py_lm
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)