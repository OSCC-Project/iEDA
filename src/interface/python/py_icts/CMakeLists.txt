aux_source_directory(. PY_ICTS_SRC)

add_library(py_icts ${PY_ICTS_SRC})

target_link_libraries(py_icts
    PUBLIC
        icts_api
        tool_manager
        pybind11::pybind11
)
target_include_directories(py_icts
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
