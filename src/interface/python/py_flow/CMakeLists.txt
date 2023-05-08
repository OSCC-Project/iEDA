aux_source_directory(. PY_FLOW_SRC)

add_library(py_flow ${PY_FLOW_SRC})

target_link_libraries(py_flow
    PUBLIC
        flow
        pybind11::pybind11
)
target_include_directories(py_flow
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
