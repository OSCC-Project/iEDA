aux_source_directory(. PY_IPL_SRC)

add_library(py_ipl ${PY_IPL_SRC})

target_link_libraries(py_ipl
    PUBLIC
        tool_manager
        ipl-api
        pybind11::pybind11
)
target_include_directories(py_ipl
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)